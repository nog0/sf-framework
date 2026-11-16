"""Run and grade SpreadsheetBench 2 workbook tasks with the upstream evaluator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from sf_framework.config import EvaluationConfig, ExecutionConfig, TaskConfig
from sf_framework.core import EvaluationInput, Grade, fingerprint, sha256_file
from sf_framework.execution import WorkspaceExecution, resolve_docker_image

IMAGE = "sf-spreadsheetbench2:a7bdc2d"
CATEGORIES = ("Debugging", "Financial_Model", "Template")

# Invoke the official UNO batch function directly so inputs and goldens can be
# recalculated too. Its directory CLI only selects files ending in output.xlsx.
_RECALCULATE = """
from pathlib import Path
import open_spreadsheet as upstream

def recalculate(names):
    paths = [Path(name).resolve() for name in names]
    before = {path: path.stat().st_ino for path in paths}
    if not upstream.start_libreoffice_service():
        raise RuntimeError("Could not start the official LibreOffice service")
    try:
        upstream.batch_open_files([str(path) for path in paths])
    finally:
        upstream.stop_libreoffice_service()
    return {path.name for path in paths if path.stat().st_ino == before[path]}
"""
_PREPARE_INPUT = (
    _RECALCULATE
    + """
if recalculate(["input.xlsx"]):
    raise RuntimeError("Input workbook recalculation failed")
"""
)
_GRADE = (
    _RECALCULATE
    + """
import json
from evaluation import process_single_item

request = json.loads(Path("request.json").read_text())
data = request["case"]
failed = recalculate(["golden.xlsx", "submission.xlsx"])
if "golden.xlsx" in failed:
    raise RuntimeError("Reference workbook recalculation failed")
if "submission.xlsx" in failed:
    report = {
        "id": data["id"],
        "error_message": "Submitted workbook could not be recalculated",
        "regression_accuracy": 0.0,
        "modification_accuracy": 0.0,
        "accuracy": 0.0,
        "missing_output": False,
    }
else:
    dataset = Path("dataset")
    for field, name in (("spreadsheet_path", "input.xlsx"), ("golden_response_path", "golden.xlsx")):
        path = dataset / data[field]
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(name).rename(path)
    outputs = Path("outputs")
    outputs.mkdir()
    Path("submission.xlsx").rename(outputs / (str(data["id"]) + "_output.xlsx"))
    report, missing = process_single_item(
        data, dataset, outputs, {str(data["id"]): request["model_calls"]}, request["category"]
    )
    report["missing_output"] = missing
Path("report.json").write_text(json.dumps(report, ensure_ascii=False))
"""
)


@dataclass(frozen=True)
class WorkbookCase:
    category: str
    row: dict
    input_file: Path
    golden_file: Path

    @property
    def output_name(self) -> str:
        return f"{self.row['id']}_output.xlsx"


class SpreadsheetBench2Suite:
    def __init__(self, config: TaskConfig):
        if unknown := set(config.options) - {"categories"}:
            raise ValueError(f"Unknown SpreadsheetBench 2 task options: {sorted(unknown)}")
        categories = config.options.get("categories", list(CATEGORIES))
        if (
            not isinstance(categories, list)
            or not categories
            or any(category not in CATEGORIES for category in categories)
            or len(set(categories)) != len(categories)
        ):
            raise ValueError(f"Select distinct SpreadsheetBench 2 categories from {CATEGORIES}")
        root = Path(config.root).resolve()
        self.config = config
        self.dataset = root / "spreadsheetbench-v2"
        self.evaluation_dir = root / "upstream" / "evaluation"
        self.cases: dict[str, WorkbookCase] = {}
        self.indexes: list[Path] = []
        for category in categories:
            directory = self.dataset / category
            index = directory / "dataset.json"
            self.indexes.append(index)
            for row in json.loads(index.read_text(encoding="utf-8")):
                case_id = str(row["id"])
                if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
                    raise ValueError(f"Invalid SpreadsheetBench 2 task ID: {case_id!r}")
                task_id = f"{category}/{case_id}"
                if task_id in self.cases:
                    raise ValueError(f"Duplicate SpreadsheetBench 2 task ID: {task_id}")
                self.cases[task_id] = WorkbookCase(
                    category,
                    row,
                    self._workbook_path(directory, row["spreadsheet_path"]),
                    self._workbook_path(directory, row["golden_response_path"]),
                )

    @staticmethod
    def _workbook_path(directory: Path, relative: str) -> Path:
        name = Path(relative)
        path = (directory / name).resolve()
        if name.is_absolute() or ".." in name.parts or not path.is_relative_to(directory):
            raise ValueError(f"Workbook must be inside its dataset category: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def task_ids(self) -> list[str]:
        return list(self.cases)

    def inputs(self, task_id: str) -> Path:
        return self.cases[task_id].input_file

    def instruction(self, task_id: str) -> str:
        case = self.cases[task_id]
        return (
            f"{case.row['instruction']}\n\n"
            f"Input workbook: /workspace/input/{case.input_file.name}\n"
            f"Output workbook: /workspace/{case.output_name}\n"
            "Preserve the existing workbook structure and formatting. "
            "Change only the cells required by the task."
        )

    def metadata(self) -> dict:
        workbooks = {
            path for case in self.cases.values() for path in (case.input_file, case.golden_file)
        }
        return {
            "name": "spreadsheetbench2",
            "split": self.config.split,
            "indexes_sha256": {path.parent.name: sha256_file(path) for path in self.indexes},
            "workbooks_sha256": fingerprint(
                {
                    str(path.relative_to(self.dataset)): sha256_file(path)
                    for path in sorted(workbooks)
                }
            ),
            "upstream_sha256": {
                name: sha256_file(self.evaluation_dir / name)
                for name in ("open_spreadsheet.py", "evaluation.py")
            },
        }


def _run_workbook_job(docker: str, image_id: str, directory: Path, upstream: Path, script: str):
    name = f"sf-spreadsheet-{uuid.uuid4().hex}"
    command = [
        docker,
        "run",
        "--rm",
        "--name",
        name,
        "--pull=never",
        "--network=none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONPATH=/sf_upstream:/usr/lib/libreoffice/program:/usr/lib/python3/dist-packages",
        "--mount",
        f"type=bind,source={upstream},target=/sf_upstream,readonly",
        "--mount",
        f"type=bind,source={directory},target=/job",
        "--workdir",
        "/job",
        "--entrypoint",
        "python",
        image_id,
        "-c",
        script,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except BaseException:
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=15)
        raise
    if result.returncode:
        diagnostic = (result.stdout + "\n" + result.stderr).strip()[-3000:]
        raise RuntimeError(f"SpreadsheetBench 2 workbook job failed: {diagnostic}")


class SpreadsheetBench2Execution(WorkspaceExecution):
    def __init__(self, config: ExecutionConfig):
        super().__init__(ExecutionConfig(config.kind, {"image": IMAGE, **config.options}))

    def stage_inputs(self, suite, task_id, workspace: Path) -> list[str]:
        input_dir = workspace / "input"
        input_dir.mkdir()
        source = suite.inputs(task_id)
        staged = input_dir / "input.xlsx"
        shutil.copyfile(source, staged)
        _run_workbook_job(
            self.docker, self.image_id, input_dir, suite.evaluation_dir, _PREPARE_INPUT
        )
        staged.rename(input_dir / source.name)
        return ["--mount", f"type=bind,source={input_dir},target=/workspace/input,readonly"]


class SpreadsheetBench2Evaluator:
    def __init__(self, config: EvaluationConfig):
        if unknown := set(config.options) - {"image"}:
            raise ValueError(f"Unknown SpreadsheetBench 2 evaluation options: {sorted(unknown)}")
        self.image = config.options.get("image", IMAGE)
        self.docker, self.image_id = resolve_docker_image(self.image)

    def metadata(self, task_metadata: dict) -> dict:
        return {"score_range": [0, 1], "grader": "upstream", "image_id": self.image_id}

    def evaluate(self, episode: EvaluationInput) -> Grade:
        suite = episode.task_suite
        if not isinstance(suite, SpreadsheetBench2Suite) or episode.workspace is None:
            raise ValueError("SpreadsheetBench 2 grading requires its task suite and workspace")
        case = suite.cases[episode.task_id]
        submission = episode.workspace / case.output_name
        if not submission.is_file() or submission.is_symlink():
            return Grade(
                0.0, False, {"missing_output": True, "error_message": "output file not exist"}
            )

        with TemporaryDirectory(prefix="sf-spreadsheet-grade-") as temp:
            job = Path(temp)
            shutil.copyfile(episode.workspace / "input" / case.input_file.name, job / "input.xlsx")
            shutil.copyfile(case.golden_file, job / "golden.xlsx")
            shutil.copyfile(submission, job / "submission.xlsx")
            (job / "request.json").write_text(
                json.dumps(
                    {
                        "case": case.row,
                        "category": case.category,
                        "model_calls": episode.agent.model_calls or 0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            _run_workbook_job(self.docker, self.image_id, job, suite.evaluation_dir, _GRADE)
            details = json.loads((job / "report.json").read_text(encoding="utf-8"))
        return Grade(float(details["accuracy"]), details["accuracy"] == 1.0, details)
