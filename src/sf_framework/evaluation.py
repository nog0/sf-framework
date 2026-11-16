"""Score a finished episode independently of its execution backend."""

from __future__ import annotations

from pathlib import PurePosixPath

from sf_framework.config import EvaluationConfig
from sf_framework.core import EvaluationInput, Grade


class EnvironmentEvaluator:
    def __init__(self, config: EvaluationConfig):
        if config.options:
            raise ValueError("evaluation.kind=environment does not take options")

    def metadata(self, task_metadata: dict) -> dict:
        return {"score_range": task_metadata.get("score_range")}

    def evaluate(self, episode: EvaluationInput) -> Grade:
        if episode.environment_grade is None:
            raise ValueError("This execution backend did not provide an environment grade")
        return episode.environment_grade


class FileTextEvaluator:
    def __init__(self, config: EvaluationConfig):
        options = config.options
        if set(options) != {"path", "expected"} or not all(
            isinstance(value, str) for value in options.values()
        ):
            raise ValueError("evaluation.kind=file_text requires string options path and expected")
        self.path = PurePosixPath(options["path"])
        if self.path.is_absolute() or ".." in self.path.parts or not self.path.parts:
            raise ValueError("evaluation.options.path must be a file relative to the workspace")
        self.expected = options["expected"].strip()

    def metadata(self, task_metadata: dict) -> dict:
        return {"score_range": [0, 1], "comparison": "UTF-8 text with outer whitespace trimmed"}

    def evaluate(self, episode: EvaluationInput) -> Grade:
        if episode.workspace is None:
            raise ValueError("evaluation.kind=file_text requires a workspace")
        workspace = episode.workspace.resolve()
        details = {"path": str(self.path)}
        try:
            path = (workspace / self.path).resolve()
            if not path.is_relative_to(workspace) or not path.is_file():
                return Grade(0.0, False, {**details, "check": "missing_or_invalid_file"})
            with path.open("rb") as stream:
                value = stream.read(1_048_577)
            if len(value) > 1_048_576:
                return Grade(0.0, False, {**details, "check": "file_too_large"})
            matched = value.decode("utf-8").strip() == self.expected
        except (OSError, RuntimeError, UnicodeError):
            return Grade(0.0, False, {**details, "check": "unreadable_file"})
        return Grade(
            float(matched), matched, {**details, "check": "match" if matched else "mismatch"}
        )
