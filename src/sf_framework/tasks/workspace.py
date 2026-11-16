"""Task instructions and input files for sandboxed workspace episodes."""

from __future__ import annotations

from pathlib import Path

from sf_framework.config import TaskConfig
from sf_framework.core import fingerprint, sha256_file


class WorkspaceSuite:
    def __init__(self, config: TaskConfig):
        self.config = config
        root = Path(config.root).resolve()
        if not root.is_dir():
            raise ValueError(f"Workspace task root is not a directory: {root}")
        directories = [root] if (root / "instruction.md").is_file() else sorted(root.iterdir())
        self.tasks = {
            path.name: path
            for path in directories
            if path.is_dir() and not path.is_symlink() and (path / "instruction.md").is_file()
        }
        if not self.tasks:
            raise ValueError(f"No workspace tasks with instruction.md found at {root}")

    def task_ids(self) -> list[str]:
        return list(self.tasks)

    def instruction(self, task_id: str) -> str:
        instruction = (self.tasks[task_id] / "instruction.md").read_text(encoding="utf-8").strip()
        if not instruction:
            raise ValueError(f"Workspace task {task_id!r} has an empty instruction")
        return instruction

    def inputs(self, task_id: str) -> Path:
        return self.tasks[task_id] / "input"

    def metadata(self) -> dict:
        hashes = {}
        for task_id, task in self.tasks.items():
            inputs = self.inputs(task_id)
            if inputs.exists() and not inputs.is_dir():
                raise ValueError(f"Workspace input must be a directory: {inputs}")
            files = [task / "instruction.md", inputs, *sorted(inputs.rglob("*"))]
            entries = {}
            for path in files:
                if path.is_symlink() or (path.exists() and not (path.is_file() or path.is_dir())):
                    raise ValueError(f"Unsupported workspace input: {path}")
                if path.exists():
                    entries[str(path.relative_to(task))] = (
                        sha256_file(path) if path.is_file() else None
                    )
            hashes[task_id] = fingerprint(entries)
        return {"name": "workspace", "split": self.config.split, "tasks_sha256": hashes}
