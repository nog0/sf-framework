"""Discover SkillFlow's family-organized Harbor task packages."""

from __future__ import annotations

import json
from pathlib import Path

from sf_framework.config import TaskConfig
from sf_framework.core import fingerprint, sha256_file


class SkillFlowSuite:
    def __init__(self, config: TaskConfig):
        self.config = config
        root = Path(config.root).resolve()
        if not root.is_dir():
            raise ValueError(f"SkillFlow task root does not exist: {root}")

        if self._is_task(root):
            candidates = [root]
        else:
            candidates = sorted(
                path.parent for path in root.rglob("task.toml") if self._is_task(path.parent)
            )
        families: dict[Path, list[Path]] = {}
        for task in candidates:
            families.setdefault(task.parent, []).append(task)

        self.tasks: dict[str, Path] = {}
        self.families: dict[str, list[str]] = {}
        for family, tasks in families.items():
            ids = []
            for task in self._ordered_tasks(family, tasks):
                task_id = f"{family.name}/{task.name}"
                if task_id in self.tasks:
                    raise ValueError(f"Duplicate SkillFlow task ID: {task_id}")
                ids.append(task_id)
                self.tasks[task_id] = task
            self.families.setdefault(family.name, []).extend(ids)

        if not self.tasks:
            raise ValueError(f"No complete SkillFlow task packages found in {root}")

    @staticmethod
    def _is_task(path: Path) -> bool:
        return all(
            (path / name).is_file() for name in ("task.toml", "instruction.md", "tests/test.sh")
        )

    @staticmethod
    def _ordered_tasks(family: Path, tasks: list[Path]) -> list[Path]:
        ranking = family / "ALL_TASK_DIFFICULTY_RANKING.json"
        if not ranking.is_file():
            return tasks
        names = json.loads(ranking.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Invalid SkillFlow task order: {ranking}")
        names = [name.strip() for name in names]
        rank = {name: index for index, name in enumerate(dict.fromkeys(names))}
        return sorted(tasks, key=lambda task: (rank.get(task.name, len(rank)), task.name))

    def task_ids(self) -> list[str]:
        return list(self.tasks)

    def task_path(self, task_id: str) -> Path:
        return self.tasks[task_id]

    def metadata(self) -> dict:
        return {
            "name": "skillflow",
            "split": self.config.split,
            "families": self.families,
            "tasks_sha256": {
                task_id: fingerprint(
                    {
                        str(path.relative_to(task)): sha256_file(path)
                        for path in sorted(task.rglob("*"))
                        if path.is_file()
                    }
                )
                for task_id, task in self.tasks.items()
            },
        }
