"""Run native SkillsBench tasks with BenchFlow's Docker runtime and verifier."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path

from sf_framework.config import EvaluationConfig, ExecutionConfig, TaskConfig
from sf_framework.core import (
    AgentResult,
    EpisodeCancelled,
    EvaluationInput,
    Grade,
    fingerprint,
    sha256_file,
)
from sf_framework.execution import WORKSPACE_TOOLS, WorkspaceToolSession


class SkillsBenchSuite:
    def __init__(self, config: TaskConfig):
        self.config = config
        root = Path(config.root).resolve()
        if not root.is_dir():
            raise ValueError(f"SkillsBench task root is not a directory: {root}")
        candidates = [root] if (root / "task.md").is_file() else sorted(root.iterdir())
        self.tasks = {path.name: path for path in candidates if (path / "task.md").is_file()}
        if not self.tasks:
            raise ValueError(f"No native SkillsBench task.md packages found at {root}")

    def task_ids(self) -> list[str]:
        return list(self.tasks)

    def task_path(self, task_id: str) -> Path:
        return self.tasks[task_id]

    def metadata(self) -> dict:
        return {
            "name": "skillsbench",
            "split": self.config.split,
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


class SkillsBenchExecution:
    def __init__(self, config: ExecutionConfig):
        from importlib.util import find_spec

        if find_spec("benchflow") is None:
            raise ValueError(
                "SkillsBench needs the skillsbench extra in a Python 3.12+ environment"
            )
        if unknown := set(config.options) - {"command_timeout_seconds"}:
            raise ValueError(f"Unknown SkillsBench execution options: {sorted(unknown)}")
        self.command_timeout = config.options.get("command_timeout_seconds", 120)
        if type(self.command_timeout) is not int or self.command_timeout <= 0:
            raise ValueError("SkillsBench command_timeout_seconds must be a positive integer")

    def metadata(self) -> dict:
        return {"sandbox": "docker", "benchflow": version("benchflow"), "skill_mode": "no-skill"}

    def max_steps(self, suite, requested: int | None) -> None:
        if not isinstance(suite, SkillsBenchSuite):
            raise ValueError("execution.kind=skillsbench requires native SkillsBench tasks")
        if requested is not None:
            raise ValueError("SkillsBench uses tool and time budgets; omit run.max_steps")

    def start(self, suite, task_id, output, max_steps, stop_event, label) -> SkillsBenchSession:
        if stop_event.is_set():
            raise EpisodeCancelled("Evaluation interrupted by the coordinator")

        from benchflow.rollout.task_runtime import TaskRuntime, TaskRuntimeConfig
        from benchflow.task import Task

        task_path = suite.task_path(task_id)
        task = Task(task_path)
        runtime = TaskRuntime(
            TaskRuntimeConfig(
                task_path=task_path,
                jobs_dir=output.resolve() / "benchflow",
                environment="docker",
                # Keep native absolute paths such as /root/answer.txt accessible.
                sandbox_user=None,
                skill_mode="no-skill",
            )
        )
        loop = asyncio.Runner()
        try:
            loop.run(runtime.start())
            return SkillsBenchSession(
                task_id,
                task.instruction,
                stop_event,
                label,
                output,
                runtime,
                loop,
                self.command_timeout,
            )
        except BaseException:
            with suppress(Exception):
                loop.run(runtime.close())
            loop.close()
            raise


class SkillsBenchSession(WorkspaceToolSession):
    tools = [
        {**WORKSPACE_TOOLS[0], "description": "Run a shell command in the official task container."}
    ]

    def __init__(self, task_id, prompt, stop_event, label, output, runtime, loop, command_timeout):
        super().__init__(task_id, prompt, stop_event, label)
        self.runtime = runtime
        self.loop = loop
        self.command_timeout = command_timeout
        self.container_workdir = runtime.workspace
        self.rollout_dir = runtime.rollout_dir.relative_to(output.resolve())
        self.evaluation_context = self
        self.instructions = (
            "Complete the task using sf_exec, which runs shell commands in the isolated "
            f"task container starting in {self.container_workdir}. "
            "Write deliverables at the paths requested in the task. "
            "Use only sf_exec for task work. Finish with a brief account of what you produced."
        )

    def execute_command(self, command: str) -> dict:
        result = self.loop.run(
            self.runtime.env.exec(
                command, cwd=self.container_workdir, timeout_sec=self.command_timeout
            )
        )
        payload = {"exit_code": result.return_code}
        for key in ("stdout", "stderr"):
            value = (getattr(result, key) or "").encode("utf-8")
            payload[key] = value[:65536].decode("utf-8", errors="replace")
            if len(value) > 65536:
                payload[key] += "\n[output truncated]"
        return payload

    def close(self) -> None:
        try:
            self.loop.run(self.runtime.close())
        finally:
            self.loop.close()

    def execution_result(self, agent: AgentResult) -> dict:
        return {
            **super().execution_result(agent),
            "container_workdir": self.container_workdir,
            "benchflow_rollout": str(self.rollout_dir),
        }


class SkillsBenchEvaluator:
    def __init__(self, config: EvaluationConfig):
        if config.options:
            raise ValueError("evaluation.kind=skillsbench does not take options")

    def metadata(self, task_metadata: dict) -> dict:
        return {"score_range": [0, 1], "verifier": "benchflow"}

    def evaluate(self, episode: EvaluationInput) -> Grade:
        session = episode.evaluation_context
        if not isinstance(session, SkillsBenchSession):
            raise ValueError("evaluation.kind=skillsbench requires a live SkillsBench session")
        result = session.loop.run(session.runtime.verify())
        if error := result.verifier_error or result.error:
            raise RuntimeError(f"BenchFlow verifier failed: {error}")
        if type(result.reward) not in (int, float) or not 0 <= result.reward <= 1:
            raise ValueError("BenchFlow verifier did not return a scalar reward in [0, 1]")
        return Grade(
            float(result.reward),
            result.reward == 1,
            {"rewards": result.rewards, "benchflow_rollout": str(session.rollout_dir)},
        )
