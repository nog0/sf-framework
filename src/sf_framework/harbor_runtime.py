"""Execute and grade single-step Harbor tasks with a shared Docker environment."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from importlib.metadata import version
from importlib.util import find_spec

from sf_framework.config import EvaluationConfig, ExecutionConfig
from sf_framework.core import AgentResult, EpisodeCancelled, EvaluationInput, Grade
from sf_framework.execution import WORKSPACE_TOOLS, WorkspaceToolSession


class HarborExecution:
    def __init__(self, config: ExecutionConfig):
        if find_spec("harbor") is None:
            raise ValueError(
                "Harbor execution needs the skillflow extra in a Python 3.12+ environment"
            )
        if unknown := set(config.options) - {"command_timeout_seconds"}:
            raise ValueError(f"Unknown Harbor execution options: {sorted(unknown)}")
        self.command_timeout = config.options.get("command_timeout_seconds", 120)
        if type(self.command_timeout) is not int or self.command_timeout <= 0:
            raise ValueError("Harbor command_timeout_seconds must be a positive integer")

    def metadata(self) -> dict:
        return {
            "sandbox": "docker",
            "harbor": version("harbor"),
            "image_policy": "build Dockerfile",
        }

    def max_steps(self, suite, requested: int | None) -> None:
        if not callable(getattr(suite, "task_path", None)):
            raise ValueError("execution.kind=harbor requires a Harbor task package suite")
        if requested is not None:
            raise ValueError("Harbor uses tool and time budgets; omit run.max_steps")

    def start(self, suite, task_id, output, max_steps, stop_event, label) -> HarborSession:
        if stop_event.is_set():
            raise EpisodeCancelled("Evaluation interrupted by the coordinator")

        from harbor.environments.docker.docker import DockerEnvironment
        from harbor.models.task.config import TaskOS, VerifierEnvironmentMode
        from harbor.models.task.task import Task
        from harbor.models.task.verifier_mode import resolve_task_verifier_mode
        from harbor.models.trial.paths import TrialPaths

        task = Task(suite.task_path(task_id))
        config = task.config
        if (
            config.environment.os != TaskOS.LINUX
            or task.has_steps
            or resolve_task_verifier_mode(config) != VerifierEnvironmentMode.SHARED
        ):
            raise ValueError(
                "Harbor execution requires a single-step Linux task with a shared verifier"
            )

        baseline = config.environment.resolve_baseline()
        agent_policy = config.agent.explicit_phase_policy() or baseline
        verifier_policy = config.verifier.explicit_phase_policy() or baseline
        paths = TrialPaths(output.resolve() / "harbor")
        paths.mkdir()
        env = DockerEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id=f"sf-harbor-{uuid.uuid4().hex[:16]}",
            trial_paths=paths,
            task_env_config=config.environment,
            network_policy=baseline,
            phase_network_policies=[agent_policy, verifier_policy],
        )
        loop = asyncio.Runner()

        async def start():
            # The released docker_image values are local prebuild tags.
            await asyncio.wait_for(
                env.start(force_build=True), timeout=config.environment.build_timeout_sec
            )
            await env.run_healthcheck()
            if agent_policy != baseline:
                await env.set_network_policy(agent_policy)
            return await env.exec("pwd", user=config.agent.user, timeout_sec=15)

        try:
            cwd = loop.run(start())
            if cwd.return_code or not (cwd.stdout or "").strip().startswith("/"):
                raise RuntimeError(f"Could not determine Harbor working directory: {cwd.stderr}")
            return HarborSession(
                task_id,
                task,
                stop_event,
                label,
                paths,
                env,
                loop,
                cwd.stdout.strip(),
                self.command_timeout,
                agent_policy,
                verifier_policy,
            )
        except BaseException:
            with suppress(Exception):
                loop.run(env.stop(delete=True))
            loop.close()
            raise


class HarborSession(WorkspaceToolSession):
    tools = [
        {**WORKSPACE_TOOLS[0], "description": "Run a shell command in the Harbor task container."}
    ]

    def __init__(
        self,
        task_id,
        task,
        stop_event,
        label,
        paths,
        env,
        loop,
        cwd,
        timeout,
        agent_policy,
        verifier_policy,
    ):
        super().__init__(task_id, task.instruction, stop_event, label)
        self.task = task
        self.paths = paths
        self.env = env
        self.loop = loop
        self.container_workdir = cwd
        self.command_timeout = timeout
        self.agent_policy = agent_policy
        self.verifier_policy = verifier_policy
        self.evaluation_context = self
        self.instructions = (
            f"Complete the task using sf_exec. Commands run in the Harbor task container at {cwd}. "
            "Use the supplied task inputs, and write the requested deliverables at their specified "
            "paths. Use only sf_exec for task work. Finish with a brief account of what you produced."
        )

    def execute_command(self, command: str) -> dict:
        result = self.loop.run(
            self.env.exec(
                command,
                cwd=self.container_workdir,
                timeout_sec=self.command_timeout,
                user=self.task.config.agent.user,
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
            self.loop.run(self.env.stop(delete=True))
        finally:
            self.loop.close()

    def execution_result(self, agent: AgentResult) -> dict:
        return {
            **super().execution_result(agent),
            "container_workdir": self.container_workdir,
            "container_network": self.agent_policy.network_mode.value,
            "harbor_trial": self.paths.trial_dir.name,
        }


class HarborEvaluator:
    def __init__(self, config: EvaluationConfig):
        if config.options:
            raise ValueError("evaluation.kind=harbor does not take options")

    def metadata(self, task_metadata: dict) -> dict:
        return {"score_range": [0, 1], "verifier": "harbor"}

    def evaluate(self, episode: EvaluationInput) -> Grade:
        from harbor.verifier.verifier import Verifier

        session = episode.evaluation_context
        if not isinstance(session, HarborSession):
            raise ValueError("evaluation.kind=harbor requires a live Harbor session")
        config = session.task.config.verifier

        async def verify():
            if session.verifier_policy != session.agent_policy:
                await session.env.set_network_policy(session.verifier_policy)
            with session.env.with_default_user(config.user):
                verifier = Verifier(session.task, session.paths, session.env)
                return await asyncio.wait_for(verifier.verify(), timeout=config.timeout_sec)

        result = session.loop.run(verify())
        rewards = result.rewards
        if not rewards:
            raise ValueError("Harbor verifier did not return any rewards")
        # The original SkillFlow runner uses the first scalar reward.
        reward = next(iter(rewards.values()))
        if type(reward) not in (int, float) or not 0 <= reward <= 1:
            raise ValueError("Harbor verifier did not return a scalar reward in [0, 1]")
        return Grade(
            float(reward),
            reward == 1,
            {"rewards": rewards, "verifier_directory": "harbor/verifier"},
        )
