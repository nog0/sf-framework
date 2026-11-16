"""Step environments and isolated workspaces share the agent tool interface."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sf_framework.config import ExecutionConfig
from sf_framework.core import AgentResult, EpisodeCancelled, EpisodeSession

WORKSPACE_TOOLS = [
    {
        "name": "sf_exec",
        "description": "Run a shell command in the task container, starting in /workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    }
]


class StepExecution:
    def __init__(self, config: ExecutionConfig):
        if config.options:
            raise ValueError("execution.kind=step does not take options")

    def metadata(self) -> dict:
        return {}

    def max_steps(self, suite, requested: int | None) -> int:
        if not hasattr(suite, "make_environment") or not hasattr(suite, "default_max_steps"):
            raise ValueError("execution.kind=step requires a task suite with a step environment")
        return requested if requested is not None else suite.default_max_steps

    def start(self, suite, task_id, output, max_steps, stop_event, label) -> EpisodeSession:
        env = suite.make_environment()
        try:
            if set_artifact_directory := getattr(env, "set_artifact_directory", None):
                set_artifact_directory(output)
            return EpisodeSession(
                env, task_id, max_steps, output, stop_event=stop_event, label=label
            )
        except BaseException:
            env.close()
            raise


def resolve_docker_image(image: str) -> tuple[str, str]:
    if not isinstance(image, str) or not image or image.startswith("-"):
        raise ValueError("Docker image must be an image name or ID")
    docker = shutil.which("docker")
    if not docker:
        raise ValueError("Docker is required")
    result = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(f"Image must be available locally: {result.stderr.strip()}")
    return docker, result.stdout.strip()


class WorkspaceExecution:
    def __init__(self, config: ExecutionConfig):
        options = config.options
        if unknown := set(options) - {"image", "command_timeout_seconds"}:
            raise ValueError(f"Unknown workspace execution options: {sorted(unknown)}")
        self.image = options.get("image", "ubuntu:24.04")
        self.command_timeout = options.get("command_timeout_seconds", 30)
        if (
            type(self.command_timeout) not in (int, float)
            or not math.isfinite(self.command_timeout)
            or self.command_timeout <= 0
        ):
            raise ValueError("execution.options.command_timeout_seconds must be positive")
        self.docker, self.image_id = resolve_docker_image(self.image)

    def metadata(self) -> dict:
        return {"image": self.image, "image_id": self.image_id, "network": "none"}

    def max_steps(self, suite, requested: int | None) -> None:
        if not hasattr(suite, "instruction") or not hasattr(suite, "inputs"):
            raise ValueError("execution.kind=workspace requires task instructions and inputs")
        if requested is not None:
            raise ValueError("Workspace execution uses tool and time budgets; omit run.max_steps")

    def stage_inputs(self, suite, task_id, workspace: Path) -> list[str]:
        inputs = suite.inputs(task_id)
        if inputs.is_dir():
            shutil.copytree(inputs, workspace, dirs_exist_ok=True)
        return []

    def start(self, suite, task_id, output, max_steps, stop_event, label) -> WorkspaceSession:
        if stop_event.is_set():
            raise EpisodeCancelled("Evaluation interrupted by the coordinator")
        prompt = suite.instruction(task_id)
        workspace = (output / "workspace").resolve()
        workspace.mkdir()
        mounts = self.stage_inputs(suite, task_id, workspace)
        name = f"sf-workspace-{uuid.uuid4().hex}"
        session = WorkspaceSession(
            task_id, workspace, prompt, self.docker, name, self.command_timeout, stop_event, label
        )
        command = [
            self.docker,
            "run",
            "--detach",
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
            "--pids-limit=128",
            "--memory=512m",
            "--cpus=1",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,source={workspace},target=/workspace",
            *mounts,
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/workspace",
            "--entrypoint",
            "/bin/sh",
            self.image_id,
            "-c",
            "exec sleep infinity",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"Could not start workspace container: {result.stderr.strip()}")
        except BaseException:
            session.close()
            raise
        return session


class WorkspaceToolSession:
    tools = WORKSPACE_TOOLS
    instructions = (
        "Complete the task using sf_exec. This tool runs commands in an isolated task container. "
        "Inspect its files and write the requested deliverables in /workspace. "
        "Use only sf_exec for task work. Finish with a brief account of what you produced."
    )
    done = False
    environment_grade = None
    finished_status = "completed"
    workspace: Path | None = None

    def __init__(self, task_id: str, prompt: str, stop_event: threading.Event, label: str):
        self.task_id = task_id
        self.prompt = prompt
        self.stop_event = stop_event
        self.label = label
        self.tool_calls = 0

    def call(self, name: str, arguments: Any) -> dict:
        if self.stop_event.is_set():
            raise EpisodeCancelled("Evaluation interrupted by the coordinator")
        self.tool_calls += 1
        if (
            name != "sf_exec"
            or not isinstance(arguments, dict)
            or set(arguments) != {"command"}
            or not isinstance(arguments["command"], str)
            or not arguments["command"].strip()
            or "\0" in arguments["command"]
        ):
            return {"error": 'Use sf_exec({"command": "shell command"}).'}
        payload = self.execute_command(arguments["command"])
        if payload["exit_code"]:
            payload["error"] = f"Command exited with status {payload['exit_code']}"
        return payload

    def execution_result(self, agent: AgentResult) -> dict:
        return {
            "task_id": self.task_id,
            "steps": None,
            "max_steps": None,
            "terminated": agent.status == "completed",
            "truncated": agent.status in {"timeout", "model_call_limit", "tool_call_limit"},
            "stopped_reason": "agent_finished" if agent.status == "completed" else agent.status,
            "agent_status": agent.status,
            "tool_calls": self.tool_calls,
            "workspace": self.workspace.name if self.workspace is not None else None,
            "agent": asdict(agent),
        }


class WorkspaceSession(WorkspaceToolSession):
    def __init__(
        self,
        task_id: str,
        workspace: Path,
        prompt: str,
        docker: str,
        container: str,
        command_timeout: float,
        stop_event: threading.Event,
        label: str,
    ):
        super().__init__(task_id, prompt, stop_event, label)
        self.workspace = workspace
        self.docker = docker
        self.container = container
        self.command_timeout = command_timeout

    def execute_command(self, command: str) -> dict:
        cmd = [self.docker, "exec", self.container, "/bin/sh", "-c", command]
        deadline = time.monotonic() + self.command_timeout
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            with subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
            ) as proc:
                try:
                    while proc.poll() is None:
                        if self.stop_event.is_set():
                            raise EpisodeCancelled("Evaluation interrupted by the coordinator")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Workspace command deadline exceeded")
                        try:
                            proc.wait(timeout=min(remaining, 0.2))
                        except subprocess.TimeoutExpired:
                            continue
                finally:
                    if proc.poll() is None:
                        proc.kill()
            payload = {"exit_code": proc.returncode}
            for key, stream in [("stdout", stdout), ("stderr", stderr)]:
                stream.seek(0)
                value = stream.read(65537)
                payload[key] = value[:65536].decode("utf-8", errors="replace")
                if len(value) > 65536:
                    payload[key] += "\n[output truncated]"
        return payload

    def close(self) -> None:
        result = subprocess.run(
            [self.docker, "rm", "-f", self.container], capture_output=True, text=True, timeout=30
        )
        if result.returncode and "No such container" not in result.stderr:
            raise RuntimeError(f"Could not stop workspace container: {result.stderr.strip()}")
