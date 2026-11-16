"""Codex's native agent loop over app-server JSON-RPC dynamic tools.

No per-action LLM wrapper, credential scraping, or reimplementation of Codex planning.
Dynamic tools are experimental; verified against codex-cli 0.166.0.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from sf_framework.agents.codex_trace import read_codex_trace
from sf_framework.agents.trajectory import AgentTrajectory
from sf_framework.config import AgentConfig
from sf_framework.core import (
    TOOLS,
    AgentResult,
    EpisodeCancelled,
    ToolSession,
    sha256_file,
)


class RpcClient:
    def __init__(
        self,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        stderr: Path,
        *,
        stop_event: threading.Event | None = None,
    ):
        self.stop_event = stop_event
        self.stderr_file = stderr.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.messages: queue.Queue = queue.Queue()
        self.sequence = 0
        self.notifications: list[dict] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.messages.put(json.loads(line))
                except json.JSONDecodeError as exc:
                    self.messages.put(RuntimeError(f"Codex emitted invalid JSON: {exc}"))
        finally:
            self.messages.put(EOFError("Codex app-server closed stdout; inspect codex.stderr.log"))

    def send(self, message: dict):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def read(self, deadline: float):
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise EpisodeCancelled("Evaluation interrupted by the coordinator")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Codex episode deadline exceeded")
            try:
                message = self.messages.get(timeout=min(remaining, 0.5))
                break
            except queue.Empty:
                continue
        if isinstance(message, Exception):
            raise message
        return message

    def request(self, method: str, params: dict, deadline: float):
        self.sequence += 1
        request_id = self.sequence
        self.send({"id": request_id, "method": method, "params": params})
        while True:
            message = self.read(deadline)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"Codex {method}: {message['error']}")
                return message.get("result", {})
            if "id" in message and "method" in message:
                self.send(
                    {
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": "Unsupported request before task startup",
                        },
                    }
                )
            else:
                self.notifications.append(message)

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        finally:
            self.process.stdout.close()
            self.stderr_file.close()


def _override(command: list[str], key: str, value: Any):
    # JSON strings and booleans are valid TOML values. Never put credentials here.
    command.extend(["-c", f"{key}={json.dumps(value)}"])


def command_for(config: AgentConfig) -> tuple[list[str], dict[str, str], str | None]:
    command = [config.codex_binary]
    env = os.environ.copy()
    if config.codex_home:
        env["CODEX_HOME"] = config.codex_home
    # Remove unrelated capabilities from this benchmark invocation only.
    for feature in [
        "shell_tool",
        "apps",
        "plugins",
        "hooks",
        "memories",
        "multi_agent",
        "multi_agent_v2",
        "skill_search",
        "code_mode",
    ]:
        _override(command, f"features.{feature}", False)
    # Recent Codex versions route dynamic function tools through this host even
    # when code_mode is false. Its presence does not grant filesystem tools.
    _override(command, "features.code_mode_host", True)
    _override(command, "web_search", "disabled")
    _override(command, "project_root_markers", [])
    # Empty TOML tables merge with user tables; explicitly disable each configured server.
    home = Path(env.get("CODEX_HOME", str(Path.home() / ".codex")))
    config_path = home / "config.toml"
    if config_path.exists():
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        user_config = tomllib.loads(config_path.read_text())
        for name in user_config.get("mcp_servers", {}):
            _override(command, f"mcp_servers.{json.dumps(name)}.enabled", False)
    provider = config.provider
    if config.auth == "chatgpt":
        if config.base_url or provider:
            raise ValueError(
                "ChatGPT login uses the OpenAI provider; choose auth=provider for a gateway"
            )
        provider = "openai"
        _override(command, "forced_login_method", "chatgpt")
    elif config.auth == "api_key" or config.base_url:
        if not os.environ.get(config.api_key_env):
            raise ValueError(f"Set {config.api_key_env} for API authentication")
        provider = provider or ("sf_api" if config.auth == "api_key" else "sf_custom")
        if provider in {"openai", "ollama", "lmstudio", "amazon-bedrock", "gcp"}:
            raise ValueError("Use a unique provider name for a custom API endpoint")
        for key, value in {
            "name": provider,
            "base_url": config.base_url or "https://api.openai.com/v1",
            "env_key": config.api_key_env,
            "wire_api": "responses",
            "requires_openai_auth": False,
        }.items():
            _override(command, f"model_providers.{provider}.{key}", value)
    elif config.auth == "provider" and not provider:
        raise ValueError("auth=provider requires provider or base_url")
    if provider:
        _override(command, "model_provider", provider)
    command.extend(["app-server", "--listen", "stdio://"])
    return command, env, provider


class CodexAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.effective_model = config.model
        self.effective_provider = config.provider
        self.initial_settings = self.metadata()

    def metadata(self):
        home = Path(
            self.config.codex_home or os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
        )
        paths = [home / "config.toml", home / "AGENTS.md", Path("/etc/codex/config.toml")]
        return {
            "codex_home": str(home.resolve()),
            "configuration_sha256": {
                str(p): sha256_file(p) if p.is_file() else None for p in paths
            },
        }

    def restore_result(self, previous: dict):
        """Pin a resumed run to its recorded effective model, including default-model runs."""
        prior = previous.get("agent", {})
        model, provider = prior.get("model"), prior.get("provider")
        if model:
            if self.effective_model and model != self.effective_model:
                raise ValueError("Existing episodes contain different Codex models")
            self.effective_model = model
        if provider:
            if self.effective_provider and provider != self.effective_provider:
                raise ValueError("Existing episodes contain different Codex providers")
            self.effective_provider = provider

    def for_worker(self):
        """Copy configuration and the pinned model, never an active conversation."""
        worker = type(self)(copy.deepcopy(self.config))
        worker.initial_settings = copy.deepcopy(self.initial_settings)
        worker.effective_model = self.effective_model
        worker.effective_provider = self.effective_provider
        return worker

    def prepare(self, output: Path):
        """Resolve an unspecified default once, before dispatching parallel episodes.

        thread/start selects the model without starting an inference turn.
        Explicitly configured models need no extra app-server process.
        """
        command, env, provider = command_for(self.config)
        if self.effective_provider is None:
            self.effective_provider = provider
        if self.effective_model:
            return
        deadline = time.monotonic() + min(self.config.timeout_seconds, 60)
        with tempfile.TemporaryDirectory(prefix="sf-model-resolution-") as workspace:
            client = RpcClient(
                command, Path(workspace), env, output / "model-resolution.stderr.log"
            )
            try:
                client.request(
                    "initialize",
                    {
                        "clientInfo": {"name": "sf_framework", "version": "0.1.0"},
                        "capabilities": {"experimentalApi": True},
                    },
                    deadline,
                )
                client.send({"method": "initialized", "params": {}})
                params = {
                    "cwd": workspace,
                    "ephemeral": True,
                    "sandbox": "read-only",
                    "approvalPolicy": "never",
                    "environments": [],
                    "dynamicTools": [{"type": "function", **tool} for tool in TOOLS],
                    "allowProviderModelFallback": False,
                }
                if self.effective_provider:
                    params["modelProvider"] = self.effective_provider
                selected = client.request("thread/start", params, deadline)
                self.effective_model = selected["model"]
                self.effective_provider = selected["modelProvider"]
            finally:
                client.close()

    def run(self, session: ToolSession, output: Path) -> AgentResult:
        if self.metadata() != self.initial_settings:
            raise ValueError("Codex configuration changed during this evaluation")
        command, env, provider = command_for(self.config)
        deadline = time.monotonic() + self.config.timeout_seconds
        version = subprocess.check_output(
            [self.config.codex_binary, "--version"], text=True
        ).strip()
        result = AgentResult(
            "agent_stopped",
            model=self.config.model,
            provider=provider,
            details={
                "codex_version": version,
                "transport": "app-server/dynamicTools",
                "auth": self.config.auth,
            },
        )
        trajectory = AgentTrajectory()
        thread_id = final_response = None
        with tempfile.TemporaryDirectory(prefix="sf-agent-") as scratch:
            workspace = Path(scratch) / "workspace"
            workspace.mkdir()
            trace_root = Path(scratch) / "rollout-trace"
            env["CODEX_ROLLOUT_TRACE_ROOT"] = str(trace_root)
            client = RpcClient(
                command,
                workspace,
                env,
                output / "codex.stderr.log",
                stop_event=session.stop_event,
            )
            try:
                client.request(
                    "initialize",
                    {
                        "clientInfo": {"name": "sf_framework", "version": "0.1.0"},
                        "capabilities": {"experimentalApi": True},
                    },
                    deadline,
                )
                client.send({"method": "initialized", "params": {}})
                if self.config.auth == "chatgpt":
                    account = client.request("account/read", {"refreshToken": False}, deadline)
                    if (account.get("account") or {}).get("type") != "chatgpt":
                        raise RuntimeError(
                            "ChatGPT login required. Run sf auth login --method chatgpt"
                        )
                params = {
                    "cwd": str(workspace),
                    "ephemeral": True,
                    "sandbox": "read-only",
                    "approvalPolicy": "never",
                    "environments": [],
                    "developerInstructions": session.instructions,
                    "dynamicTools": [{"type": "function", **tool} for tool in session.tools],
                    "allowProviderModelFallback": False,
                }
                if self.effective_model:
                    params["model"] = self.effective_model
                if self.effective_provider or provider:
                    params["modelProvider"] = self.effective_provider or provider
                started = client.request("thread/start", params, deadline)
                thread_id = started["thread"]["id"]
                result.model, result.provider = started["model"], started["modelProvider"]
                if self.effective_model and self.effective_model != result.model:
                    raise ValueError("Codex did not select the requested model")
                self.effective_model, self.effective_provider = result.model, result.provider
                result.details.update(
                    thread_id=thread_id,
                    reasoning_effort=self.config.reasoning_effort or started.get("reasoningEffort"),
                )
                prompt = session.prompt or "Complete the current task using the task tools."
                turn_params = {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                }
                if self.config.reasoning_effort:
                    turn_params["effort"] = self.config.reasoning_effort
                trajectory.message("user", prompt)
                turn = client.request("turn/start", turn_params, deadline)["turn"]
                interrupt_sent = False
                result.model_calls = 0
                last_usage = None
                while True:
                    message = (
                        client.notifications.pop(0)
                        if client.notifications
                        else client.read(deadline)
                    )
                    method, params = message.get("method"), message.get("params", {})
                    if method == "item/tool/call":
                        call_id = params.get("callId") or str(message["id"])
                        trajectory.start_tool(call_id, params["tool"], params.get("arguments"))
                        if interrupt_sent:
                            payload = {"error": "Episode already stopped"}
                        elif session.tool_calls >= self.config.max_tool_calls:
                            payload = {"error": "Task tool-call budget exhausted"}
                            result.status = "tool_call_limit"
                        else:
                            try:
                                payload = session.call(params["tool"], params.get("arguments"))
                            except Exception:
                                trajectory.finish_tool(call_id, "failed")
                                raise
                            if params["tool"] == "sf_step":
                                print(
                                    f"  [{session.label}] step {session.steps}/{session.max_steps}: {params.get('arguments')}",
                                    flush=True,
                                )
                        tool_success = "error" not in payload
                        content = json.dumps(payload, ensure_ascii=False)
                        trajectory.record_tool_result(call_id, content)
                        client.send(
                            {
                                "id": message["id"],
                                "result": {
                                    "success": tool_success,
                                    "contentItems": [{"type": "inputText", "text": content}],
                                },
                            }
                        )
                        trajectory.finish_tool(call_id, "completed" if tool_success else "failed")
                        if (
                            session.done or result.status == "tool_call_limit"
                        ) and not interrupt_sent:
                            client.sequence += 1
                            client.send(
                                {
                                    "id": client.sequence,
                                    "method": "turn/interrupt",
                                    "params": {"threadId": thread_id, "turnId": turn["id"]},
                                }
                            )
                            interrupt_sent = True
                            deadline = min(deadline, time.monotonic() + 20)
                    elif method == "thread/tokenUsage/updated":
                        result.usage = params.get("tokenUsage", {})
                        if result.usage != last_usage:
                            result.model_calls += 1
                            last_usage = result.usage
                        if (
                            result.model_calls >= self.config.max_model_calls
                            and not session.done
                            and not interrupt_sent
                        ):
                            result.status = "model_call_limit"
                            client.sequence += 1
                            client.send(
                                {
                                    "id": client.sequence,
                                    "method": "turn/interrupt",
                                    "params": {"threadId": thread_id, "turnId": turn["id"]},
                                }
                            )
                            interrupt_sent = True
                            deadline = min(deadline, time.monotonic() + 20)
                    elif method == "item/completed":
                        item = params.get("item", {})
                        if item.get("type") == "agentMessage":
                            result.message = item.get("text", "")
                            trajectory.message("agent", result.message)
                            if item.get("phase") == "final_answer":
                                final_response = result.message
                        elif item.get("type") == "dynamicToolCall":
                            trajectory.finish_tool(item["id"], item["status"])
                    elif method == "turn/completed":
                        ended = params["turn"]
                        if not interrupt_sent:
                            result.status = (
                                session.finished_status
                                if ended["status"] == "completed"
                                else ended["status"]
                            )
                        if ended.get("error"):
                            result.details["error"] = ended["error"]
                        break
                    elif "id" in message and method:
                        # No user questions, approvals, outside tools, or token refresh owned by the host.
                        client.send(
                            {
                                "id": message["id"],
                                "error": {
                                    "code": -32601,
                                    "message": "Only the task's dynamic tool calls are supported",
                                },
                            }
                        )
                if session.done:
                    result.status = "completed"
            except EpisodeCancelled as exc:
                result.status = "completed" if session.done else "interrupted"
                result.details["error"] = str(exc)
            except TimeoutError as exc:
                result.status = "completed" if session.done else "timeout"
                result.details["error"] = str(exc)
            except Exception as exc:
                result.status = "completed" if session.done else "error"
                result.details["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                client.close()
                try:
                    calls, messages, origins = read_codex_trace(trace_root, thread_id)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    result.details["llm_trajectory_error"] = str(exc)
                    calls, messages, origins = [], None, {}
                if not calls:
                    result.details["llm_trajectory_status"] = "unavailable"
                elif messages is None:
                    result.details["llm_trajectory_status"] = "partial"
                trajectory.save(output, calls, messages, final_response, origins)
        return result
