"""Portable tool-calling baseline for non-GPT OpenAI Chat Completions endpoints.

This is a distinct agent backend, not a claim that every chat endpoint supports Codex.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from sf_framework.agents.trajectory import AgentTrajectory, ModelCall
from sf_framework.config import AgentConfig
from sf_framework.core import AgentResult, EpisodeCancelled, ToolSession


class ChatAgent:
    def __init__(self, config: AgentConfig):
        self.config = config

    def run(self, session: ToolSession, output: Path) -> AgentResult:
        from openai import OpenAI

        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ValueError(f"Set {self.config.api_key_env} for this endpoint")
        deadline = time.monotonic() + self.config.timeout_seconds
        messages = [
            {"role": "system", "content": session.instructions},
            {
                "role": "user",
                "content": session.prompt or "Complete the current task using the task tools.",
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in session.tools
        ]
        result = AgentResult(
            "model_call_limit",
            model=self.config.model,
            provider=self.config.base_url or "https://api.openai.com/v1",
            model_calls=0,
        )
        trajectory = AgentTrajectory()
        trajectory.message("user", messages[-1]["content"])
        calls: list[ModelCall] = []
        last_messages = final_response = None
        try:
            with OpenAI(api_key=key, base_url=self.config.base_url, max_retries=0) as client:
                for _ in range(self.config.max_model_calls):
                    if session.stop_event.is_set():
                        result.status = "interrupted"
                        return result
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        result.status = "timeout"
                        break
                    last_messages = list(messages)
                    kwargs = {
                        "model": self.config.model,
                        "messages": messages,
                        "tools": tools,
                        "timeout": min(remaining, 120),
                    }
                    if self.config.temperature is not None:
                        kwargs["temperature"] = self.config.temperature
                    model_call = ModelCall()
                    calls.append(model_call)
                    try:
                        response = client.chat.completions.create(**kwargs)
                    except Exception:
                        model_call.status = "failed"
                        raise
                    model_call.status = "completed"
                    result.model_calls += 1
                    if response.usage:
                        for name in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            result.usage[name] = result.usage.get(name, 0) + getattr(
                                response.usage, name, 0
                            )
                        details = response.usage.completion_tokens_details
                        model_call.reasoning_tokens = details.reasoning_tokens if details else None
                    message = response.choices[0].message
                    body = message.model_dump(exclude_none=True)
                    messages.append(body)
                    model_call.items = []
                    if body.get("reasoning") or body.get("reasoning_content"):
                        model_call.items.append({"type": "reasoning"})
                    model_call.items.extend(
                        {"type": "function_call", "call_id": call.id}
                        for call in message.tool_calls or []
                    )
                    result.message = message.content or ""
                    trajectory.message("agent", result.message)
                    if not message.tool_calls:
                        final_response = message.content
                    if session.stop_event.is_set():
                        result.status = "interrupted"
                        return result
                    if not message.tool_calls:
                        result.status = session.finished_status
                        break
                    for call in message.tool_calls:
                        if session.stop_event.is_set():
                            result.status = "interrupted"
                            return result
                        if session.tool_calls >= self.config.max_tool_calls:
                            result.status = "tool_call_limit"
                            return result
                        try:
                            args = json.loads(call.function.arguments)
                        except json.JSONDecodeError:
                            trajectory.start_tool(
                                call.id, call.function.name, call.function.arguments
                            )
                            payload = {"error": "Tool arguments must be valid JSON"}
                        else:
                            trajectory.start_tool(call.id, call.function.name, args)
                            try:
                                payload = session.call(call.function.name, args)
                            except EpisodeCancelled:
                                result.status = "interrupted"
                                return result
                            except Exception:
                                trajectory.finish_tool(call.id, "failed")
                                raise
                        trajectory.finish_tool(
                            call.id, "failed" if "error" in payload else "completed"
                        )
                        content = json.dumps(payload, ensure_ascii=False)
                        trajectory.record_tool_result(call.id, content)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": content,
                            }
                        )
                        if session.done:
                            result.status = "completed"
                            return result
        finally:
            trajectory.save(output, calls, last_messages, final_response)
        return result
