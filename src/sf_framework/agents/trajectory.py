"""Compact, linked views of agent activity and model requests."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sf_framework.core import fingerprint


@dataclass
class ModelCall:
    status: str = "incomplete"
    items: list[dict] | None = None
    reasoning_tokens: int | None = None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _has_reasoning(items: list[dict]) -> bool:
    return any(item.get("type") == "reasoning" for item in items)


def _tool_output(item: dict) -> tuple[str | None, Any]:
    if item.get("role") == "tool":
        return item.get("tool_call_id"), item.get("content")
    if item.get("type", "").endswith("_call_output"):
        return item.get("call_id"), item.get("output")
    return None, None


def _model_tool_outputs(
    messages: list[dict] | None,
) -> dict[tuple[str, str], tuple[int, int | None] | None]:
    """Index unique full JSON results by model call and payload."""
    outputs: dict[tuple[str, str], tuple[int, int | None] | None] = {}
    for message_index, item in enumerate(messages or []):
        call_id, content = _tool_output(item)
        if not call_id:
            continue
        if isinstance(content, list):
            parts = [
                (index, part.get("text"))
                for index, part in enumerate(content)
                if isinstance(part, dict)
            ]
        else:
            parts = [(None, content)]
        for content_index, text in parts:
            if not isinstance(text, str):
                continue
            try:
                payload = json.loads(text)
            except ValueError:
                continue
            if isinstance(payload, dict):
                key = (call_id, fingerprint(payload))
                outputs[key] = None if key in outputs else (message_index, content_index)
    return outputs


def resolve_tool_result(event: dict, llm_trajectory: dict) -> dict | None:
    """Resolve an ACP tool event against the LLM record from the same attempt."""
    ref = event.get("result_ref")
    if ref is None:
        return None
    if ref["source"] == "messages":
        message_index, content_index = ref["message_index"], ref["content_index"]
        if message_index < 0 or (content_index is not None and content_index < 0):
            raise ValueError("Tool result reference indices must be nonnegative")
        _, content = _tool_output(llm_trajectory["messages"][message_index])
        text = content if content_index is None else content[content_index]["text"]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Referenced tool result is not a JSON object")
        return payload
    if ref["source"] == "tool_results":
        tool_id = ref["tool_call_id"]
        if tool_id != event["tool_call_id"]:
            raise ValueError("Supplemental result reference has the wrong ACP tool call ID")
        for entry in llm_trajectory.get("tool_results", []):
            if entry["tool_call_id"] == tool_id:
                return entry["result"]
        raise ValueError(f"Missing supplemental result for {tool_id}")
    raise ValueError(f"Unknown tool result source: {ref['source']}")


def _reasoning_between(
    previous: tuple[int, int],
    current: tuple[int, int],
    calls: list[ModelCall],
    messages: list[dict] | None,
    previous_id: str,
    current_id: str,
    history_positions: dict[str, int],
) -> bool | None:
    if previous == current:
        return False  # Multiple runtime tools issued by the same model tool item.
    if previous > current:
        return None  # Concurrent execution can reorder model-authored tool items.
    first, start = previous
    last, end = current
    left, right = history_positions.get(previous_id), history_positions.get(current_id)
    if messages is not None and left is not None and right is not None and left < right:
        if _has_reasoning(messages[left + 1 : right]):
            return True

    known = True
    for index in range(first, last + 1):
        call = calls[index]
        if call.items is None:
            known = False
            continue
        begin = start + 1 if index == first else 0
        stop = end if index == last else len(call.items)
        if _has_reasoning(call.items[begin:stop]):
            return True
        if call.reasoning_tokens:
            if index != first and index != last:
                return True
            # A count for the entire response cannot locate reasoning relative to a tool.
            if not _has_reasoning(call.items):
                known = False
        elif call.reasoning_tokens is None and not _has_reasoning(call.items) and first != last:
            known = False
        if call.status != "completed":
            known = False
    return False if known else None


class AgentTrajectory:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.tools: dict[str, dict] = {}
        self.tool_results: dict[str, str] = {}

    def message(self, speaker: str, text: str) -> None:
        if text:
            self.events.append({"type": f"{speaker}_message", "text": text})

    def start_tool(self, call_id: str, tool: str, arguments: Any) -> None:
        action = arguments.get("action") if isinstance(arguments, dict) else None
        event = {
            "type": "tool_call",
            "tool_call_id": call_id,
            "kind": "execute",
            "title": f"{tool}: {action}" if isinstance(action, str) else tool,
            "status": "in_progress",
            "content": [],
            "tool": tool,
            "arguments": arguments,
        }
        self.events.append(event)
        self.tools[call_id] = event

    def finish_tool(self, call_id: str, status: str) -> None:
        if call_id in self.tools:
            self.tools[call_id]["status"] = status

    def record_tool_result(self, call_id: str, content: str) -> None:
        self.tool_results[call_id] = content

    def save(
        self,
        output: Path,
        calls: list[ModelCall],
        messages: list[dict] | None,
        final_response: str | None,
        tool_origins: dict[str, str] | None = None,
    ) -> None:
        origins = tool_origins or {}
        positions = {
            item["call_id"]: (index, position)
            for index, call in enumerate(calls)
            for position, item in enumerate(call.items or [])
            if item.get("call_id") and item.get("type", "").endswith("_call")
        }
        history_positions = {
            item["call_id"]: position
            for position, item in enumerate(messages or [])
            if item.get("call_id") and item.get("type", "").endswith("_call")
        }
        call_index = [
            {"index": index, "status": call.status, "tool_call_ids": []}
            for index, call in enumerate(calls, 1)
        ]
        model_outputs = _model_tool_outputs(messages)
        payloads = {
            tool_id: json.loads(self.tool_results[tool_id])
            for tool_id in self.tools
            if tool_id in self.tool_results
        }
        result_keys = {
            tool_id: (origins.get(tool_id, tool_id), fingerprint(payload))
            for tool_id, payload in payloads.items()
        }
        result_counts = Counter(result_keys.values())
        supplemental_results = []
        previous = None
        previous_id = ""
        for tool_id, event in self.tools.items():
            model_id = origins.get(tool_id, tool_id)
            current = positions.get(model_id)
            event["llm_call_index"] = None if current is None else current[0] + 1
            event["reasoning_since_previous_tool"] = (
                _reasoning_between(
                    previous, current, calls, messages, previous_id, model_id, history_positions
                )
                if previous is not None and current is not None
                else None
            )
            if current is not None:
                call_index[current[0]]["tool_call_ids"].append(tool_id)
            event["result_ref"] = None
            if tool_id in result_keys:
                key = result_keys[tool_id]
                location = model_outputs.get(key)
                if location is not None and result_counts[key] == 1:
                    message_index, content_index = location
                    event["result_ref"] = {
                        "source": "messages",
                        "message_index": message_index,
                        "content_index": content_index,
                    }
                else:
                    event["result_ref"] = {"source": "tool_results", "tool_call_id": tool_id}
                    supplemental_results.append(
                        {
                            "tool_call_id": tool_id,
                            "llm_call_index": event["llm_call_index"],
                            "model_visible_call_id": (
                                model_id if current is not None or tool_id in origins else None
                            ),
                            "tool": event["tool"],
                            "result": payloads[tool_id],
                        }
                    )
            previous, previous_id = current, model_id
        compact = {"messages": messages, "final_response": final_response, "calls": call_index}
        if supplemental_results:
            compact["tool_results"] = supplemental_results
        output.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output / "llm_trajectory.jsonl", [compact])
        _write_jsonl(output / "acp_trajectory.jsonl", self.events)
