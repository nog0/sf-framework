"""Read the local Codex rollout trace without retaining its raw payload files."""

from __future__ import annotations

import json
from pathlib import Path

from sf_framework.agents.trajectory import ModelCall


def _read_payload(bundle: Path, reference: dict | None) -> dict:
    if reference:
        try:
            return json.loads((bundle / reference["path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _last_messages(
    bundle: Path, requests: dict, responses: dict, calls: dict[str, ModelCall]
) -> list[dict] | None:
    # Responses WebSocket continuations can send only the new input. Follow the
    # actual response IDs to recover the context even across retries or resets.
    chunks = []
    visited = set()
    call_id = next(reversed(requests), None)
    while call_id is not None and call_id not in visited:
        visited.add(call_id)
        request = _read_payload(bundle, requests.get(call_id))
        messages = request.get("messages", request.get("input"))
        if not isinstance(messages, list):
            return None
        chunks.append(messages)
        previous_response = request.get("previous_response_id")
        if not previous_response:
            return [item for chunk in reversed(chunks) for item in chunk]
        call_id = responses.get(previous_response)
        if call_id is None or calls[call_id].items is None:
            return None
        # Codex adds diagnostic reasoning text to the response trace that its
        # ordinary request serializer omits. Keep the request-shaped item.
        chunks.append(
            [
                {key: value for key, value in item.items() if key != "content"}
                if item.get("type") == "reasoning"
                else item
                for item in calls[call_id].items
            ]
        )
    return None


def read_codex_trace(
    root: Path, thread_id: str | None
) -> tuple[list[ModelCall], list[dict] | None, dict[str, str]]:
    log = next(root.glob("*/trace.jsonl"), None)
    if log is None:
        return [], None, {}
    calls: dict[str, ModelCall] = {}
    requests: dict[str, dict] = {}
    responses: dict[str, str] = {}
    cells: dict[str, str] = {}
    origins: dict[str, str] = {}
    with log.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # An interrupted process can leave the last row incomplete.
            if thread_id is not None and row.get("thread_id") != thread_id:
                continue
            event = row["payload"]
            kind = event["type"]
            if kind == "inference_started":
                call_id = event["inference_call_id"]
                calls[call_id] = ModelCall()
                requests[call_id] = event["request_payload"]
            elif kind in {"inference_completed", "inference_failed", "inference_cancelled"}:
                call_id = event["inference_call_id"]
                call = calls.get(call_id)
                if call is None:
                    continue
                call.status = kind.removeprefix("inference_")
                response = _read_payload(
                    log.parent,
                    event.get("response_payload") or event.get("partial_response_payload"),
                )
                call.items = response.get("output_items")
                call.reasoning_tokens = (response.get("token_usage") or {}).get(
                    "reasoning_output_tokens"
                )
                if event.get("response_id"):
                    responses[event["response_id"]] = call_id
            elif kind == "code_cell_started" and event.get("model_visible_call_id"):
                cells[event["runtime_cell_id"]] = event["model_visible_call_id"]
            elif kind == "tool_call_started":
                requester = event.get("requester") or {}
                model_id = event.get("model_visible_call_id") or cells.get(
                    requester.get("runtime_cell_id")
                )
                if model_id:
                    origins[event["tool_call_id"]] = model_id
    return list(calls.values()), _last_messages(log.parent, requests, responses, calls), origins
