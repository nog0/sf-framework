import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_core import CounterEnv

from sf_framework.agents.openai_chat import ChatAgent
from sf_framework.agents.trajectory import resolve_tool_result
from sf_framework.config import AgentConfig
from sf_framework.core import EpisodeSession


@pytest.mark.parametrize("observe_first", [False, True])
def test_arbitrary_model_tool_call_loop_against_http_endpoint(tmp_path, monkeypatch, observe_first):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer local-test-only"
            assert "Continue until done is true." in request["messages"][0]["content"]
            assert json.loads(request["messages"][1]["content"]) == {
                "observation": "A public room",
                "available_actions": ["finish", "wait"],
            }
            tool = "sf_observe" if observe_first and len(requests) == 1 else "sf_step"
            args = {} if tool == "sf_observe" else {"action": "finish"}
            response = {
                "id": "chat-test",
                "object": "chat.completion",
                "created": 1,
                "model": "non-gpt-test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{len(requests)}",
                                    "type": "function",
                                    "function": {"name": tool, "arguments": json.dumps(args)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SF_TEST_KEY", "local-test-only")
    cfg = AgentConfig(
        kind="openai_chat",
        model="non-gpt-test-model",
        api_key_env="SF_TEST_KEY",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
    )
    session = EpisodeSession(CounterEnv(), "test", 3, tmp_path)
    try:
        result = ChatAgent(cfg).run(session, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    count = 2 if observe_first else 1
    assert session.success and result.model_calls == count
    assert result.usage["total_tokens"] == 13 * count
    assert all(r["model"] == "non-gpt-test-model" for r in requests)
    if observe_first:
        assert requests[1]["messages"][-1]["role"] == "tool"
        assert json.loads(requests[1]["messages"][-1]["content"]) == {
            "observation": "A public room",
            "available_actions": ["finish", "wait"],
            "done": False,
        }
    compact = (tmp_path / "llm_trajectory.jsonl").read_text()
    assert "local-test-only" not in compact
    llm = json.loads(compact)
    assert llm == {
        "messages": requests[-1]["messages"],
        "final_response": None,
        "calls": [
            {"index": i, "status": "completed", "tool_call_ids": [f"call-{i}"]}
            for i in range(1, count + 1)
        ],
        "tool_results": [
            {
                "tool_call_id": f"call-{count}",
                "llm_call_index": count,
                "model_visible_call_id": f"call-{count}",
                "tool": "sf_step",
                "result": session.public_state(),
            },
        ],
    }
    acp = [
        json.loads(line) for line in (tmp_path / "acp_trajectory.jsonl").read_text().splitlines()
    ]
    tools = [row for row in acp if row["type"] == "tool_call"]
    if observe_first:
        observe = tools[0]
        assert observe["result_ref"] == {
            "source": "messages",
            "message_index": len(requests[-1]["messages"]) - 1,
            "content_index": None,
        }
        assert resolve_tool_result(observe, llm) == json.loads(
            requests[-1]["messages"][-1]["content"]
        )
    step = tools[-1]
    assert step["result_ref"] == {"source": "tool_results", "tool_call_id": f"call-{count}"}
    assert resolve_tool_result(step, llm) == {"observation": "Done", "done": True}
