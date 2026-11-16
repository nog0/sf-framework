import json
import sys
import threading
import time

import pytest
from test_core import CounterEnv

from sf_framework.agents.codex import CodexAgent, command_for
from sf_framework.agents.codex_trace import read_codex_trace
from sf_framework.agents.trajectory import AgentTrajectory, resolve_tool_result
from sf_framework.config import AgentConfig
from sf_framework.core import EpisodeSession


def fake_codex(tmp_path, *, observe_first=True):
    script = tmp_path / "fake-codex"
    script.write_text(
        f"#!{sys.executable}\n"
        + f"OBSERVE_FIRST = {observe_first!r}\n"
        + r"""
import json, sys
if '--version' in sys.argv:
    print('fake-codex-for-protocol-tests'); raise SystemExit
def send(d):
    print(json.dumps(d), flush=True)
def call(n, tool, args):
    send({'id': 'tool-'+str(n), 'method': 'item/tool/call',
          'params': {'threadId':'t', 'turnId':'u', 'callId':str(n), 'tool':tool, 'arguments':args}})
for line in sys.stdin:
    d=json.loads(line); m=d.get('method'); rid=d.get('id')
    if m=='initialize': send({'id':rid,'result':{}})
    elif m=='account/read': send({'id':rid,'result':{'account':{'type':'chatgpt'}}})
    elif m=='thread/start':
        assert d['params']['environments']==[]
        assert len(d['params']['dynamicTools'])==2
        developer_instructions = d['params'].get('developerInstructions')
        send({'id':rid,'result':{'thread':{'id':'t'}, 'model':'native-test', 'modelProvider':'test'}})
    elif m=='turn/start':
        assert 'Continue until done is true.' in developer_instructions
        assert json.loads(d['params']['input'][0]['text']) == {
            'observation': 'A public room', 'available_actions': ['finish', 'wait']}
        send({'id':rid,'result':{'turn':{'id':'u'}}})
        call(1, 'sf_observe', {}) if OBSERVE_FIRST else call(1, 'sf_step', {'action':'finish'})
    elif rid=='tool-1' and OBSERVE_FIRST:
        assert json.loads(d['result']['contentItems'][0]['text']) == {
            'observation': 'A public room', 'available_actions': ['finish', 'wait'], 'done': False}
        call(2,'sf_step',{'action':'finish'})
    elif rid in ('tool-1', 'tool-2'):
        assert json.loads(d['result']['contentItems'][0]['text']) == {'observation':'Done', 'done': True}
    elif m=='turn/interrupt':
        send({'id':rid,'result':{}})
        send({'method':'thread/tokenUsage/updated','params':{'tokenUsage':{'total':{'totalTokens':12}}}})
        send({'method':'item/completed','params':{'item':{'type':'agentMessage','phase':'final_answer','text':'Finished.'}}})
        send({'method':'turn/completed','params':{'turn':{'id':'u','status':'completed'}}})
"""
    )
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize("observe_first", [False, True])
def test_native_agent_process_tool_round_trip(tmp_path, monkeypatch, observe_first):
    home = tmp_path / "codex-home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    config = AgentConfig(
        codex_binary=fake_codex(tmp_path, observe_first=observe_first),
        auth="chatgpt",
        timeout_seconds=10,
    )
    output = tmp_path / "episode"
    session = EpisodeSession(CounterEnv(), "one", 2, output)
    result = CodexAgent(config).run(session, output)
    assert session.success and result.model == "native-test"
    assert result.status == "completed" and result.usage["total"]["totalTokens"] == 12
    compact = json.loads((output / "llm_trajectory.jsonl").read_text())
    assert compact["messages"] is None
    assert compact["final_response"] == "Finished."
    assert compact["calls"] == []
    ids = ["1", "2"] if observe_first else ["1"]
    assert [row["tool_call_id"] for row in compact["tool_results"]] == ids
    assert compact["tool_results"][-1] == {
        "tool_call_id": ids[-1],
        "llm_call_index": None,
        "model_visible_call_id": None,
        "tool": "sf_step",
        "result": session.public_state(),
    }
    rows = [json.loads(line) for line in (output / "acp_trajectory.jsonl").read_text().splitlines()]
    tools = [row for row in rows if row["type"] == "tool_call"]
    assert [tool["tool_call_id"] for tool in tools] == ids
    assert all(tool["kind"] == "execute" for tool in tools)
    assert all(tool["reasoning_since_previous_tool"] is None for tool in tools)
    assert tools[-1]["result_ref"] == {"source": "tool_results", "tool_call_id": ids[-1]}
    assert resolve_tool_result(tools[-1], compact) == session.public_state()
    assert not (output / "trajectory.jsonl").exists()


@pytest.mark.parametrize("reasoning_in_response", [True, False])
def test_native_trace_links_nested_calls_and_uses_encrypted_reasoning_evidence(
    tmp_path, reasoning_in_response
):
    root = tmp_path / "traces"
    bundle = root / "trace-1"
    bundle.mkdir(parents=True)
    events = []

    def event(kind, **fields):
        events.append({"thread_id": "thread", "payload": {"type": kind, **fields}})

    def record(index, request, output, *, reasoning_tokens):
        request_path = f"request-{index}.json"
        response_path = f"response-{index}.json"
        (bundle / request_path).write_text(json.dumps(request))
        (bundle / response_path).write_text(
            json.dumps(
                {
                    "output_items": output,
                    "token_usage": {"reasoning_output_tokens": reasoning_tokens},
                }
            )
        )
        event("inference_started", inference_call_id=index, request_payload={"path": request_path})
        event(
            "inference_completed",
            inference_call_id=index,
            response_id=f"response-{index}",
            response_payload={"path": response_path},
        )

    initial = {"type": "message", "role": "user", "content": "task"}
    first_tool = {"type": "custom_tool_call", "call_id": "exec-1"}
    second_tool = {"type": "custom_tool_call", "call_id": "exec-2"}
    hidden = {"type": "reasoning", "summary": [], "encrypted_content": "opaque"}
    printed = {"observation": "full result", "done": False}
    hidden_result = {"observation": "not printed by exec", "done": False}
    terminal = {"observation": "terminal result", "done": True}
    first_result = {
        "type": "custom_tool_call_output",
        "call_id": "exec-1",
        "output": [
            {"type": "input_text", "text": "Script completed"},
            {"type": "input_text", "text": json.dumps(printed)},
        ],
    }
    second_result = {"type": "custom_tool_call_output", "call_id": "exec-2", "output": "result"}

    record("1", {"input": [initial]}, [first_tool], reasoning_tokens=0)
    event("code_cell_started", runtime_cell_id="cell-1", model_visible_call_id="exec-1")
    event("tool_call_started", tool_call_id="tool-1", requester={"runtime_cell_id": "cell-1"})
    event("tool_call_started", tool_call_id="tool-2", requester={"runtime_cell_id": "cell-1"})
    record(
        "2",
        {"input": [first_result], "previous_response_id": "response-1"},
        [hidden, second_tool] if reasoning_in_response else [second_tool],
        reasoning_tokens=42,
    )
    event("code_cell_started", runtime_cell_id="cell-2", model_visible_call_id="exec-2")
    event("tool_call_started", tool_call_id="tool-3", requester={"runtime_cell_id": "cell-2"})
    last_messages = [initial, first_tool, first_result, hidden, second_tool, second_result]
    last_request = (
        {"input": [second_result], "previous_response_id": "response-2"}
        if reasoning_in_response
        else {"input": last_messages}
    )
    record("3", last_request, [{"type": "message", "role": "assistant"}], reasoning_tokens=0)
    (bundle / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    calls, messages, origins = read_codex_trace(root, "thread")
    trajectory = AgentTrajectory()
    for call_id, payload in zip(("tool-1", "tool-2", "tool-3"), (printed, hidden_result, terminal)):
        trajectory.start_tool(call_id, "sf_step", {"action": "look"})
        trajectory.record_tool_result(call_id, json.dumps(payload))
        trajectory.finish_tool(call_id, "completed")
    trajectory.save(tmp_path, calls, messages, "done", origins)
    rows = [
        json.loads(line) for line in (tmp_path / "acp_trajectory.jsonl").read_text().splitlines()
    ]
    assert [row["llm_call_index"] for row in rows] == [1, 1, 2]
    assert [row["reasoning_since_previous_tool"] for row in rows] == [None, False, True]
    assert "opaque" not in (tmp_path / "acp_trajectory.jsonl").read_text()
    compact = json.loads((tmp_path / "llm_trajectory.jsonl").read_text())
    assert rows[0]["result_ref"] == {
        "source": "messages",
        "message_index": 2,
        "content_index": 1,
    }
    assert [resolve_tool_result(row, compact) for row in rows] == [printed, hidden_result, terminal]
    assert compact == {
        "messages": last_messages,
        "final_response": "done",
        "calls": [
            {"index": 1, "status": "completed", "tool_call_ids": ["tool-1", "tool-2"]},
            {"index": 2, "status": "completed", "tool_call_ids": ["tool-3"]},
            {"index": 3, "status": "completed", "tool_call_ids": []},
        ],
        "tool_results": [
            {
                "tool_call_id": "tool-2",
                "llm_call_index": 1,
                "model_visible_call_id": "exec-1",
                "tool": "sf_step",
                "result": hidden_result,
            },
            {
                "tool_call_id": "tool-3",
                "llm_call_index": 2,
                "model_visible_call_id": "exec-2",
                "tool": "sf_step",
                "result": terminal,
            },
        ],
    }


@pytest.mark.parametrize(
    ("runtime_values", "printed_values", "content_indices"),
    [
        ([1, 2], [2, 1], [2, 1]),
        ([1, 1], [1], [None, None]),
        ([1], [1, 1], [None]),
    ],
)
def test_result_references_for_tools_in_one_exec(
    tmp_path, runtime_values, printed_values, content_indices
):
    trajectory = AgentTrajectory()
    origins = {}
    for i, value in enumerate(runtime_values):
        call_id = f"tool-{i}"
        origins[call_id] = "exec-1"
        trajectory.start_tool(call_id, "sf_observe", {})
        trajectory.record_tool_result(call_id, json.dumps({"value": value}))
        trajectory.finish_tool(call_id, "completed")
    trajectory.start_tool("no-result", "sf_observe", {})
    messages = [
        {
            "type": "custom_tool_call_output",
            "call_id": "exec-1",
            "output": [
                {"type": "input_text", "text": "Script completed"},
                *[
                    {"type": "input_text", "text": json.dumps({"value": value})}
                    for value in printed_values
                ],
            ],
        }
    ]
    trajectory.save(tmp_path, [], messages, None, origins)
    rows = [
        json.loads(line) for line in (tmp_path / "acp_trajectory.jsonl").read_text().splitlines()
    ]
    compact = json.loads((tmp_path / "llm_trajectory.jsonl").read_text())
    assert compact["messages"] == messages
    for row, value, index in zip(rows[:-1], runtime_values, content_indices):
        assert row["result_ref"] == (
            {"source": "tool_results", "tool_call_id": row["tool_call_id"]}
            if index is None
            else {"source": "messages", "message_index": 0, "content_index": index}
        )
        assert resolve_tool_result(row, compact) == {"value": value}
    assert rows[-1]["result_ref"] is None
    assert resolve_tool_result(rows[-1], compact) is None
    assert resolve_tool_result({"tool_call_id": "old-run"}, compact) is None


def test_api_key_is_never_put_in_process_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_MODEL_KEY", "secret-test-value")
    cmd, env, provider = command_for(AgentConfig(auth="api_key", api_key_env="TEST_MODEL_KEY"))
    assert "secret-test-value" not in " ".join(cmd)
    assert env["TEST_MODEL_KEY"] == "secret-test-value"
    assert provider == "sf_api"
    assert any('wire_api="responses"' in x for x in cmd)


def test_configured_mcp_servers_are_disabled_per_invocation(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('[mcp_servers."some-server"]\ncommand="irrelevant"\n')
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    cmd, _, _ = command_for(AgentConfig())
    assert 'mcp_servers."some-server".enabled=false' in cmd


def test_missing_key_and_incompatible_login_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("MISSING_KEY_FOR_TEST", raising=False)
    with pytest.raises(ValueError, match="MISSING_KEY_FOR_TEST"):
        command_for(AgentConfig(auth="api_key", api_key_env="MISSING_KEY_FOR_TEST"))
    with pytest.raises(ValueError, match="ChatGPT login"):
        command_for(AgentConfig(auth="chatgpt", base_url="http://localhost/v1"))


def test_resumed_default_model_is_pinned_and_mixed_models_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    driver = CodexAgent(AgentConfig())
    driver.restore_result({"agent": {"model": "first-model", "provider": "openai"}})
    assert driver.effective_model == "first-model"
    with pytest.raises(ValueError, match="different Codex models"):
        driver.restore_result({"agent": {"model": "other-model"}})


def test_codex_user_settings_are_fingerprinted_without_storing_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    path.write_text('model="first-model"\n')
    driver = CodexAgent(AgentConfig())
    old = driver.metadata()
    assert "first-model" not in json.dumps(old)
    path.write_text('model="other-model"\n')
    assert driver.metadata() != old


def test_native_cancel_interrupts_a_blocked_protocol_read(tmp_path, monkeypatch):
    script = tmp_path / "waiting-codex"
    script.write_text(
        f"#!{sys.executable}\n"
        + """
import json,sys
if '--version' in sys.argv:
 print('fake-waiting-codex');raise SystemExit
for line in sys.stdin:
 d=json.loads(line);m=d.get('method');i=d.get('id')
 if m=='initialize': print(json.dumps({'id':i,'result':{}}),flush=True)
 elif m=='thread/start': print(json.dumps({'id':i,'result':{'thread':{'id':'t'},'model':'fake','modelProvider':'test'}}),flush=True)
 elif m=='turn/start': print(json.dumps({'id':i,'result':{'turn':{'id':'u'}}}),flush=True)
"""
    )
    script.chmod(0o755)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    stop = threading.Event()
    session = EpisodeSession(CounterEnv(), "one", 3, tmp_path, stop_event=stop)
    config = AgentConfig(codex_binary=str(script), timeout_seconds=30)
    timer = threading.Timer(0.3, stop.set)
    started = time.monotonic()
    timer.start()
    try:
        result = CodexAgent(config).run(session, tmp_path)
    finally:
        timer.cancel()
    assert result.status == "interrupted" and session.steps == 0
    assert time.monotonic() - started < 3
