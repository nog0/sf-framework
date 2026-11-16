import json

from sf_framework.core import AgentResult, EpisodeSession, Observation, Transition


class CounterEnv:
    def __init__(self):
        self.executed = []

    def reset(self, task_id):
        return Observation("A public room", "Finish", ["finish", "wait"])

    def step(self, action):
        self.executed.append(action)
        return Transition(
            Observation("Done" if action == "finish" else "Unchanged", "Finish"),
            1.0 if action == "finish" else 0.0,
            action == "finish",
            action == "finish",
        )

    def close(self):
        pass


def test_invalid_actions_consume_budget_and_no_reset_or_reward_tool(tmp_path):
    env = CounterEnv()
    session = EpisodeSession(env, "one", 2, tmp_path)
    initial = {
        "observation": "A public room",
        "available_actions": ["finish", "wait"],
    }
    assert json.loads(session.prompt) == initial
    assert session.tool_calls == session.steps == 0
    assert session.call("sf_observe", {}) == {**initial, "done": False}
    assert "error" in session.call("reset", {})
    assert "error" in session.call("sf_step", {"action": 3})
    assert session.steps == 0
    state = session.call("sf_step", {"action": "invented illegal action"})
    assert state == {"observation": "Unchanged", "done": False}
    state = session.call("sf_step", {"action": "another illegal action"})
    assert state == {"observation": "Unchanged", "done": True}
    assert session.done and session.truncated
    assert json.loads(session.prompt) == initial
    session.call("sf_step", {"action": "finish"})
    assert len(env.executed) == 2
    assert not session.result(AgentResult("completed"))["success"]


def test_success_comes_from_environment_and_terminal_action_is_counted(tmp_path):
    session = EpisodeSession(CounterEnv(), "one", 1, tmp_path)
    state = session.call("sf_step", {"action": "finish"})
    assert state == {"observation": "Done", "done": True}
    assert session.done and session.terminated and not session.truncated
    result = session.result(AgentResult("completed", message="I failed"))
    assert result["success"] and result["steps"] == 1
    assert result["score"] == 1.0 and result["stopped_reason"] == "terminal"


def test_agent_cannot_claim_success(tmp_path):
    session = EpisodeSession(CounterEnv(), "one", 3, tmp_path)
    assert not session.result(AgentResult("completed", message="Task succeeded!"))["success"]
