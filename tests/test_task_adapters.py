import json
from pathlib import Path
from threading import Event

import pytest

from sf_framework.config import TaskConfig, load_config
from sf_framework.execution import StepExecution
from sf_framework.tasks.alfworld import ALFWorldSuite
from sf_framework.tasks.webshop import WebShopEnvironment
from sf_framework.tasks.werewolf import WerewolfSuite


def test_alfworld_official_task_filter_and_split(tmp_path):
    names = [
        "pick_and_place_simple-Apple-None-CounterTop-1",
        "pick_and_place_with_movable_recep-Apple-Bowl-CounterTop-1",
        "pick_and_place_simple-AppleSliced-None-CounterTop-1",
        "pick_heat_then_place_in_recep-Apple-None-CounterTop-1",
    ]
    for index, name in enumerate(names):
        directory = tmp_path / "json_2.1.1/valid_unseen" / name / "trial"
        directory.mkdir(parents=True)
        (directory / "game.tw-pddl").write_text(json.dumps({"solvable": index != 3}))
    suite = ALFWorldSuite(TaskConfig(root=str(tmp_path), split="test"))
    assert len(suite.task_ids()) == 1
    assert suite.split == "valid_unseen"


def test_webshop_reset_uses_actual_goal_id_and_clears_previous_session():
    class Env:
        session_prefix = "sf_"
        instruction_text = "Buy a thing"

        def __init__(self):
            self.server = type("Server", (), {"user_sessions": {"sf_507": "old"}})()

        def reset(self, session):
            assert session == 507
            assert "sf_507" not in self.server.user_sessions
            return "page", None

        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": ["search"]}

    adapter = WebShopEnvironment.__new__(WebShopEnvironment)
    adapter.env = Env()
    adapter.suite = type("Suite", (), {"indices": list(range(500, 1500))})()
    state = adapter.reset("507")
    assert state.text == "page" and state.actions == ["search[YOUR QUERY]"]


def test_generation_mode_does_not_expose_legal_commands():
    from sf_framework.tasks.alfworld import ALFWorldEnvironment

    suite = type("Suite", (), {"config": TaskConfig(action_mode="generation")})()
    env = ALFWorldEnvironment(suite)
    env.state = type(
        "State", (), {"feedback": "Visible room", "admissible_commands": ["secret command"]}
    )()
    env.goal = "Find the object"
    assert env._observation().actions is None
    suite.config.action_mode = "admissible"
    assert env._observation().actions == ["secret command"]


def test_lucene_empty_document_handling_matches_official_format():
    from sf_framework.tasks.webshop import search_document

    product = {"asin": "x", "Title": "", "Description": "", "BulletPoints": [""], "options": {}}
    assert not search_document(product)["contents"].strip()
    product["options"] = {"size": ["small", "large"]}
    assert search_document(product)["contents"] == "   size: small, large"


def test_webshop_workers_have_distinct_sessions_even_for_same_task():
    class FakeEnv:
        def __init__(self, *, server, session_prefix, **kwargs):
            self.server, self.session_prefix = server, session_prefix
            self.session = session_prefix + "initial"
            server.user_sessions[self.session] = {"goal": "initial"}

        def reset(self, session):
            self.session = self.session_prefix + str(session)
            self.instruction_text = f"goal {session}"
            self.server.user_sessions[self.session] = {"goal": self.instruction_text}
            return self.session, None

        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": ["search"]}

        def close(self):
            pass

    server = type("Server", (), {"user_sessions": {}})()
    suite = type(
        "Suite",
        (),
        {
            "env_class": FakeEnv,
            "indices": [0, 1],
            "prototype": type("Prototype", (), {"server": server})(),
        },
    )()
    first, second = WebShopEnvironment(suite), WebShopEnvironment(suite)
    a, b = first.reset("0"), second.reset("0")
    assert a.text != b.text and len(server.user_sessions) == 4
    first.close()
    assert b.text in server.user_sessions
    assert not any(k.startswith(first.env.session_prefix) for k in server.user_sessions)
    assert second.reset("1").goal == "goal 1"
    second.close()
    assert not server.user_sessions


@pytest.mark.parametrize("abort", [False, True])
def test_werewolf_environment_events_follow_match_lifecycle(tmp_path, abort):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/werewolf.codex.toml")
    controller = Path(config.task.root) / "seeh/environments/werewolf_six.py"
    if not controller.is_file():
        pytest.skip("Requires the local SEEH checkout used by the Werewolf benchmark")
    session = StepExecution(config.execution).start(
        WerewolfSuite(config.task), "one_wolf-0002", tmp_path, 50, Event(), "test"
    )
    path = tmp_path / "environment_events.jsonl"
    try:
        assert json.loads(path.read_text().splitlines()[0])["phase"] == "setup"
        if not abort:
            for action in (
                "VERIFY_TARGET: player_1\nREASON: Establish an early verified read.",
                "I will compare public claims with the vote record.",
                "VOTE_TARGET: player_6\nREASON: Ask for a concrete position.",
            ):
                state = session.call("sf_step", {"action": action})
            assert set(state) == {"observation", "available_actions", "done"}
            assert state["available_actions"] == [] and state["done"] and session.terminated
    finally:
        session.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    if abort:
        assert events[-1]["type"] == "match_aborted"
        assert events[-1]["stopped_reason"] == "episode_closed"
        return

    (eliminated,) = [event for event in events if event.get("tested_player_eliminated")]
    assert (eliminated["day"], eliminated["target"]) == (1, "player_2")
    later = events[eliminated["seq"] : -1]
    assert sum(e["phase"] == "day_speech" and e["relation"] == "defense" for e in later) == 4
    assert sum(e["phase"] == "day_vote" and e["relation"] == "vote" for e in later) == 4
    assert any(e["visibility"].startswith("private:") for e in later)
    assert events[-1]["type"] == "match_finished"
    assert events[-1]["winner"] == session.grade_details["winner"] == "wolves"
    assert events[-1]["survived"] is False
