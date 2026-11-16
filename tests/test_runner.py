import json

import pytest
from test_core import CounterEnv

import sf_framework.runner as runner
from sf_framework.config import Config, load_config
from sf_framework.core import AgentResult, Grade


class Suite:
    default_max_steps = 3

    def task_ids(self):
        return ["a", "b"]

    def make_environment(self):
        return CounterEnv()

    def metadata(self):
        return {"name": "test-only", "version": 1}


class Driver:
    def __init__(self):
        self.calls = 0

    def run(self, session, output):
        self.calls += 1
        session.call("sf_step", {"action": "finish"})
        return AgentResult("completed", model="fake-test-model")


def test_resume_and_manifest_reject_changed_conditions(tmp_path, monkeypatch):
    driver = Driver()
    monkeypatch.setattr(runner, "task_suite", lambda _: Suite())
    monkeypatch.setattr(runner, "agent", lambda _: driver)
    config = Config()
    config.agent.kind = "fake"
    config.run.output = str(tmp_path / "run")
    config.run.episodes = 2
    assert runner.run(config)["successes"] == 2
    assert driver.calls == 2
    assert runner.run(config, resume=True)["successes"] == 2
    assert driver.calls == 2
    config.run.max_steps = 7
    with pytest.raises(ValueError, match="Resume rejected"):
        runner.run(config, resume=True)


class PartialEvaluator:
    def __init__(self, config):
        self.score = config.options["score"]

    def metadata(self, task_metadata):
        return {"score_range": [0, 1]}

    def evaluate(self, episode):
        assert episode.environment_grade.success
        assert episode.workspace is None
        return Grade(self.score, False, {"rule": "test_override"})


def test_evaluator_can_change_the_grade_without_changing_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "task_suite", lambda _: Suite())
    monkeypatch.setattr(runner, "agent", lambda _: Driver())
    config = Config()
    config.agent.kind = "fake"
    config.run.output = str(tmp_path / "run")
    config.evaluation.kind = f"{__name__}:PartialEvaluator"
    config.evaluation.options = {"score": 0.25}
    result = runner.run(config)["results"][0]
    assert result["terminated"] and result["stopped_reason"] == "terminal"
    assert result["score"] == 0.25 and not result["success"]
    assert result["evaluation"]["status"] == "completed"
    config.evaluation.options["score"] = 0.5
    with pytest.raises(ValueError, match="Resume rejected"):
        runner.run(config, resume=True)


@pytest.mark.parametrize("verifier_error", [False, True])
def test_live_evaluator_scores_before_session_cleanup(tmp_path, monkeypatch, verifier_error):
    calls = []

    class Session:
        workspace = None
        environment_grade = None
        evaluation_context = object()

        def close(self):
            calls.append("close")

        def execution_result(self, outcome):
            return {"task_id": "a", "steps": None, "agent_status": outcome.status}

    session = Session()

    class Executor:
        def start(self, *_):
            return session

    class Agent:
        def run(self, *_):
            calls.append("solve")
            return AgentResult("completed")

    class Evaluator:
        def evaluate(self, episode):
            calls.append("grade")
            assert episode.evaluation_context is session.evaluation_context
            if verifier_error:
                raise RuntimeError("verifier crashed")
            return Grade(1.0, True)

    monkeypatch.setattr(runner, "evaluator", lambda _: Evaluator())
    result = runner._execute_episode(
        Suite(),
        Executor(),
        Config().evaluation,
        Agent,
        0,
        "a",
        1,
        None,
        tmp_path,
        runner.threading.Event(),
    )
    assert calls == ["solve", "grade", "close"]
    assert result["evaluation"]["status"] == ("error" if verifier_error else "completed")
    assert result["success"] is not verifier_error


def test_partial_agent_failure_is_preserved_and_retry_is_separate(tmp_path, monkeypatch):
    class Failing:
        def run(self, session, output):
            session.call("sf_step", {"action": "wait"})
            raise RuntimeError("transport lost")

    monkeypatch.setattr(runner, "task_suite", lambda _: Suite())
    monkeypatch.setattr(runner, "agent", lambda _: Failing())
    config = Config()
    config.agent.kind = "fake"
    config.run.output = str(tmp_path / "run")
    first = runner.run(config)
    assert first["results"][0]["steps"] == 1
    assert first["agent_statuses"] == {"error": 1}
    monkeypatch.setattr(runner, "agent", lambda _: Driver())
    assert runner.run(config, resume=True, retry_errors=True)["successes"] == 1
    episode = tmp_path / "run/episodes/00000"
    failed = json.loads((episode / "attempt_001/result.json").read_text())
    retried = json.loads((episode / "attempt_002/result.json").read_text())
    assert failed["agent_status"] == "error" and failed["steps"] == 1
    assert retried["success"] and retried["steps"] == 1


def test_config_paths_and_unknown_fields(tmp_path):
    p = tmp_path / "configs" / "test.toml"
    p.parent.mkdir()
    p.write_text('[task]\nroot="../data"\n[run]\noutput="../out"\n')
    config = load_config(p)
    assert config.task.root == str(tmp_path / "data")
    p.write_text('[agent]\nmodle="typo"\n')
    with pytest.raises(ValueError, match="Unknown agent"):
        load_config(p)


def test_requested_task_ids_are_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "task_suite", lambda _: Suite())
    config = Config()
    config.run.output = str(tmp_path / "run")
    config.run.task_ids = ["outside-split"]
    with pytest.raises(ValueError, match="task_ids"):
        runner.run(config)


def test_plugin_factory_without_modifying_core():
    from sf_framework.registry import resolve

    assert resolve("sf_framework.config:Config", {}) is Config


def test_summary_retains_errors_in_denominator():
    rows = [
        {
            "success": True,
            "score": 1,
            "steps": 1,
            "agent_status": "completed",
            "stopped_reason": "terminal",
        },
        {
            "success": False,
            "score": 0,
            "steps": 1,
            "agent_status": "error",
            "stopped_reason": "error",
        },
    ]
    report = runner.summarize(rows, 3)
    assert report["success_rate"] == 0.5 and report["mean_score_0_100"] == 50
    assert not report["complete"]


def test_general_task_score_range_is_explicit():
    rows = [
        {
            "success": True,
            "score": 7,
            "steps": 1,
            "agent_status": "completed",
            "stopped_reason": "terminal",
        }
    ]
    assert runner.summarize(rows, 1, (0, 10))["mean_score_0_100"] == 70
    assert runner.summarize(rows, 1, None)["mean_score_0_100"] is None


def test_output_lock_rejects_concurrent_run(tmp_path):
    import fcntl

    config = Config()
    config.run.output = str(tmp_path)
    with (tmp_path / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another evaluation"):
            runner.run(config)
