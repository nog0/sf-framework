import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pytest
from test_agents import fake_codex
from test_core import CounterEnv

import sf_framework.runner as runner
from sf_framework.config import Config, TaskConfig, load_config
from sf_framework.core import AgentResult, EpisodeCancelled, EpisodeSession


class FourTaskSuite:
    default_max_steps = 3

    def task_ids(self):
        return ["a", "b", "c", "d"]

    def make_environment(self):
        return CounterEnv()

    def metadata(self):
        return {"name": "test-only", "score_range": [0, 1]}


class Tracker:
    def __init__(self, workers=2):
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(workers)
        self.active = self.peak = 0
        self.instances = []
        self.completed = []


class OverlapDriver:
    def __init__(self, tracker):
        self.tracker = tracker
        self.calls = 0
        with tracker.lock:
            tracker.instances.append(self)

    def run(self, session, output):
        self.calls += 1
        with self.tracker.lock:
            self.tracker.active += 1
            self.tracker.peak = max(self.tracker.peak, self.tracker.active)
        try:
            self.tracker.barrier.wait(timeout=5)
            time.sleep(0.08 if session.task_id == "a" else 0.01)
            session.call("sf_step", {"action": "finish"})
            with self.tracker.lock:
                self.tracker.completed.append(session.task_id)
            return AgentResult("completed", model="fake-only")
        finally:
            with self.tracker.lock:
                self.tracker.active -= 1


def fixture_config(tmp_path, workers=2):
    config = Config()
    config.agent.kind = "test-only"
    config.run.output = str(tmp_path / "run")
    config.run.episodes = 4
    config.run.workers = workers
    return config


@pytest.mark.parametrize("workers", [2, 4])
def test_tasks_overlap_with_independent_agents_and_ordered_results(tmp_path, monkeypatch, workers):
    tracker = Tracker(workers)
    monkeypatch.setattr(runner, "task_suite", lambda _: FourTaskSuite())
    monkeypatch.setattr(runner, "agent", lambda _: OverlapDriver(tracker))
    config = fixture_config(tmp_path, workers)
    result = runner.run(config)
    assert result["successes"] == 4 and result["complete"]
    assert tracker.peak == workers
    assert tracker.completed[0] != "a"
    assert [r["task_id"] for r in result["results"]] == ["a", "b", "c", "d"]
    assert sorted(d.calls for d in tracker.instances) == [0, 1, 1, 1, 1]
    assert result["execution"]["workers"] == workers
    for index, task_id in enumerate(["a", "b", "c", "d"]):
        episode = Path(config.run.output) / "episodes" / f"{index:05d}"
        row = json.loads((episode / "result.json").read_text())
        attempt = json.loads((episode / "attempt_001/result.json").read_text())
        assert row == attempt
        assert row["task_id"] == task_id
        assert row["worker"].startswith("sf-worker")
        assert row["steps"] == 1


def test_worker_count_can_change_on_resume_without_rerunning_tasks(tmp_path, monkeypatch):
    tracker = Tracker()
    monkeypatch.setattr(runner, "task_suite", lambda _: FourTaskSuite())
    monkeypatch.setattr(runner, "agent", lambda _: OverlapDriver(tracker))
    config = fixture_config(tmp_path)
    runner.run(config)
    original = (Path(config.run.output) / "manifest.json").read_bytes()
    config.run.workers = 3
    result = runner.run(config, resume=True)
    assert result["successes"] == 4 and result["execution"]["reused_results"] == 4
    assert sum(d.calls for d in tracker.instances) == 4
    assert (Path(config.run.output) / "manifest.json").read_bytes() == original
    events = [
        json.loads(x)
        for x in (Path(config.run.output) / "invocations.jsonl").read_text().splitlines()
    ]
    assert [e["workers"] for e in events if e["event"] == "start"] == [2, 3]


def test_interrupt_saves_partial_tasks_then_resumes_with_different_workers(tmp_path, monkeypatch):
    started = threading.Barrier(3)

    class CooperativeDriver:
        def run(self, session, output):
            started.wait(timeout=5)
            while not session.stop_event.wait(0.01):
                pass
            session.raise_if_cancelled()

    class FinishingDriver:
        def run(self, session, output):
            session.call("sf_step", {"action": "finish"})
            return AgentResult("completed")

    real_as_completed = runner.as_completed

    def interrupt_when_running(futures):
        started.wait(timeout=5)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "task_suite", lambda _: FourTaskSuite())
    monkeypatch.setattr(runner, "agent", lambda _: CooperativeDriver())
    monkeypatch.setattr(runner, "as_completed", interrupt_when_running)
    config = fixture_config(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        runner.run(config)
    first = json.loads((Path(config.run.output) / "summary.json").read_text())
    assert not first["complete"] and first["execution"]["interrupted"]
    assert first["agent_statuses"] == {"interrupted": 2}
    assert len(list((Path(config.run.output) / "episodes").glob("*/attempt_001"))) == 2
    monkeypatch.setattr(runner, "as_completed", real_as_completed)
    monkeypatch.setattr(runner, "agent", lambda _: FinishingDriver())
    config.run.workers = 3
    result = runner.run(config, resume=True)
    assert result["complete"] and result["successes"] == 4
    for index in [0, 1]:
        episode = Path(config.run.output) / "episodes" / f"{index:05d}"
        assert (episode / "attempt_001/result.json").exists()
        assert (episode / "attempt_002/result.json").exists()


def test_parallel_native_codex_processes_and_default_model_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(runner, "task_suite", lambda _: FourTaskSuite())
    config = fixture_config(tmp_path)
    config.agent.kind = "codex"
    config.agent.codex_binary = fake_codex(tmp_path)
    config.agent.auth = "chatgpt"
    config.agent.timeout_seconds = 10
    result = runner.run(config)
    assert result["successes"] == 4 and result["complete"]
    assert {r["agent"]["model"] for r in result["results"]} == {"native-test"}
    assert all(r["tool_calls"] == 2 for r in result["results"])


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_invalid_worker_counts_rejected(workers):
    config = Config()
    config.run.workers = workers
    with pytest.raises(ValueError, match="positive integer"):
        config.validate()


def test_worker_count_in_toml(tmp_path):
    path = tmp_path / "test.toml"
    path.write_text("[run]\nworkers=4\n")
    assert load_config(path).run.workers == 4


def test_cancelled_session_executes_no_further_actions(tmp_path):
    stop = threading.Event()
    env = CounterEnv()
    session = EpisodeSession(env, "one", 3, tmp_path, stop_event=stop)
    stop.set()
    with pytest.raises(EpisodeCancelled):
        session.call("sf_step", {"action": "finish"})
    assert not env.executed and session.steps == 0


def test_real_alfworld_parser_is_safe_across_threads():
    from sf_framework.tasks.alfworld import ALFWorldSuite

    root = Path(__file__).resolve().parents[1] / "data/alfworld"
    if not root.exists():
        pytest.skip("Official ALFWorld data not installed")
    pytest.importorskip("textworld")
    suite = ALFWorldSuite(TaskConfig(root=str(root)))
    tasks = suite.task_ids()[:8]

    def observe(task_id):
        env = suite.make_environment()
        try:
            initial = env.reset(task_id)
            after = env.step("look")
            assert after.observation.goal == initial.goal
            return task_id, asdict(initial)
        finally:
            env.close()

    reference = dict(map(observe, tasks))
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = dict(pool.map(observe, tasks))
    assert actual == reference
