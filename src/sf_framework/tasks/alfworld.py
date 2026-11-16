from __future__ import annotations

import importlib.metadata
import json
import threading
from pathlib import Path

from sf_framework.config import TaskConfig
from sf_framework.core import Observation, Transition, fingerprint, sha256_file

TASK_TYPES = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}

# TextWorld owns a module-global PDDL parser. Protect short environment calls;
# model requests run outside this lock and can overlap across task workers.
_ENV_LOCK = threading.RLock()


class ALFWorldSuite:
    default_max_steps = 50

    def __init__(self, config: TaskConfig):
        self.config = config
        self.root = Path(config.root).resolve()
        split = {"test": "valid_unseen", "validation": "valid_seen"}.get(config.split, config.split)
        if split not in {"train", "valid_seen", "valid_unseen"}:
            raise ValueError(
                "ALFWorld split must be train, validation/valid_seen or test/valid_unseen"
            )
        self.split = split
        self.games = {}
        for path in sorted((self.root / config.games_subdir / split).glob("*/*/game.tw-pddl")):
            if "movable" in str(path) or "Sliced" in str(path):
                continue
            kind = path.parent.parent.name.split("-", 1)[0]
            if kind not in TASK_TYPES:
                continue
            if not json.loads(path.read_text()).get("solvable", False):
                continue
            trajectory = path.with_name("traj_data.json")
            if trajectory.exists():
                if json.loads(trajectory.read_text()).get("task_type") not in TASK_TYPES:
                    continue
            self.games[str(path.relative_to(self.root))] = path
        if not self.games:
            raise FileNotFoundError(
                f"No solvable ALFWorld text games in {self.root / config.games_subdir / split}. Run sf setup alfworld --root ..."
            )

    def task_ids(self):
        return list(self.games)

    def make_environment(self):
        return ALFWorldEnvironment(self)

    def metadata(self):
        provenance = self.root / "provenance.json"
        return {
            "name": "alfworld",
            "modality": "text",
            "split": self.split,
            "root": str(self.root),
            "action_mode": self.config.action_mode,
            "task_count": len(self.games),
            "games_sha256": fingerprint({k: sha256_file(v) for k, v in self.games.items()}),
            "alfworld_version": importlib.metadata.version("alfworld"),
            "textworld_version": importlib.metadata.version("textworld"),
            "success": "TextWorld won flag",
            "score_range": [0, 1],
            "default_max_steps": self.default_max_steps,
            "data_provenance": json.loads(provenance.read_text()) if provenance.exists() else None,
        }


class ALFWorldEnvironment:
    def __init__(self, suite: ALFWorldSuite):
        self.suite, self.env = suite, None

    def reset(self, task_id: str):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        with _ENV_LOCK:
            self.close()
            self.env = textworld.start(
                str(self.suite.games[task_id]),
                request_infos=textworld.EnvInfos(won=True, admissible_commands=True),
                wrappers=[AlfredDemangler],
            )
            self.state = self.env.reset()
        self.goal = self.state.feedback.split("Your task is to:", 1)[-1].strip()
        return self._observation()

    def _observation(self):
        actions = (
            list(self.state.admissible_commands)
            if self.suite.config.action_mode == "admissible"
            else None
        )
        return Observation(self.state.feedback, self.goal, actions)

    def step(self, action: str):
        with _ENV_LOCK:
            self.state, _, done = self.env.step(action)
        won = bool(self.state.won)
        return Transition(self._observation(), float(won), bool(done), won)

    def close(self):
        with _ENV_LOCK:
            if self.env is not None:
                self.env.close()
                self.env = None
