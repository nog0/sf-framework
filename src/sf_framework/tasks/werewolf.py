"""Expose a single seat of the original SEEH six-player match as a step environment."""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import TextIO

from sf_framework.config import TaskConfig
from sf_framework.core import Observation, Transition, fingerprint, sha256_file


@dataclass(frozen=True)
class Game:
    seat: int
    role: str
    seed: int


class WerewolfSuite:
    default_max_steps = 50

    def __init__(self, config: TaskConfig):
        self.root = Path(config.root).resolve()
        self.source = self.root / "seeh"
        controller = self.source / "environments" / "werewolf_six.py"
        if not controller.is_file():
            raise ValueError(f"Missing SEEH Werewolf controller at {controller}")
        if config.action_mode != "admissible":
            raise ValueError("Werewolf uses the source controller's legal-action prompts")
        if unknown := set(config.options) - {"role_preset", "games", "max_days"}:
            raise ValueError(f"Unknown Werewolf task options: {sorted(unknown)}")

        # Load the configured checkout without installing it or reading its .env.
        sys.path.insert(0, str(self.root))
        try:
            self.rules = importlib.import_module("seeh.environments.werewolf_six")
        finally:
            sys.path.remove(str(self.root))
        if Path(self.rules.__file__).resolve() != controller.resolve():
            raise ValueError("A different SEEH checkout is already loaded")

        self.preset = config.options.get("role_preset", "one_wolf")
        self.roles = self.rules.roles_for_preset(self.preset)
        games = config.options.get("games", 36)
        self.max_days = config.options.get("max_days", 6)
        if any(type(value) is not int or value < 1 for value in (games, self.max_days)):
            raise ValueError("Werewolf games and max_days must be positive integers")
        if type(config.seed) is not int:
            raise ValueError("Werewolf seed must be an integer")
        self.split = config.split
        self.games = {
            f"{self.preset}-{index + 1:04d}": Game(
                seat=1 + index % 6,
                role=self.rules.balanced_tested_role(self.preset, index),
                seed=config.seed + index,
            )
            for index in range(games)
        }

    def task_ids(self) -> list[str]:
        return list(self.games)

    def make_environment(self) -> WerewolfEnvironment:
        return WerewolfEnvironment(self)

    def metadata(self) -> dict:
        return {
            "name": "werewolf",
            "split": self.split,
            "score_range": [0, 1],
            "role_preset": self.preset,
            "max_days": self.max_days,
            "opponents": "SEEH ScriptedGLM(respect_legal_targets=True)",
            "games": {
                task_id: {"seat": game.seat, "role": game.role, "seed": game.seed}
                for task_id, game in self.games.items()
            },
            "source_sha256": fingerprint(
                {
                    str(path.relative_to(self.source)): sha256_file(path)
                    for path in sorted(self.source.rglob("*.py"))
                }
            ),
        }


class MatchClosed(Exception):
    """The framework stopped a match while it waited for the controlled seat."""


class WerewolfEnvironment:
    def __init__(self, suite: WerewolfSuite):
        self.suite = suite
        self.thread: Thread | None = None
        self.awaiting_action = False
        self.events_path: Path | None = None

    def set_artifact_directory(self, output: Path) -> None:
        self.events_path = output / "environment_events.jsonl"

    def reset(self, task_id: str) -> Observation:
        self.close()
        game = self.suite.games[task_id]
        self.actions: Queue[str | None] = Queue()
        self.states: Queue[Observation | Transition | Exception] = Queue()
        self.thread = Thread(target=self._run_match, args=(task_id, game), name=task_id)
        self.thread.start()
        state = self._receive()
        if not isinstance(state, Observation):
            raise RuntimeError("Werewolf match ended before the controlled seat could act")
        return state

    def step(self, action: str) -> Transition:
        if not self.awaiting_action:
            raise RuntimeError("Werewolf is not waiting for an action")
        self.awaiting_action = False
        self.actions.put(action)
        state = self._receive()
        if isinstance(state, Transition):
            return state
        return Transition(state, reward=0.0, terminated=False, success=False)

    def _receive(self) -> Observation | Transition:
        state = self.states.get()
        if isinstance(state, Exception):
            raise state
        self.awaiting_action = isinstance(state, Observation)
        return state

    def close(self) -> None:
        if self.thread is not None:
            self.actions.put(None)
            self.thread.join()
            self.thread = None
        self.awaiting_action = False

    def _run_match(self, task_id: str, game: Game) -> None:
        try:
            path = self.events_path
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
            event_log = path.open("w", encoding="utf-8") if path is not None else nullcontext()
            with event_log as log:
                transition = self._play_match(task_id, game, log)
            self.states.put(transition)
        except MatchClosed:
            pass
        except Exception as exc:
            self.states.put(exc)

    def _play_match(self, task_id: str, game: Game, log: TextIO | None) -> Transition:
        rules = self.suite.rules
        scripted = rules.ScriptedGLM(respect_legal_targets=True)
        request = None
        faction = "wolves" if game.role == "wolf" else "village"
        goal = f"Play player_{game.seat} as {game.role} and help the {faction} faction win."
        actions, states = self.actions, self.states
        sequence = 0
        tested_alive = True

        def record(kind: str, **fields) -> None:
            nonlocal sequence
            if log is not None:
                sequence += 1
                row = {"seq": sequence, "type": kind, **fields}
                print(json.dumps(row, ensure_ascii=False), file=log, flush=True)

        def on_event(event: dict) -> None:
            nonlocal tested_alive
            alive = match.players[game.seat - 1].alive
            record("match_event", **event, tested_player_eliminated=tested_alive and not alive)
            tested_alive = alive

        class ControlledMatch(rules.SixPlayerWerewolfMatch):
            target_actions = None
            poison_targets = ()

            def _target_action(self, player, day, phase, relation, prompt, label, legal_targets):
                self.target_actions = [f"{label}: {target}" for target in legal_targets]
                try:
                    return super()._target_action(
                        player, day, phase, relation, prompt, label, legal_targets
                    )
                finally:
                    self.target_actions = None

            def _witch_poison_analysis(self, actor, day):
                analysis = super()._witch_poison_analysis(actor, day)
                self.poison_targets = analysis["eligible_targets"]
                return analysis

            def _write_output(self, result):
                # The framework keeps its own trajectories and outcome.
                pass

        def on_action(action):
            nonlocal request
            request = action

        class Policy:
            def complete(self, *, system, prompt, temperature, max_tokens):
                nonlocal request
                decision, request = request, None
                # The match also calls the model for its public speech summary.
                if decision is None or decision["player_id"] != game.seat:
                    return scripted.complete(
                        system=system, prompt=prompt, temperature=temperature, max_tokens=max_tokens
                    )
                legal = match.target_actions
                if game.role == "witch" and decision["phase"] == "night":
                    legal = _witch_actions(match)
                states.put(Observation(prompt, goal, legal))
                action = actions.get()
                if action is None:
                    raise MatchClosed()
                return action

        match = ControlledMatch(
            config=rules.MatchConfig(
                game_id=task_id,
                tested_player_id=game.seat,
                tested_role=game.role,
                tested_backend="direct",
                claims_policy="legacy",
                poison_policy="corroborated",
                update_skills=False,
                role_seed=game.seed,
                roles=self.suite.roles,
                max_days=self.suite.max_days,
                dry_run=True,
            ),
            model=Policy(),
            action_callback=on_action,
            event_callback=on_event if log is not None else None,
        )
        try:
            result = match.run()
        except MatchClosed:
            record("match_aborted", visibility="public", stopped_reason="episode_closed")
            raise
        winner = result["winner"]
        player = match.players[game.seat - 1]
        success = winner == faction
        public_end = (
            f"The {winner} faction won."
            if winner
            else "The day limit was reached without a winner."
        )
        recent = match._visible_events_for(player, limit=match.config.recent_events)
        votes = _public_vote_history(match.events)
        record(
            "match_finished",
            visibility="public",
            winner=winner,
            tested_player=player.name,
            survived=player.alive,
            stopped_reason=result["stopped_reason"],
        )
        return Transition(
            Observation(
                f"Match over. {public_end}\n"
                f"Public vote history:\n{json.dumps(votes, ensure_ascii=False)}\n"
                f"Visible events:\n{json.dumps(recent, ensure_ascii=False)}",
                goal,
                [],
            ),
            reward=float(success),
            terminated=True,
            success=success,
            grade_details={
                "seat": game.seat,
                "role": game.role,
                "faction": faction,
                "winner": winner,
                "survived": player.alive,
                "role_seed": game.seed,
                "stopped_reason": result["stopped_reason"],
            },
        )


def _public_vote_history(events: list[dict]) -> list[dict]:
    ballots: dict[int, dict[str, str]] = {}
    history = []
    for event in events:
        if event["phase"] != "day_vote" or event["visibility"] != "public":
            continue
        day = event["day"]
        if event["relation"] == "vote":
            ballots.setdefault(day, {})[event["actor"]] = event["target"]
        elif event["relation"] == "decision":
            history.append(
                {
                    "day": day,
                    "ballots": ballots.get(day, {}),
                    "tally": json.loads(event["content"]),
                    "exiled": event["target"],
                }
            )
    return history


def _witch_actions(match) -> list[str]:
    killed = match._player_by_name(getattr(match, "pending_wolf_kill", None))
    if match.witch_save_available and killed and killed.alive:
        return [f"WITCH_ACTION: save {killed.name}"]
    actions = ["WITCH_ACTION: skip"]
    if match.witch_poison_available:
        actions.extend(f"WITCH_ACTION: poison {target}" for target in match.poison_targets)
    return actions
