from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class TaskConfig:
    kind: str = "alfworld"
    split: str = "test"
    root: str = "data/alfworld"
    action_mode: str = "admissible"
    catalog: str = "full"
    games_subdir: str = "json_2.1.1"
    seed: int = 233
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    kind: str = "codex"
    model: str | None = None
    auth: str = "existing"
    codex_binary: str = "codex"
    codex_home: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str | None = None
    timeout_seconds: float = 1800
    max_model_calls: int = 200
    max_tool_calls: int = 400
    temperature: float | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    output: str = "runs/default"
    episodes: int | None = 1
    start: int = 0
    task_ids: list[str] = field(default_factory=list)
    max_steps: int | None = None
    workers: int = 1


@dataclass
class ExecutionConfig:
    kind: str = "step"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    kind: str = "environment"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if type(self.run.workers) is not int or self.run.workers < 1:
            raise ValueError("run.workers must be a positive integer")
        if self.task.action_mode not in {"admissible", "generation"}:
            raise ValueError("task.action_mode must be admissible or generation")
        if self.agent.auth not in {"existing", "chatgpt", "api_key", "provider"}:
            raise ValueError("agent.auth must be existing, chatgpt, api_key or provider")
        if self.run.start < 0 or (self.run.episodes is not None and self.run.episodes < 1):
            raise ValueError("start must be nonnegative and episodes must be positive")
        if self.run.max_steps is not None and self.run.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if (
            min(self.agent.timeout_seconds, self.agent.max_model_calls, self.agent.max_tool_calls)
            <= 0
        ):
            raise ValueError("Agent budgets must be positive")
        if self.agent.kind == "openai_chat" and not self.agent.model:
            raise ValueError("openai_chat requires an explicit model")
        if self.agent.base_url and not self.agent.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if self.agent.base_url and ("@" in self.agent.base_url or "?" in self.agent.base_url):
            raise ValueError("Keep credentials out of base_url; use api_key_env")


def load_config(path: Path) -> Config:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    types = {
        "task": TaskConfig,
        "agent": AgentConfig,
        "run": RunConfig,
        "execution": ExecutionConfig,
        "evaluation": EvaluationConfig,
    }
    if unknown := set(data) - types.keys():
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    sections = {}
    for name, cls in types.items():
        values = data.get(name, {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
        sections[name] = cls(**values)
    cfg = Config(**sections)
    # Paths are relative to the config file, not the caller's working directory.
    for obj, key in [(cfg.task, "root"), (cfg.run, "output"), (cfg.agent, "codex_home")]:
        value = getattr(obj, key)
        if value:
            p = Path(value).expanduser()
            setattr(obj, key, str((path.parent / p).resolve() if not p.is_absolute() else p))
    cfg.validate()
    return cfg
