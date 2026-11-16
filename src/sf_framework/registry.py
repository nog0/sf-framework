"""Built-ins plus module:factory plugins; core code never branches on a new task type."""

from __future__ import annotations

import importlib
from typing import Any

from sf_framework.config import AgentConfig, EvaluationConfig, ExecutionConfig, TaskConfig


def resolve(spec: str, builtins: dict[str, str]) -> Any:
    target = builtins.get(spec, spec)
    module, sep, attr = target.partition(":")
    if not sep:
        raise ValueError(f"Unknown plugin {spec!r}. Use a built-in name or module:factory.")
    return getattr(importlib.import_module(module), attr)


def task_suite(config: TaskConfig):
    return resolve(
        config.kind,
        {
            "alfworld": "sf_framework.tasks.alfworld:ALFWorldSuite",
            "webshop": "sf_framework.tasks.webshop:WebShopSuite",
            "workspace": "sf_framework.tasks.workspace:WorkspaceSuite",
            "skillsbench": "sf_framework.tasks.skillsbench:SkillsBenchSuite",
            "skillflow": "sf_framework.tasks.skillflow:SkillFlowSuite",
            "spreadsheetbench2": "sf_framework.tasks.spreadsheetbench2:SpreadsheetBench2Suite",
            "werewolf": "sf_framework.tasks.werewolf:WerewolfSuite",
        },
    )(config)


def agent(config: AgentConfig):
    return resolve(
        config.kind,
        {
            "codex": "sf_framework.agents.codex:CodexAgent",
            "openai_chat": "sf_framework.agents.openai_chat:ChatAgent",
        },
    )(config)


def execution_backend(config: ExecutionConfig):
    return resolve(
        config.kind,
        {
            "step": "sf_framework.execution:StepExecution",
            "workspace": "sf_framework.execution:WorkspaceExecution",
            "skillsbench": "sf_framework.tasks.skillsbench:SkillsBenchExecution",
            "harbor": "sf_framework.harbor_runtime:HarborExecution",
            "spreadsheetbench2": "sf_framework.tasks.spreadsheetbench2:SpreadsheetBench2Execution",
        },
    )(config)


def evaluator(config: EvaluationConfig):
    return resolve(
        config.kind,
        {
            "environment": "sf_framework.evaluation:EnvironmentEvaluator",
            "file_text": "sf_framework.evaluation:FileTextEvaluator",
            "skillsbench": "sf_framework.tasks.skillsbench:SkillsBenchEvaluator",
            "harbor": "sf_framework.harbor_runtime:HarborEvaluator",
            "spreadsheetbench2": "sf_framework.tasks.spreadsheetbench2:SpreadsheetBench2Evaluator",
        },
    )(config)
