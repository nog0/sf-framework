from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sf_framework.config import load_config


def parser():
    p = argparse.ArgumentParser(prog="sf", description="Run existing agents on reproducible tasks")
    commands = p.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run", help="Run/resume an evaluation")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--model")
    evaluate.add_argument("--agent", dest="agent_kind", help="codex, openai_chat or module:factory")
    evaluate.add_argument("--auth", choices=["existing", "chatgpt", "api_key", "provider"])
    evaluate.add_argument("--base-url")
    evaluate.add_argument("--api-key-env")
    evaluate.add_argument("--codex-home", type=Path)
    evaluate.add_argument("--reasoning-effort")
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--all", action="store_true", help="Run the full configured split")
    evaluate.add_argument("--task-id", action="append", help="Select a task by ID (repeatable)")
    evaluate.add_argument("--start", type=int)
    evaluate.add_argument("--max-steps", type=int)
    evaluate.add_argument(
        "--workers", type=int, help="Concurrent task workers (default: config or 1)"
    )
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--retry-errors", action="store_true")
    tasks = commands.add_parser("tasks", help="List actual ids from an official split")
    tasks.add_argument("--config", type=Path, required=True)
    tasks.add_argument("--limit", type=int, default=10)
    setup = commands.add_parser(
        "setup", help="Download official task assets and build search index"
    )
    setup.add_argument("benchmark", choices=["alfworld", "webshop"])
    setup.add_argument("--root", type=Path)
    setup.add_argument("--catalog", choices=["full", "small"], default="full")
    doctor = commands.add_parser("doctor", help="Inspect installed tools; makes no model calls")
    doctor.add_argument("--config", type=Path)
    auth = commands.add_parser("auth", help="Delegate authentication to the official Codex CLI")
    auth_sub = auth.add_subparsers(dest="action", required=True)
    for action in ["login", "status"]:
        sub = auth_sub.add_parser(action)
        sub.add_argument("--codex-home", type=Path)
        sub.add_argument("--codex-binary", default="codex")
        if action == "login":
            sub.add_argument("--method", choices=["chatgpt", "api-key"], default="chatgpt")
            sub.add_argument("--device-auth", action="store_true")
            sub.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "auth":
            env = os.environ.copy()
            if args.codex_home:
                env["CODEX_HOME"] = str(args.codex_home.expanduser().resolve())
            cmd = [args.codex_binary, "login"]
            stdin = None
            if args.action == "status":
                cmd.append("status")
            elif args.method == "api-key":
                if args.device_auth:
                    raise ValueError("--device-auth applies only to ChatGPT login")
                key = os.environ.get(args.api_key_env)
                if not key:
                    raise ValueError(
                        f"Set {args.api_key_env}; keys are read from environment, never CLI arguments"
                    )
                cmd.append("--with-api-key")
                stdin = key + "\n"
            elif args.device_auth:
                cmd.append("--device-auth")
            return subprocess.run(cmd, env=env, input=stdin, text=True).returncode
        if args.command == "setup":
            from sf_framework.setup_data import setup_alfworld, setup_webshop

            root = (
                args.root
                or Path("data/alfworld" if args.benchmark == "alfworld" else "vendor/WebShop")
            ).resolve()
            if args.benchmark == "alfworld":
                setup_alfworld(root)
            else:
                setup_webshop(root, args.catalog)
            return 0
        if args.command == "doctor":
            report = {
                "python": sys.version,
                "python_executable": sys.executable,
                "codex": shutil.which("codex"),
                "java": shutil.which("java"),
            }
            for package in [
                "sf-framework",
                "alfworld",
                "textworld",
                "pyserini",
                "openai",
                "torch",
                "spacy",
            ]:
                try:
                    report[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    report[package] = None
            if args.config:
                from sf_framework.registry import task_suite

                config = load_config(args.config.resolve())
                suite = task_suite(config.task)
                report["task"] = suite.metadata()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        config = load_config(args.config.resolve())
        if args.command == "tasks":
            from sf_framework.registry import task_suite

            ids = task_suite(config.task).task_ids()
            print(json.dumps({"count": len(ids), "task_ids": ids[: args.limit]}, indent=2))
            return 0
        from sf_framework.runner import run

        for arg, field in [
            (args.episodes, "episodes"),
            (args.start, "start"),
            (args.max_steps, "max_steps"),
            (args.workers, "workers"),
        ]:
            if arg is not None:
                setattr(config.run, field, arg)
        if args.all:
            config.run.episodes = None
        if args.task_id:
            if args.all or args.episodes is not None or args.start is not None:
                raise ValueError("--task-id cannot be combined with --all, --episodes or --start")
            config.run.task_ids = args.task_id
        if args.output:
            config.run.output = str(args.output.resolve())
        if args.model:
            config.agent.model = args.model
        for value, name in [
            (args.agent_kind, "kind"),
            (args.auth, "auth"),
            (args.base_url, "base_url"),
            (args.api_key_env, "api_key_env"),
            (args.reasoning_effort, "reasoning_effort"),
        ]:
            if value is not None:
                setattr(config.agent, name, value)
        if args.codex_home:
            config.agent.codex_home = str(args.codex_home.expanduser().resolve())
        if config.agent.kind == "openai_chat":
            # This backend always uses API credentials, even when its task
            # settings were inherited from an account-authenticated Codex config.
            config.agent.auth = "api_key"
        result = run(config, resume=args.resume, retry_errors=args.retry_errors)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
        return 1 if result["agent_statuses"].get("error") else 0
    except KeyboardInterrupt:
        print("Interrupted; completed episodes and partial trajectory saved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"sf: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
