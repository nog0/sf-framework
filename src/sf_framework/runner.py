from __future__ import annotations

import copy
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from sf_framework.config import Config, EvaluationConfig
from sf_framework.core import (
    AgentResult,
    EpisodeCancelled,
    EvaluationInput,
    fingerprint,
    sha256_file,
    write_json,
)
from sf_framework.registry import agent, evaluator, execution_backend, task_suite


def summarize(results: list[dict], planned: int, score_range=(0, 1)) -> dict:
    n = len(results)
    successes = sum(bool(x["success"]) for x in results)
    p = successes / n if n else 0
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n) if n else 0
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else 0
    mean_score = sum(r["score"] for r in results) / n if n else None
    normalized = None
    if mean_score is not None and score_range is not None:
        lo, hi = score_range
        if hi > lo:
            normalized = 100 * (mean_score - lo) / (hi - lo)
    calls = [r.get("agent", {}).get("model_calls") for r in results]
    steps = [r["steps"] for r in results if r["steps"] is not None]
    return {
        "planned": planned,
        "evaluated": n,
        "complete": n == planned and all(r["agent_status"] != "interrupted" for r in results),
        "successes": successes,
        "success_rate": p if n else None,
        "success_rate_wilson_95": [center - half, center + half] if n else None,
        "mean_score": mean_score,
        "mean_score_0_100": normalized,
        "score_range": score_range,
        "total_observed_model_calls": sum(x for x in calls if x is not None),
        "mean_steps": sum(steps) / len(steps) if steps else None,
        "agent_statuses": dict(Counter(r["agent_status"] for r in results)),
        "stopped_reasons": dict(Counter(r["stopped_reason"] for r in results)),
        "results": results,
    }


def _identity(manifest: dict) -> dict:
    """Worker count is a scheduling choice, not an episode's evaluation condition."""
    identity = copy.deepcopy(manifest)
    identity.pop("fingerprint", None)
    identity.get("config", {}).get("run", {}).pop("workers", None)
    return identity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def run(config: Config, *, resume=False, retry_errors=False) -> dict:
    # Advisory process lock prevents two evaluations from writing the same run.
    import fcntl

    config.validate()
    output = Path(config.run.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another evaluation is using {output}") from exc
        try:
            return _run(config, resume=resume, retry_errors=retry_errors)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _execute_episode(
    suite,
    executor,
    evaluation: EvaluationConfig,
    driver_factory,
    index: int,
    task_id: str,
    planned: int,
    max_steps: int | None,
    output: Path,
    stop_event: threading.Event,
) -> dict | None:
    if stop_event.is_set():
        return None
    episode = output / "episodes" / f"{index:05d}"
    episode.mkdir(parents=True, exist_ok=True)
    attempt_number = 1
    while (episode / f"attempt_{attempt_number:03d}").exists():
        attempt_number += 1
    attempt = episode / f"attempt_{attempt_number:03d}"
    attempt.mkdir()
    worker = threading.current_thread().name
    print(f"[{index + 1}/{planned} {worker}] {task_id}", flush=True)
    started, started_at = time.monotonic(), _now()
    session = None
    outcome = AgentResult("error")
    assessment = {"kind": evaluation.kind, "status": "skipped"}
    score, success = 0.0, False

    def close_session():
        try:
            session.close()
        except Exception as exc:
            outcome.details["cleanup_error"] = f"{type(exc).__name__}: {exc}"

    with ExitStack() as cleanup:
        try:
            driver = driver_factory()
            session = executor.start(
                suite,
                task_id,
                attempt,
                max_steps,
                stop_event,
                f"{index + 1}/{planned} {worker}",
            )
            cleanup.callback(close_session)
            outcome = driver.run(session, attempt)
        except (KeyboardInterrupt, EpisodeCancelled):
            stop_event.set()
            outcome = AgentResult("interrupted")
        except Exception as exc:
            outcome = AgentResult("error", message=f"{type(exc).__name__}: {exc}")

        context = getattr(session, "evaluation_context", None)
        if context is None:
            cleanup.close()
        if (
            session is not None
            and outcome.status != "interrupted"
            and not (session.workspace is not None and "cleanup_error" in outcome.details)
        ):
            try:
                grader = evaluator(copy.deepcopy(evaluation))
                grade = grader.evaluate(
                    EvaluationInput(
                        task_id,
                        attempt,
                        outcome,
                        session.environment_grade,
                        session.workspace,
                        evaluation_context=context,
                        task_suite=suite,
                    )
                )
                if not math.isfinite(grade.score) or type(grade.success) is not bool:
                    raise ValueError("Evaluator must return a finite score and a boolean success")
                score, success = float(grade.score), grade.success
                assessment.update(status="completed", details=grade.details)
            except (KeyboardInterrupt, EpisodeCancelled):
                stop_event.set()
                outcome.status = "interrupted"
            except Exception as exc:
                assessment.update(status="error", error=f"{type(exc).__name__}: {exc}")

    if session is not None:
        result = session.execution_result(outcome)
    else:
        result = {
            "task_id": task_id,
            "steps": 0 if max_steps is not None else None,
            "agent_status": outcome.status,
            "stopped_reason": "interrupted"
            if outcome.status == "interrupted"
            else "environment_error",
            "error": outcome.message,
        }
    result.update(success=success, score=score, evaluation=assessment)
    result.update(
        episode_index=index,
        worker=worker,
        started_at=started_at,
        finished_at=_now(),
        elapsed_seconds=time.monotonic() - started,
        attempt_directory=str(attempt.relative_to(output)),
    )
    # A task owns its result and trajectory files. Only the coordinator writes summaries.
    write_json(attempt / "result.json", result)
    write_json(episode / "result.json", result)
    print(
        f"  [{index + 1}/{planned} {worker}] success={result['success']} "
        f"score={result['score']:.3f} steps={result['steps']} status={outcome.status}",
        flush=True,
    )
    return result


def _run(config: Config, *, resume=False, retry_errors=False) -> dict:
    started, started_at = time.monotonic(), _now()
    output = Path(config.run.output)
    if any(p.name != ".run.lock" for p in output.iterdir()) and not resume:
        raise ValueError(f"Output directory is not empty: {output}. Use --resume or a new path")
    suite = task_suite(config.task)
    available = suite.task_ids()
    if config.run.task_ids:
        selected = config.run.task_ids
        if len(set(selected)) != len(selected) or any(x not in available for x in selected):
            raise ValueError("task_ids must be distinct ids from the configured split")
    else:
        count = config.run.episodes
        selected = available[config.run.start : None if count is None else config.run.start + count]
        if not selected or (count is not None and len(selected) != count):
            raise ValueError(f"Requested range exceeds the {len(available)} available tasks")
    executor = execution_backend(config.execution)
    max_steps = executor.max_steps(suite, config.run.max_steps)
    if config.evaluation.kind == "environment" and config.execution.kind == "workspace":
        raise ValueError("Workspace execution requires a workspace evaluator, such as file_text")
    grader = evaluator(config.evaluation)
    environment_metadata = suite.metadata()
    coordinator_driver = agent(config.agent)
    source = Path(__file__).parent
    manifest = {
        "schema_version": 3,
        "config": config.as_dict(),
        "task_ids": selected,
        "max_steps": max_steps,
        "environment": environment_metadata,
        "execution": executor.metadata(),
        "evaluation": grader.metadata(environment_metadata),
        "agent_settings": coordinator_driver.metadata()
        if hasattr(coordinator_driver, "metadata")
        else {},
        "plugins_sha256": {
            name: sha256_file(Path(inspect.getfile(type(obj))))
            for name, obj in [
                ("task", suite),
                ("agent", coordinator_driver),
                ("execution", executor),
                ("evaluation", grader),
            ]
        },
        "framework_sha256": fingerprint(
            {str(p.relative_to(source)): sha256_file(p) for p in sorted(source.rglob("*.py"))}
        ),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "packages": dict(
            sorted(
                (p.metadata["Name"], p.version)
                for p in importlib.metadata.distributions()
                if p.metadata["Name"]
            )
        ),
        "protocol": "fresh agent and execution session per episode; independent task evaluator",
    }
    if config.agent.kind == "codex":
        manifest["codex_version"] = subprocess.check_output(
            [config.agent.codex_binary, "--version"], text=True
        ).strip()
    manifest["fingerprint"] = fingerprint(_identity(manifest))
    score_range = manifest["evaluation"].get("score_range")
    saved = output / "manifest.json"
    if saved.exists():
        if _identity(json.loads(saved.read_text())) != _identity(manifest):
            raise ValueError(
                "Resume rejected: config, tasks, environment, dependencies or framework changed"
            )
    else:
        write_json(saved, manifest)

    results: dict[int, dict] = {}
    jobs = []
    for index, task_id in enumerate(selected):
        path = output / "episodes" / f"{index:05d}" / "result.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("task_id") != task_id:
                raise ValueError(f"Saved task id differs from manifest at episode {index}")
            if hasattr(coordinator_driver, "restore_result"):
                coordinator_driver.restore_result(prior)
            needs_retry = prior["agent_status"] == "interrupted" or (
                retry_errors
                and (
                    prior["agent_status"] in {"error", "failed", "timeout"}
                    or prior.get("evaluation", {}).get("status") == "error"
                )
            )
            if not needs_retry:
                results[index] = prior
                continue
        jobs.append((index, task_id))
    reused = len(results)
    stop_event = threading.Event()
    if jobs and config.run.workers > 1 and hasattr(coordinator_driver, "prepare"):
        coordinator_driver.prepare(output)

    def driver_factory():
        if config.run.workers == 1:
            return coordinator_driver
        if hasattr(coordinator_driver, "for_worker"):
            return coordinator_driver.for_worker()
        return agent(copy.deepcopy(config.agent))

    invocation_id = uuid.uuid4().hex
    _append_event(
        output / "invocations.jsonl",
        {
            "event": "start",
            "id": invocation_id,
            "started_at": started_at,
            "pid": os.getpid(),
            "workers": config.run.workers,
            "pending": len(jobs),
            "reused": reused,
        },
    )

    def save_summary():
        report = summarize([results[i] for i in sorted(results)], len(selected), score_range)
        report["execution"] = {
            "workers": config.run.workers,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "new_results": len(results) - reused,
            "reused_results": reused,
            "interrupted": stop_event.is_set(),
        }
        write_json(output / "summary.json", report)
        return report

    def collect(row):
        if row is not None:
            results[row["episode_index"]] = row
            save_summary()

    print(f"Workers: {config.run.workers}; pending: {len(jobs)}; reused: {reused}", flush=True)
    interrupted = False
    try:
        if config.run.workers == 1:
            for index, task_id in jobs:
                collect(
                    _execute_episode(
                        suite,
                        executor,
                        config.evaluation,
                        driver_factory,
                        index,
                        task_id,
                        len(selected),
                        max_steps,
                        output,
                        stop_event,
                    )
                )
                if stop_event.is_set():
                    raise KeyboardInterrupt
        elif jobs:
            pool = ThreadPoolExecutor(
                max_workers=config.run.workers, thread_name_prefix="sf-worker"
            )
            futures = {}
            try:
                for index, task_id in jobs:
                    future = pool.submit(
                        _execute_episode,
                        suite,
                        executor,
                        config.evaluation,
                        driver_factory,
                        index,
                        task_id,
                        len(selected),
                        max_steps,
                        output,
                        stop_event,
                    )
                    futures[future] = index
                for future in as_completed(futures):
                    collect(future.result())
                    if stop_event.is_set():
                        raise KeyboardInterrupt
            except BaseException:
                stop_event.set()
                for future in futures:
                    future.cancel()
                # Running tasks save their partial attempts while queued work is canceled.
                for future, index in futures.items():
                    if not future.cancelled() and index not in results:
                        collect(future.result())
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        summary = save_summary()
        _append_event(
            output / "invocations.jsonl",
            {
                "event": "end",
                "id": invocation_id,
                "finished_at": _now(),
                "workers": config.run.workers,
                "elapsed_seconds": time.monotonic() - started,
                "interrupted": interrupted,
                "new_results": len(results) - reused,
                "reused": reused,
            },
        )
    if interrupted:
        raise KeyboardInterrupt
    return summary
