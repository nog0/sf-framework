from __future__ import annotations

import importlib.metadata
import os
import random
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from sf_framework.config import TaskConfig
from sf_framework.core import Observation, Transition, sha256_file

UPSTREAM_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
_ENV_LOCK = threading.RLock()


def search_document(product):
    """Same document fields as upstream search_engine/convert_product_file_format.py."""
    options = ", and ".join(f"{k}: {', '.join(v)}" for k, v in product.get("options", {}).items())
    contents = " ".join(
        [product["Title"], product["Description"], product["BulletPoints"][0], options]
    ).lower()
    return {"id": product["asin"], "contents": contents, "product": product}


class WebShopSuite:
    default_max_steps = 100

    def __init__(self, config: TaskConfig):
        if config.catalog not in {"full", "small"}:
            raise ValueError("WebShop catalog must be full or small")
        if config.split not in {"train", "validation", "test"}:
            raise ValueError("WebShop split must be train, validation or test")
        self.config, self.root = config, Path(config.root).resolve()
        suffix = "" if config.catalog == "full" else "_1000"
        self.files = [
            self.root / "data" / f"items_shuffle{suffix}.json",
            self.root / "data" / f"items_ins_v2{suffix}.json",
            self.root / "data" / "items_human_ins.json",
        ]
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Run sf setup webshop --root ...")
        index = (
            self.root / "search_engine" / ("indexes" if config.catalog == "full" else "indexes_1k")
        )
        if not list(index.glob("segments_*")):
            raise FileNotFoundError(
                f"Missing Lucene index at {index}. Run sf setup webshop --root ..."
            )
        existing = sys.modules.get("web_agent_site.utils")
        if existing and Path(existing.BASE_DIR).resolve() != self.root / "web_agent_site":
            raise RuntimeError("Use a separate process for each WebShop source checkout")
        sys.path.insert(0, str(self.root))
        java = shutil.which("java")
        if java and not os.environ.get("JAVA_HOME"):
            os.environ["JAVA_HOME"] = str(Path(java).resolve().parent.parent)
        # Upstream captures these defaults at import time; select the catalog first.
        import web_agent_site.utils as utils

        utils.DEFAULT_FILE_PATH, utils.DEFAULT_ATTR_PATH = map(str, self.files[:2])
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        self.env_class = WebAgentTextEnv
        previous_random = random.getstate()
        try:
            random.seed(config.seed)
            self.prototype = WebAgentTextEnv(
                observation_mode="text_rich",
                file_path=str(self.files[0]),
                human_goals=True,
                show_attrs=False,
                num_products=None if config.catalog == "full" else 1000,
            )
        finally:
            random.setstate(previous_random)
        # The official indexer skips whitespace-only documents (60 in the full catalog).
        expected_docs = sum(
            bool(search_document(p)["contents"].strip()) for p in self.prototype.server.all_products
        )
        if self.prototype.server.search_engine.num_docs != expected_docs:
            raise RuntimeError("WebShop Lucene index is incomplete or does not match this catalog")
        n = len(self.prototype.server.goals)
        if config.catalog == "full" and n < 1500:
            raise RuntimeError(f"Full WebShop catalog has only {n} goals")
        self.indices = (
            list(range(min(500, n)))
            if config.split == "test"
            else list(range(500, min(1500, n)))
            if config.split == "validation"
            else list(range(1500, n))
        )
        if not self.indices:
            raise ValueError("This catalog has no tasks in the selected split")

    def task_ids(self):
        return [str(i) for i in self.indices]

    def make_environment(self):
        return WebShopEnvironment(self)

    def metadata(self):
        commit = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(self.root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if commit != UPSTREAM_COMMIT or dirty:
            raise RuntimeError("WebShop official source differs from the pinned clean revision")
        return {
            "name": "webshop",
            "split": self.config.split,
            "catalog": self.config.catalog,
            "root": str(self.root),
            "source_commit": commit,
            "task_count": len(self.indices),
            "product_count": len(self.prototype.server.all_products),
            "instruction_count": len(self.prototype.server.goals),
            "seed": self.config.seed,
            "human_goals": True,
            "observation_mode": "text_rich",
            "images": False,
            "show_attrs": False,
            "search": "official Lucene",
            "index_document_count": self.prototype.server.search_engine.num_docs,
            "files_sha256": {p.name: sha256_file(p) for p in self.files},
            "index_files_sha256": {
                p.name: sha256_file(p)
                for p in sorted(
                    (
                        self.root
                        / "search_engine"
                        / ("indexes" if self.config.catalog == "full" else "indexes_1k")
                    ).iterdir()
                )
                if p.is_file() and p.name != "write.lock"
            },
            "pyserini_version": importlib.metadata.version("pyserini"),
            "default_max_steps": self.default_max_steps,
            "score_range": [0, 1],
            "success": "terminal official reward == 1",
        }


class WebShopEnvironment:
    def __init__(self, suite: WebShopSuite):
        self.suite = suite
        with _ENV_LOCK:
            self.env = suite.env_class(
                observation_mode="text_rich",
                server=suite.prototype.server,
                human_goals=True,
                session_prefix=f"sf_{uuid.uuid4().hex}_",
            )

    def reset(self, task_id):
        index = int(task_id)
        if index not in self.suite.indices:
            raise ValueError(f"Task {task_id} is outside the configured split")
        with _ENV_LOCK:
            prefix = self.env.session_prefix or ""
            self.env.server.user_sessions.pop(f"{prefix}{index}", None)
            text, _ = self.env.reset(session=index)
            return self._observation(text)

    def _observation(self, text):
        valid = self.env.get_available_actions()
        actions = [f"click[{x}]" for x in valid["clickables"] if x != "search"]
        if valid["has_search_bar"]:
            actions.insert(0, "search[YOUR QUERY]")
        return Observation(text, self.env.instruction_text, actions)

    def step(self, action):
        with _ENV_LOCK:
            text, reward, done, _ = self.env.step(action)
            return Transition(
                self._observation(text), float(reward), bool(done), bool(done and reward >= 1.0)
            )

    def close(self):
        with _ENV_LOCK:
            prefix = self.env.session_prefix
            if prefix:
                for key in list(self.env.server.user_sessions):
                    if key.startswith(prefix):
                        self.env.server.user_sessions.pop(key, None)
            self.env.close()
