"""Pinned public sources. No hidden small-catalog fallback or modified environment rules."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from sf_framework.core import sha256_file, write_json

WEB_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
HF_REVISION = "0129d4a81dbdb827e76afd20a1e2c38b61098613"
HASHES = {
    "items_shuffle.json": "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
    "items_ins_v2.json": "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e",
    "items_human_ins.json": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
}
ALF_URL = "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
ALF_SHA256 = "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"


def download(url: str, path: Path):
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temp.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temp.replace(path)


def setup_alfworld(root: Path):
    archive = root / "downloads" / ALF_URL.rsplit("/", 1)[-1]
    print(f"Downloading official compiled text games: {ALF_URL}", flush=True)
    download(ALF_URL, archive)
    if sha256_file(archive) != ALF_SHA256:
        raise ValueError("Official ALFWorld archive checksum mismatch")
    with zipfile.ZipFile(archive) as games:
        for entry in games.infolist():
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive member")
        games.extractall(root)
    # The official 0.4.2 archive name is 2.1.3; retain its actual directory names.
    game_files = list(root.rglob("game.tw-pddl"))
    if not game_files:
        raise RuntimeError("Official archive contained no game.tw-pddl files")
    write_json(
        root / "provenance.json",
        {
            "source": ALF_URL,
            "archive_sha256": sha256_file(archive),
            "games": len(game_files),
            "subdirectories": sorted({str(p.relative_to(root).parts[0]) for p in game_files}),
        },
    )
    print(f"ALFWorld: {len(game_files)} compiled games; see {root / 'provenance.json'}", flush=True)


def setup_webshop(root: Path, catalog="full"):
    from huggingface_hub import hf_hub_download

    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/princeton-nlp/WebShop.git", str(root)], check=True
        )
        subprocess.run(["git", "-C", str(root), "checkout", WEB_COMMIT], check=True)
    if not (root / ".git").exists():
        raise ValueError(f"Expected the official git checkout at {root}")
    current = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != WEB_COMMIT:
        # Do not overwrite an existing checkout or the user's changes.
        raise ValueError(f"WebShop must be at {WEB_COMMIT}; found {current}")
    suffix = "" if catalog == "full" else "_1000"
    files = [f"items_shuffle{suffix}.json", f"items_ins_v2{suffix}.json", "items_human_ins.json"]
    data = root / "data"
    checksums = {}
    for name in files:
        path = data / name
        if not path.exists():
            hf_hub_download(
                repo_id="HongbangYuan/webshop",
                repo_type="dataset",
                revision=HF_REVISION,
                filename=name,
                local_dir=data,
            )
        checksums[name] = sha256_file(path)
        if name in HASHES and checksums[name] != HASHES[name]:
            raise ValueError(f"Dataset checksum mismatch for {name}")
        print(f"Verified {name}", flush=True)
    write_json(
        data / "provenance.json",
        {
            "source_commit": WEB_COMMIT,
            "mirror": "HongbangYuan/webshop",
            "mirror_revision": HF_REVISION,
            "catalog": catalog,
            "sha256": checksums,
        },
    )
    index = root / "search_engine" / ("indexes" if catalog == "full" else "indexes_1k")
    if list(index.glob("segments_*")):
        print(f"WebShop {catalog} data and official Lucene index ready at {root}", flush=True)
        return

    java = shutil.which("java")
    if java and not os.environ.get("JAVA_HOME"):
        os.environ["JAVA_HOME"] = str(Path(java).resolve().parent.parent)
    sys.path.insert(0, str(root))
    import web_agent_site.utils as utils

    utils.DEFAULT_FILE_PATH = str(data / files[0])
    utils.DEFAULT_ATTR_PATH = str(data / files[1])
    from web_agent_site.engine.engine import load_products

    from sf_framework.tasks.webshop import search_document

    products, *_ = load_products(
        filepath=utils.DEFAULT_FILE_PATH, num_products=None if catalog == "full" else 1000
    )
    with tempfile.TemporaryDirectory(prefix="sf_resources_", dir=index.parent) as resources:
        with (Path(resources) / "documents.jsonl").open("w") as stream:
            for i, product in enumerate(products):
                stream.write(json.dumps(search_document(product)) + "\n")
                if i % 100000 == 0:
                    print(f"Converted {i}/{len(products)} products", flush=True)
        del products
        # Match the upstream indexer, including its single indexing thread.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                resources,
                "--index",
                str(index),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                "1",
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ],
            check=True,
        )
    print(f"WebShop {catalog} data and official Lucene index ready at {root}", flush=True)
