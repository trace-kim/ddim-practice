"""Run provenance: what code, data and environment produced a checkpoint.

An audit of the burst-diffusion experiments (report S9.1) found the run
directories carried a checkpoint and a config but no way to answer "which
commit, which dataset, which environment produced this?" -- the `ddimctl`
workflow records all of that in its run bundles, the standalone burst pipeline
did not. This module closes that gap with a single ``provenance.json`` per run.

What is recorded, and why each item earns its place:

- **git commit + dirty flag + diff hash**: a dirty tree means the commit alone
  does not identify the code, so the diff is hashed too (never stored -- it may
  hold anything) and the dirty file list is named.
- **dataset fingerprint**: the sorted content hashes of every clean source,
  folded into one digest, plus the distinct-content count. This is the item the
  audit's leakage finding turned on: a dataset that silently gained duplicates
  would change this digest.
- **checkpoint hash + step**: identifies the exact weights being reported.
- **environment**: python/torch/CUDA/GPU, because a CUDA or driver change moves
  numerics.

Deliberately NOT recorded: absolute paths beyond the run and dataset roots,
credentials, or the diff contents.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from .config import Config, load_config
from .data import BurstCache, content_key, resolve_burst_dir

PROVENANCE_NAME = "provenance.json"
SCHEMA_VERSION = 1


def _git(*args: str, repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state(repo: Path) -> dict:
    """Commit, branch, and how the working tree differed from it."""
    commit = _git("rev-parse", "HEAD", repo=repo)
    if commit is None:
        return {"available": False}
    status = _git("status", "--porcelain", repo=repo) or ""
    dirty_files = sorted(line[3:] for line in status.splitlines() if line.strip())
    diff = _git("diff", "HEAD", repo=repo) or ""
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        # The diff itself is never stored (it can contain anything); its hash
        # still distinguishes two runs made from the same commit.
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }


def dataset_fingerprint(config: Config) -> dict:
    """Content digest of the dataset's clean sources plus split composition."""
    cache = BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=1,
        min_size=1,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        split_seed=config.data.split_seed,
    )
    keys = sorted(content_key(source.clean) for source in cache.all_sources)
    digest = hashlib.sha256("".join(keys).encode("ascii")).hexdigest()
    return {
        "burst_dir": str(resolve_burst_dir(config.data.dataset_dir)),
        "sources": len(keys),
        "distinct_contents": len(set(keys)),
        "content_digest_sha256": digest,
        "split": {
            "train": [s.source_index for s in cache.train_sources],
            "val": [s.source_index for s in cache.val_sources],
            "test": [s.source_index for s in cache.test_sources],
        },
        "duplicate_groups": [list(group) for group in cache.duplicate_groups],
    }


def environment_state() -> dict:
    cuda = {
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": cuda,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(
    run_dir: str | Path,
    config: Config,
    *,
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    command: str | None = None,
    repo: str | Path | None = None,
) -> Path:
    """Write ``provenance.json`` into ``run_dir``; returns its path."""
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo) if repo is not None else Path(__file__).resolve().parent.parent

    checkpoint_record: dict | None = None
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_file():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint_record = {
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "step": int(payload.get("step", -1)),
                "bytes": checkpoint_path.stat().st_size,
            }

    config_json = config.model_dump(mode="json")
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(destination),
        "config_path": str(config_path) if config_path is not None else None,
        "config": config_json,
        "config_sha256": hashlib.sha256(
            json.dumps(config_json, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "git": git_state(repo_root),
        "dataset": dataset_fingerprint(config),
        "environment": environment_state(),
        "checkpoint": checkpoint_record,
        "command": command,
    }
    path = destination / PROVENANCE_NAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def record_run(
    config_path: str | Path, *, checkpoint: str | Path | None = None, command: str | None = None
) -> Path:
    """Write provenance for the run described by a config file."""
    config = load_config(config_path)
    run_dir = Path(config.training.run_dir)
    resolved = checkpoint
    if resolved is None:
        candidate = run_dir / "ckpt_latest.pt"
        resolved = candidate if candidate.is_file() else None
    return write_provenance(
        run_dir,
        config,
        config_path=config_path,
        checkpoint=resolved,
        command=command,
    )
