"""Versioned, atomic checkpoints for resumable training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .run_logging import atomic_write_json, read_json, utc_now


CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            if len(cuda_state) != torch.cuda.device_count():
                raise CheckpointError(
                    "Checkpoint has RNG state for {} CUDA devices, but {} are visible".format(
                        len(cuda_state), torch.cuda.device_count()
                    )
                )
            torch.cuda.set_rng_state_all(cuda_state)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("Invalid RNG state in checkpoint") from error


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_state: Mapping[str, Any] | None,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    sampler_state: Mapping[str, Any],
    config_sha256: str,
    run_id: str,
    scheduler: Any = None,
    scaler: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at": utc_now(),
        "run_id": str(run_id),
        "config_sha256": str(config_sha256),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "ema_state": dict(ema_state) if ema_state is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "sampler_state": dict(sampler_state),
        "extra": dict(extra or {}),
    }
    return payload


def validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
    expected_config_sha256: str | None = None,
) -> None:
    required = {
        "checkpoint_version",
        "run_id",
        "config_sha256",
        "global_step",
        "epoch",
        "batch_in_epoch",
        "model_state",
        "optimizer_state",
        "rng_state",
        "sampler_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointError("Checkpoint is missing fields: {}".format(", ".join(missing)))
    if payload["checkpoint_version"] != CHECKPOINT_VERSION:
        raise CheckpointError(
            "Unsupported checkpoint version {}; expected {}".format(
                payload["checkpoint_version"], CHECKPOINT_VERSION
            )
        )
    if expected_run_id is not None and str(payload["run_id"]) != str(expected_run_id):
        raise CheckpointError(
            "Checkpoint belongs to run {}, not {}".format(payload["run_id"], expected_run_id)
        )
    if expected_config_sha256 is not None and payload["config_sha256"] != expected_config_sha256:
        raise CheckpointError(
            "Resolved training configuration differs from the checkpoint; fork the run instead"
        )


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.{}.{}.tmp".format(destination.name, os.getpid(), uuid.uuid4().hex)
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        checksum = sha256_file(temporary)
        os.replace(temporary, destination)
        return checksum
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")


def save_checkpoint(
    checkpoint_dir: os.PathLike[str] | str,
    payload: Mapping[str, Any],
    *,
    milestone: bool = False,
    is_best: bool = False,
    retain_recovery: int = 2,
) -> Path:
    """Save a checkpoint, update pointers, and retain recent recovery saves.

    Step-triggered snapshots should pass ``milestone=True``.  Elapsed-time and
    stop-request snapshots remain recovery files and are pruned to the newest
    ``retain_recovery`` entries.  Best checkpoints are always protected.
    """

    validate_checkpoint(payload)
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    step = int(payload["global_step"])
    prefix = "step" if milestone else "recovery_step"
    destination = root / "{}_{:09d}.pth".format(prefix, step)
    checksum = _atomic_torch_save(payload, destination)
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "path": destination.name,
        "sha256": checksum,
        "bytes": destination.stat().st_size,
        "step": step,
        "created_at": payload.get("created_at", utc_now()),
        "milestone": bool(milestone),
        "best": bool(is_best),
    }
    atomic_write_json(_metadata_path(destination), metadata)
    atomic_write_json(root / "latest.json", metadata)
    if is_best:
        atomic_write_json(root / "best.json", metadata)
    _prune_recovery_checkpoints(root, retain=max(1, int(retain_recovery)))
    return destination


def _protected_names(root: Path) -> set[str]:
    protected: set[str] = set()
    for pointer_name in ("latest.json", "best.json"):
        pointer = root / pointer_name
        if pointer.exists():
            try:
                protected.add(str(read_json(pointer)["path"]))
            except Exception:
                pass
    return protected


def _prune_recovery_checkpoints(root: Path, *, retain: int) -> None:
    recovery = sorted(
        root.glob("recovery_step_*.pth"),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
        reverse=True,
    )
    protected = _protected_names(root)
    keep = set(path.name for path in recovery[:retain]).union(protected)
    for path in recovery:
        if path.name in keep:
            continue
        path.unlink(missing_ok=True)
        _metadata_path(path).unlink(missing_ok=True)


def _torch_load(path: Path, map_location: Any) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, dict):
        raise CheckpointError("Checkpoint {} is not a named checkpoint".format(path))
    return value


def _candidate_paths(root: Path, pointer: str) -> Iterable[Path]:
    seen: set[Path] = set()
    pointer_path = root / pointer
    if pointer_path.exists():
        try:
            target = root / str(read_json(pointer_path)["path"])
            seen.add(target)
            yield target
        except Exception:
            pass
    candidates = list(root.glob("step_*.pth")) + list(root.glob("recovery_step_*.pth"))
    for candidate in sorted(
        candidates,
        key=lambda item: (item.stat().st_mtime_ns, item.name),
        reverse=True,
    ):
        if candidate not in seen:
            yield candidate


def load_checkpoint(
    checkpoint: os.PathLike[str] | str,
    *,
    map_location: Any = "cpu",
    expected_run_id: str | None = None,
    expected_config_sha256: str | None = None,
    pointer: str = "latest.json",
) -> tuple[dict[str, Any], Path]:
    """Load a checkpoint and fall back from a corrupt latest recovery file."""

    selected = Path(checkpoint)
    candidates = [selected] if selected.is_file() and selected.suffix == ".pth" else None
    if candidates is None:
        root = selected if selected.is_dir() else selected.parent
        candidates = list(_candidate_paths(root, pointer if selected.is_dir() else selected.name))
    failures: list[str] = []
    for path in candidates:
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata_path = _metadata_path(path)
            if metadata_path.exists():
                expected_checksum = read_json(metadata_path).get("sha256")
                if expected_checksum and sha256_file(path) != expected_checksum:
                    raise CheckpointError("SHA-256 mismatch")
            payload = _torch_load(path, map_location)
            validate_checkpoint(
                payload,
                expected_run_id=expected_run_id,
                expected_config_sha256=expected_config_sha256,
            )
            return payload, path
        except Exception as error:
            failures.append("{}: {}".format(path.name, error))
    if not failures:
        raise CheckpointError("No checkpoints found at {}".format(selected))
    raise CheckpointError("No valid checkpoint found ({})".format("; ".join(failures)))


def seconds_since_last_checkpoint(checkpoint_dir: os.PathLike[str] | str) -> float:
    pointer = Path(checkpoint_dir) / "latest.json"
    if not pointer.exists():
        return float("inf")
    try:
        checkpoint = Path(checkpoint_dir) / str(read_json(pointer)["path"])
        return max(0.0, time.time() - checkpoint.stat().st_mtime)
    except Exception:
        return float("inf")
