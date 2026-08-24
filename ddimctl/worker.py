"""Scheduler-safe entry point for executing an immutable run manifest."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .run_logging import Heartbeat, MetricLogger, StateStore, read_json, utc_now
from .training import StopController, train_from_manifest


EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_CUDA_OOM = 10
EXIT_INTERRUPTED = 75


def _load_manifest(path: Path) -> Any:
    raw = read_json(path)
    try:
        from .schemas import RunManifest

        # The contracts are strict for Python callers, while a persisted JSON
        # manifest necessarily represents Path/datetime/enum values as strings.
        return RunManifest.model_validate(raw, strict=False)
    except ImportError:
        return raw


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _next_attempt(run_dir: Path, requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("attempt must be a positive integer")
        return requested
    existing: list[int] = []
    attempts = run_dir / "attempts"
    if attempts.exists():
        for child in attempts.iterdir():
            if child.is_dir() and child.name.isdigit():
                existing.append(int(child.name))
    return max(existing, default=0) + 1


def _append_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("{} {}\n".format(utc_now(), message))
        handle.flush()


def verify_dataset_fingerprint(manifest: Any) -> None:
    """Refuse queued/resumed execution if the snapshotted dataset has changed."""

    from .bundles import fingerprint_dataset

    training = _field(manifest, "training")
    dataset = _field(manifest, "dataset")
    if training is None or dataset is None:
        raise ValueError("manifest is missing training or dataset fingerprint data")
    expected_method = str(_field(dataset, "method"))
    actual = fingerprint_dataset(
        Path(_field(dataset, "root")),
        tuple(_field(training, "extensions", (".png", ".tif", ".tiff", ".jpg", ".jpeg"))),
        recursive=bool(_field(training, "recursive", False)),
        hash_contents=expected_method == "sha256-content-v1",
    )
    mismatches = []
    for name in ("file_count", "total_bytes", "sha256", "method"):
        if getattr(actual, name) != _field(dataset, name):
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            "dataset no longer matches the run manifest (changed: {}); create a new run "
            "instead of resuming or executing this queued bundle".format(", ".join(mismatches))
        )


class _SignalHandlers:
    def __init__(self, controller: StopController):
        self.controller = controller
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> "_SignalHandlers":
        selected = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGUSR1"):
            selected.append(signal.SIGUSR1)

        def handle(signum: int, frame: Any) -> None:
            try:
                name = signal.Signals(signum).name
            except Exception:
                name = str(signum)
            self.controller.request("received {}".format(name))

        for selected_signal in selected:
            try:
                self.previous[selected_signal] = signal.getsignal(selected_signal)
                signal.signal(selected_signal, handle)
            except (OSError, RuntimeError, ValueError):
                pass
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        for selected_signal, previous in self.previous.items():
            try:
                signal.signal(selected_signal, previous)
            except (OSError, RuntimeError, ValueError):
                pass


def run_worker(
    run_dir: os.PathLike[str] | str,
    *,
    manifest_path: os.PathLike[str] | str | None = None,
    attempt: int | None = None,
    resume: os.PathLike[str] | str | None = None,
    device: str | torch.device | None = None,
    heartbeat_seconds: float = 30.0,
) -> int:
    root = Path(run_dir).resolve()
    selected_manifest = Path(manifest_path).resolve() if manifest_path else root / "manifest.json"
    if not selected_manifest.is_file():
        print("Manifest does not exist: {}".format(selected_manifest), file=sys.stderr)
        return EXIT_USAGE

    try:
        manifest = _load_manifest(selected_manifest)
        attempt_number = _next_attempt(root, attempt)
    except Exception as error:
        print("Invalid worker input: {}".format(error), file=sys.stderr)
        return EXIT_USAGE

    attempt_dir = root / "attempts" / "{:03d}".format(attempt_number)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    state = StateStore(
        root / "state.json",
        {
            "schema_version": 1,
            "attempt": attempt_number,
            "status": "prepared",
            "updated_at": utc_now(),
        },
    )
    state.update(
        attempt=attempt_number,
        status="running",
        started_at=utc_now(),
        ended_at=None,
        heartbeat_at=utc_now(),
        pid=os.getpid(),
        exit_code=None,
        message=None,
    )
    _append_line(stdout_path, "worker started pid={} manifest={}".format(os.getpid(), selected_manifest))

    if resume is not None:
        resume_path = root / "checkpoints" if str(resume).lower() == "latest" else Path(resume)
    else:
        resume_path = None
    stop_controller = StopController(root / "stop.request")
    def progress(fields: Mapping[str, Any]) -> None:
        # Fine-grained progress is canonical in metrics.jsonl.  state.json stays
        # conformant with the strict AttemptState lifecycle schema.
        return None

    exit_code = EXIT_FAILURE
    try:
        with Heartbeat(state, heartbeat_seconds), _SignalHandlers(stop_controller):
            _append_line(stdout_path, "verifying dataset fingerprint")
            verify_dataset_fingerprint(manifest)
            _append_line(stdout_path, "dataset fingerprint verified")
            canonical = _field(manifest, "canonical_argv", [])
            with MetricLogger(root / "metrics.jsonl", root / "tensorboard") as metrics:
                metrics.add_text("run/canonical_argv", json.dumps(canonical, ensure_ascii=False))
                if metrics.tensorboard_error:
                    _append_line(stdout_path, "TensorBoard disabled: {}".format(metrics.tensorboard_error))
                result = train_from_manifest(
                    manifest,
                    root,
                    resume=resume_path,
                    stop_controller=stop_controller,
                    metric_logger=metrics,
                    progress_callback=progress,
                    device=device,
                )
        if result.status == "completed":
            exit_code = EXIT_SUCCESS
            state.update(
                status="completed",
                ended_at=utc_now(),
                exit_code=exit_code,
                message="training completed at step {}".format(result.global_step),
            )
        else:
            exit_code = EXIT_INTERRUPTED
            state.update(
                status="interrupted",
                ended_at=utc_now(),
                exit_code=exit_code,
                message="{} at step {}".format(stop_controller.reason, result.global_step),
            )
            (root / "stop.request").unlink(missing_ok=True)
        _append_line(stdout_path, "worker finished status={} exit_code={}".format(result.status, exit_code))
        return exit_code
    except BaseException as error:
        details = traceback.format_exc()
        if isinstance(error, FileNotFoundError):
            exit_code = EXIT_DATA
            message = "{}: {}".format(type(error).__name__, error)
        elif isinstance(error, (ValueError, TypeError)):
            exit_code = EXIT_USAGE
            message = "{}: {}".format(type(error).__name__, error)
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            exit_code = EXIT_CUDA_OOM
            message = "CUDA out of memory: {}".format(error)
        elif isinstance(error, KeyboardInterrupt):
            exit_code = EXIT_INTERRUPTED
            message = "worker interrupted before a safe checkpoint could be confirmed"
        else:
            exit_code = EXIT_FAILURE
            message = "{}: {}".format(type(error).__name__, error)

    _append_line(stderr_path, details.rstrip())
    state.update(
        status="interrupted" if exit_code == EXIT_INTERRUPTED else "failed",
        ended_at=utc_now(),
        exit_code=exit_code,
        message=message,
    )
    print(message, file=sys.stderr)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute an immutable DDIM run manifest")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run bundle directory containing manifest.json")
    source.add_argument("--manifest", help="Manifest path (run directory is its parent)")
    parser.add_argument("--attempt", type=int, help="Explicit 1-based attempt number")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume from a checkpoint path; without a value use checkpoints/latest.json",
    )
    parser.add_argument("--device", help="Explicit Torch device, primarily for smoke tests")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Path(args.manifest).resolve() if args.manifest else None
    run_dir = Path(args.run).resolve() if args.run else manifest.parent
    return run_worker(
        run_dir,
        manifest_path=manifest,
        attempt=args.attempt,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
