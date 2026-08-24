"""Durable, tracker-independent logging for DDIM runs.

The JSON files in a run bundle are the source of truth.  TensorBoard is a
best-effort view over the same events: an unavailable or broken TensorBoard
installation must never stop training.
"""

from __future__ import annotations

import json
import math
import os
import threading
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".{}.{}.tmp".format(destination.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in {}".format(path))
    return value


class StateStore:
    """Serialize worker state updates made by the trainer and heartbeat thread."""

    def __init__(self, path: os.PathLike[str] | str, initial: Mapping[str, Any] | None = None):
        self.path = Path(path)
        self._lock = threading.RLock()
        if initial is not None and not self.path.exists():
            payload = dict(initial)
            payload.setdefault("updated_at", utc_now())
            atomic_write_json(self.path, payload)

    def read(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return {}
            return read_json(self.path)

    def update(self, **fields: Any) -> dict[str, Any]:
        with self._lock:
            state = self.read()
            state.update(fields)
            state["updated_at"] = utc_now()
            atomic_write_json(self.path, state)
            return state


class Heartbeat(AbstractContextManager["Heartbeat"]):
    """Periodically refresh an AttemptState heartbeat without touching training."""

    def __init__(self, state: StateStore, interval_seconds: float = 30.0):
        self.state = state
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Heartbeat":
        self.state.update(heartbeat_at=utc_now())
        self._thread = threading.Thread(target=self._run, name="ddim-heartbeat", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.state.update(heartbeat_at=utc_now())
            except Exception:
                # A heartbeat is advisory; the trainer's foreground state writes
                # still report a filesystem problem with the real exception.
                pass

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(5.0, self.interval_seconds + 0.5))


class MetricLogger(AbstractContextManager["MetricLogger"]):
    """Append canonical JSONL metrics and mirror supported values to TensorBoard."""

    def __init__(
        self,
        metrics_path: os.PathLike[str] | str,
        tensorboard_dir: os.PathLike[str] | str | None = None,
        *,
        flush_each_event: bool = True,
    ):
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.metrics_path.open("a", encoding="utf-8", newline="\n", buffering=1)
        self._lock = threading.Lock()
        self.flush_each_event = flush_each_event
        self.tensorboard_error: str | None = None
        self._writer: Any = None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(tensorboard_dir))
            except Exception as error:  # optional dependency or filesystem policy
                self.tensorboard_error = "{}: {}".format(type(error).__name__, error)

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and value.numel() == 1:
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    def log(self, event: str, step: int, values: Mapping[str, Any]) -> None:
        cleaned = {key: self._clean_value(value) for key, value in values.items()}
        record = {"time": utc_now(), "event": event, "step": int(step), **cleaned}
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self._lock:
            self._handle.write(encoded + "\n")
            if self.flush_each_event:
                self._handle.flush()
                os.fsync(self._handle.fileno())

        if self._writer is not None:
            try:
                for key, value in cleaned.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self._writer.add_scalar("{}/{}".format(event, key), value, int(step))
            except Exception as error:
                self.tensorboard_error = "{}: {}".format(type(error).__name__, error)
                try:
                    self._writer.close()
                finally:
                    self._writer = None

    def add_image(self, tag: str, image: Any, step: int) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_image(tag, image, int(step))
        except Exception as error:
            self.tensorboard_error = "{}: {}".format(type(error).__name__, error)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_text(tag, text, int(step))
        except Exception as error:
            self.tensorboard_error = "{}: {}".format(type(error).__name__, error)

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
