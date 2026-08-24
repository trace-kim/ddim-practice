"""Optional experiment publication and terminal inspection helpers.

The portable run directory is always authoritative.  Importing this module does
not import MLflow; the dependency is loaded only when publication is requested.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit


class TrackingError(RuntimeError):
    """Raised by explicit tracking operations, never by the training worker."""


@dataclass(frozen=True)
class PublishResult:
    run_id: str
    experiment_id: str
    bundle_id: str
    fingerprint: str
    tracking_uri: str
    reused: bool
    params_logged: int = 0
    metrics_logged: int = 0
    artifacts_logged: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "bundle_id": self.bundle_id,
            "fingerprint": self.fingerprint,
            "tracking_uri": self.tracking_uri,
            "reused": self.reused,
            "params_logged": self.params_logged,
            "metrics_logged": self.metrics_logged,
            "artifacts_logged": self.artifacts_logged,
        }


@dataclass(frozen=True)
class TrackingServerProcess:
    argv: tuple[str, ...]
    url: str
    pid: int
    stdout_path: Path
    stderr_path: Path
    returncode: int | None = None


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    run_id: str
    label: str
    state: str
    global_step: int | None
    max_steps: int | None
    latest_loss: float | None
    latest_validation_loss: float | None
    backend: str | None
    started_at: str | None
    updated_at: str | None
    detail: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float | None:
        if self.global_step is None or not self.max_steps:
            return None
        return min(1.0, max(0.0, self.global_step / self.max_steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "label": self.label,
            "state": self.state,
            "global_step": self.global_step,
            "max_steps": self.max_steps,
            "latest_loss": self.latest_loss,
            "latest_validation_loss": self.latest_validation_loss,
            "backend": self.backend,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "detail": self.detail,
            "extra": dict(self.extra),
        }

    def __str__(self) -> str:
        return format_run_summary(self)


def _read_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise TrackingError(f"required run-bundle file is missing: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackingError(f"expected a JSON object in {path}")
    return payload


def _resolved_config(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("resolved_config", "training_spec", "config", "training"):
        value = manifest.get(name)
        if isinstance(value, Mapping):
            return dict(value)
    config_path = root / "resolved_config.yml"
    if not config_path.is_file():
        config_path = root / "resolved_config.yaml"
    if not config_path.is_file():
        return {}
    try:
        yaml = importlib.import_module("yaml")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        return {}
    except Exception as exc:
        raise TrackingError(f"could not parse {config_path}: {exc}") from exc
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _flatten_params(value: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for raw_key, item in value.items():
        key = f"{prefix}.{raw_key}" if prefix else str(raw_key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_params(item, key))
        elif isinstance(item, (list, tuple)):
            flattened[key] = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
        elif item is None:
            flattened[key] = "null"
        elif isinstance(item, bool):
            flattened[key] = "true" if item else "false"
        else:
            flattened[key] = str(item)
    return flattened


def _bundle_identity(root: Path, manifest: Mapping[str, Any]) -> tuple[str, str]:
    bundle_id = str(
        manifest.get("run_id")
        or manifest.get("id")
        or manifest.get("bundle_id")
        or root.name
    )
    digest = hashlib.sha256()
    for name in (
        "manifest.json",
        "resolved_config.yml",
        "resolved_config.yaml",
        "argv.json",
        "state.json",
        "metrics.jsonl",
    ):
        path = root / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    samples = root / "samples"
    if samples.is_dir():
        for path in sorted(
            (item for item in samples.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(samples).as_posix(),
        ):
            relative = path.relative_to(samples).as_posix()
            digest.update(f"samples/{relative}".encode("utf-8"))
            digest.update(path.read_bytes())
    if digest.digest() == hashlib.sha256().digest():
        digest.update(bundle_id.encode("utf-8"))
    return bundle_id, digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _mlflow_module(override: Any | None = None) -> Any:
    if override is not None:
        return override
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as exc:
        raise TrackingError(
            "MLflow is not installed. Install the optional tracking dependencies on "
            "the machine that will host the local catalog."
        ) from exc


def _safe_tracking_uri(value: str) -> str:
    if any(character in value for character in "\r\n\0"):
        raise TrackingError("tracking URI must be a single line")
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise TrackingError("tracking URI must not embed credentials")
    return value


def _tracking_client(mlflow: Any, tracking_uri: str | None) -> tuple[Any, str]:
    if tracking_uri:
        mlflow.set_tracking_uri(_safe_tracking_uri(tracking_uri))
    effective_uri = _safe_tracking_uri(str(mlflow.get_tracking_uri()))
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=effective_uri)
    except TypeError:
        client = mlflow.tracking.MlflowClient()
    return client, effective_uri


def _experiment_id(client: Any, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return str(experiment.experiment_id)
    try:
        return str(client.create_experiment(name))
    except Exception:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise
        return str(experiment.experiment_id)


def _find_published_runs(
    client: Any,
    experiment_id: str,
    fingerprint: str,
    bundle_id: str,
) -> tuple[Any | None, Any | None]:
    filter_string = f"tags.`ddim.bundle_fingerprint` = '{fingerprint}'"
    try:
        matches = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=1000,
        )
    except (AttributeError, TypeError):
        return None, None
    completed: list[Any] = []
    unfinished: list[Any] = []
    unsafe: list[str] = []
    for run in matches:
        tags = getattr(getattr(run, "data", None), "tags", {}) or {}
        if str(tags.get("ddim.bundle_fingerprint")) != fingerprint:
            continue
        completion = str(tags.get("ddim.publication_complete", "")).casefold()
        if completion == "true":
            completed.append(run)
            continue
        run_id = str(getattr(getattr(run, "info", None), "run_id", "unknown"))
        tagged_bundle_id = str(tags.get("ddim.bundle_id", ""))
        run_status = str(getattr(getattr(run, "info", None), "status", "")).upper()
        if completion != "false":
            unsafe.append(f"{run_id} has no explicit incomplete-publication tag")
        elif tagged_bundle_id != bundle_id:
            unsafe.append(f"{run_id} belongs to bundle {tagged_bundle_id or 'unknown'}")
        elif run_status not in {"", "RUNNING"}:
            unsafe.append(f"{run_id} has MLflow status {run_status}")
        else:
            unfinished.append(run)
    if completed:
        return completed[0], None
    candidate_count = len(unfinished) + len(unsafe)
    if candidate_count > 1:
        raise TrackingError(
            "found multiple incomplete MLflow runs for the same bundle fingerprint; "
            "refusing to choose one automatically"
        )
    if unsafe:
        raise TrackingError(
            "found an incomplete MLflow run for the same bundle fingerprint that "
            f"cannot be safely resumed: {unsafe[0]}"
        )
    return None, unfinished[0] if unfinished else None


def _metric_records(metrics_path: Path) -> Iterator[tuple[int, int, dict[str, float]]]:
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrackingError(
                    f"invalid JSON in {metrics_path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, Mapping):
                continue
            step_value = record.get("step", record.get("global_step", 0))
            try:
                step = int(step_value)
            except (TypeError, ValueError):
                step = 0
            timestamp_ms = int(time.time() * 1000)
            timestamp = record.get("timestamp", record.get("time"))
            if isinstance(timestamp, (int, float)):
                timestamp_ms = int(timestamp * 1000) if timestamp < 10_000_000_000 else int(timestamp)
            elif isinstance(timestamp, str):
                try:
                    timestamp_ms = int(datetime.fromisoformat(timestamp).timestamp() * 1000)
                except ValueError:
                    pass
            values = record.get("metrics", record)
            if not isinstance(values, Mapping):
                continue
            event = str(record.get("event", "")).strip()
            metrics: dict[str, float] = {}
            for key, value in values.items():
                if key in {"step", "global_step", "timestamp", "time", "event"} or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metric_name = f"{event}.{key}" if event else str(key)
                    metrics[metric_name] = float(value)
            if metrics:
                yield step, timestamp_ms, metrics


_MLFLOW_MAX_METRICS_PER_BATCH = 1000


def _metric_batches(mlflow: Any, metrics_path: Path) -> Iterator[list[Any]]:
    entities = getattr(mlflow, "entities", None)
    metric_type = getattr(entities, "Metric", None)
    if metric_type is None:
        try:
            metric_type = importlib.import_module("mlflow.entities").Metric
        except (AttributeError, ModuleNotFoundError) as exc:
            raise TrackingError("MLflow does not expose the Metric entity type") from exc
    batch: list[Any] = []
    for step, timestamp_ms, metrics in _metric_records(metrics_path):
        for key, value in metrics.items():
            batch.append(
                metric_type(
                    key=key[:250],
                    value=value,
                    timestamp=timestamp_ms,
                    step=step,
                )
            )
            if len(batch) == _MLFLOW_MAX_METRICS_PER_BATCH:
                yield batch
                batch = []
    if batch:
        yield batch


def _metadata_artifacts(root: Path) -> list[Path]:
    names = (
        "manifest.json",
        "resolved_config.yml",
        "resolved_config.yaml",
        "argv.json",
        "command.txt",
        "command.ps1",
        "command.sh",
        "environment.json",
        "environment.txt",
        "dataset.json",
        "source.sha256",
        "source.tar.gz",
    )
    return [root / name for name in names if (root / name).is_file()]


def publish_run(
    run_dir: str | os.PathLike[str],
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "ddim",
    experiment: str | None = None,
    run_name: str | None = None,
    include_samples: bool = True,
    include_checkpoints: bool = False,
    mlflow_module: Any | None = None,
) -> PublishResult:
    """Idempotently replay a portable run bundle into an MLflow catalog."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise TrackingError(f"run directory does not exist: {root}")
    manifest = _read_json(root / "manifest.json", required=True)
    state = _read_json(root / "state.json", required=True)
    status = str(state.get("status") or state.get("state") or "").casefold()
    terminal = {"completed", "failed", "cancelled", "timed_out", "interrupted", "lost"}
    if status not in terminal:
        raise TrackingError(
            f"run state is {status or 'unknown'}; publish only a terminal, stable run bundle"
        )
    bundle_id, fingerprint = _bundle_identity(root, manifest)
    mlflow = _mlflow_module(mlflow_module)
    client, effective_uri = _tracking_client(mlflow, tracking_uri)
    if experiment is not None:
        if experiment_name != "ddim" and experiment_name != experiment:
            raise ValueError("experiment and experiment_name disagree")
        experiment_name = experiment
    experiment_id = _experiment_id(client, experiment_name)
    marker_path = root / "tracker.json"
    marker = _read_json(marker_path)
    if (
        marker.get("status") == "complete"
        and marker.get("fingerprint") == fingerprint
        and str(marker.get("tracking_uri")) == effective_uri
        and str(marker.get("experiment_id")) == experiment_id
        and marker.get("run_id")
    ):
        return PublishResult(
            run_id=str(marker["run_id"]),
            experiment_id=experiment_id,
            bundle_id=bundle_id,
            fingerprint=fingerprint,
            tracking_uri=effective_uri,
            reused=True,
            params_logged=int(marker.get("params_logged", 0)),
            metrics_logged=int(marker.get("metrics_logged", 0)),
            artifacts_logged=int(marker.get("artifacts_logged", 0)),
        )
    published, unfinished = _find_published_runs(
        client, experiment_id, fingerprint, bundle_id
    )
    if published is not None:
        run_id = str(published.info.run_id)
        result = PublishResult(
            run_id=run_id,
            experiment_id=experiment_id,
            bundle_id=bundle_id,
            fingerprint=fingerprint,
            tracking_uri=effective_uri,
            reused=True,
        )
        _atomic_json(marker_path, {**result.to_dict(), "status": "complete"})
        return result

    resumable = (
        marker.get("status") == "incomplete"
        and marker.get("fingerprint") == fingerprint
        and str(marker.get("tracking_uri")) == effective_uri
        and str(marker.get("experiment_id")) == experiment_id
        and marker.get("run_id")
    )
    reused_run = False
    if resumable:
        run_id = str(marker["run_id"])
        reused_run = True
        if unfinished is not None and str(unfinished.info.run_id) != run_id:
            raise TrackingError(
                "the local publication marker and MLflow identify different incomplete "
                "runs for the same bundle fingerprint"
            )
    elif unfinished is not None:
        run_id = str(unfinished.info.run_id)
        reused_run = True
    else:
        training = manifest.get("training") if isinstance(manifest.get("training"), Mapping) else {}
        label = training.get("label") or manifest.get("label") or bundle_id
        tags = {
            "mlflow.runName": run_name or str(label),
            "ddim.bundle_id": bundle_id,
            "ddim.bundle_fingerprint": fingerprint,
            "ddim.bundle_path": str(root),
            "ddim.publication_complete": "false",
        }
        command = (
            manifest.get("canonical_argv")
            or manifest.get("command")
            or manifest.get("canonical_command")
        )
        if command:
            tags["ddim.command"] = (
                json.dumps(command, ensure_ascii=False) if isinstance(command, list) else str(command)
            )[:5000]
        created = client.create_run(experiment_id, tags=tags)
        run_id = str(created.info.run_id)
    initial_marker = {
        "status": "incomplete",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "bundle_id": bundle_id,
        "fingerprint": fingerprint,
        "tracking_uri": effective_uri,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(marker_path, initial_marker)

    params_logged = 0
    metrics_logged = 0
    artifacts_logged = 0
    try:
        config = _resolved_config(root, manifest)
        for key, value in _flatten_params(config).items():
            client.log_param(run_id, key[:250], value[:6000])
            params_logged += 1
        for metric_batch in _metric_batches(mlflow, root / "metrics.jsonl"):
            client.log_batch(run_id, metrics=metric_batch, synchronous=True)
            metrics_logged += len(metric_batch)
        for path in _metadata_artifacts(root):
            client.log_artifact(run_id, str(path), artifact_path="bundle")
            artifacts_logged += 1
        samples_path = root / "samples"
        if include_samples and samples_path.is_dir():
            client.log_artifacts(run_id, str(samples_path), artifact_path="samples")
            artifacts_logged += 1
        if include_checkpoints:
            checkpoints_path = root / "checkpoints"
            if checkpoints_path.is_dir():
                client.log_artifacts(run_id, str(checkpoints_path), artifact_path="checkpoints")
                artifacts_logged += 1
        client.set_tag(run_id, "ddim.publication_complete", "true")
        # MLflow may print a run URL containing Unicode symbols after termination.
        # Keep that optional console output from breaking publication on legacy
        # Windows code pages such as cp949.
        with contextlib.redirect_stdout(io.StringIO()):
            client.set_terminated(run_id, status="FINISHED")
    except Exception as exc:
        failed = {
            **initial_marker,
            "status": "incomplete",
            "error": f"{type(exc).__name__}: {exc}",
            "params_logged": params_logged,
            "metrics_logged": metrics_logged,
            "artifacts_logged": artifacts_logged,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(marker_path, failed)
        try:
            client.set_tag(run_id, "ddim.publication_error", str(exc)[:5000])
        except Exception:
            pass
        raise TrackingError(f"MLflow publication failed for {bundle_id}: {exc}") from exc
    result = PublishResult(
        run_id=run_id,
        experiment_id=experiment_id,
        bundle_id=bundle_id,
        fingerprint=fingerprint,
        tracking_uri=effective_uri,
        reused=reused_run,
        params_logged=params_logged,
        metrics_logged=metrics_logged,
        artifacts_logged=artifacts_logged,
    )
    _atomic_json(
        marker_path,
        {
            **result.to_dict(),
            "status": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


def _sqlite_uri(path: Path) -> str:
    normalized = path.resolve().as_posix()
    return f"sqlite:///{normalized}"


def build_mlflow_server_argv(
    root: str | os.PathLike[str],
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    python_executable: str | os.PathLike[str] = sys.executable,
    allow_remote: bool = False,
) -> tuple[str, ...]:
    if not (1 <= int(port) <= 65535):
        raise ValueError("port must be between 1 and 65535")
    loopback_names = {"127.0.0.1", "::1", "localhost"}
    if host not in loopback_names and not allow_remote:
        raise ValueError("MLflow must bind to loopback unless allow_remote=True is explicit")
    target = Path(root).expanduser().resolve()
    database = target / "mlflow.db"
    artifacts = target / "artifacts"
    return (
        str(Path(python_executable).expanduser().resolve()),
        "-m",
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        _sqlite_uri(database),
        "--artifacts-destination",
        artifacts.as_uri(),
    )


def serve_mlflow(
    root: str | os.PathLike[str] | None = None,
    *,
    data_dir: str | os.PathLike[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    python_executable: str | os.PathLike[str] = sys.executable,
    allow_remote: bool = False,
    wait: bool = False,
    popen_factory: Any = subprocess.Popen,
) -> TrackingServerProcess:
    """Start a loopback MLflow server without involving a command interpreter."""

    if importlib.util.find_spec("mlflow") is None:
        raise TrackingError(
            "MLflow is not installed; install this project with the 'tracking' extra first"
        )
    if root is None and data_dir is None:
        raise ValueError("root or data_dir is required")
    if root is not None and data_dir is not None:
        raise ValueError("pass root or data_dir, not both")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as error:
        raise TrackingError(f"MLflow port {host}:{port} is unavailable: {error}") from error
    target = Path(root if root is not None else data_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "artifacts").mkdir(parents=True, exist_ok=True)
    argv = build_mlflow_server_argv(
        target,
        host=host,
        port=port,
        python_executable=python_executable,
        allow_remote=allow_remote,
    )
    stdout_path = target / "mlflow-server.stdout.log"
    stderr_path = target / "mlflow-server.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = popen_factory(
            argv,
            cwd=str(target),
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        returncode = process.wait() if wait else None
    display_host = f"[{host}]" if ":" in host else host
    return TrackingServerProcess(
        argv=argv,
        url=f"http://{display_host}:{port}",
        pid=int(process.pid),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=returncode,
    )


def _deep_value(value: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    for candidate in candidates:
        if candidate in value:
            return value[candidate]
    for item in value.values():
        if isinstance(item, Mapping):
            found = _deep_value(item, candidates)
            if found is not None:
                return found
    return None


def _last_metric_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    latest: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                values = item.get("metrics", item)
                if isinstance(values, Mapping):
                    event = str(item.get("event", ""))
                    for key, value in values.items():
                        if key in {"event", "step", "global_step", "time", "timestamp"}:
                            continue
                        if event == "train" and key == "loss":
                            latest["train_loss"] = value
                        elif event == "validation" and key == "loss":
                            latest["validation_loss"] = value
                        else:
                            latest[key] = value
                if "step" in item:
                    latest["step"] = item["step"]
                if "global_step" in item:
                    latest["global_step"] = item["global_step"]
    return latest


def summarize_run(run_dir: str | os.PathLike[str]) -> RunSummary:
    root = Path(run_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    state = _read_json(root / "state.json")
    backend = _read_json(root / "backend.json")
    if not backend:
        attempts = root / "attempts"
        candidates = sorted(attempts.glob("*/backend.json")) if attempts.is_dir() else []
        if candidates:
            backend = _read_json(candidates[-1])
    metrics = _last_metric_record(root / "metrics.jsonl")
    config = _resolved_config(root, manifest)

    def integer(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def number(value: Any) -> float | None:
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    global_step = integer(
        state.get("global_step", state.get("step", metrics.get("global_step", metrics.get("step"))))
    )
    max_steps = integer(_deep_value(config, ("max_steps", "n_iters")))
    run_id = str(manifest.get("run_id") or manifest.get("id") or root.name)
    training = manifest.get("training") if isinstance(manifest.get("training"), Mapping) else {}
    return RunSummary(
        run_dir=root,
        run_id=run_id,
        label=str(training.get("label") or manifest.get("label") or manifest.get("name") or run_id),
        state=str(state.get("status") or state.get("state") or "prepared"),
        global_step=global_step,
        max_steps=max_steps,
        latest_loss=number(metrics.get("loss", metrics.get("train_loss"))),
        latest_validation_loss=number(
            metrics.get("validation_loss", metrics.get("val_loss"))
        ),
        backend=str(backend.get("backend") or backend.get("executor") or backend.get("name")) if backend else None,
        started_at=state.get("started_at") or manifest.get("started_at") or manifest.get("created_at"),
        updated_at=state.get("updated_at") or state.get("finished_at"),
        detail=state.get("detail") or state.get("error"),
        extra={"attempt": state.get("attempt"), "job_id": backend.get("job_id")},
    )


def format_run_summary(summary: RunSummary) -> str:
    progress = ""
    if summary.global_step is not None:
        progress = f"step {summary.global_step:,}"
        if summary.max_steps:
            progress += f"/{summary.max_steps:,} ({summary.progress * 100:.1f}%)"
    fields = [
        f"Run: {summary.label} ({summary.run_id})",
        f"State: {summary.state}" + (f" | {progress}" if progress else ""),
    ]
    if summary.latest_loss is not None:
        loss = f"train={summary.latest_loss:.6g}"
        if summary.latest_validation_loss is not None:
            loss += f", validation={summary.latest_validation_loss:.6g}"
        fields.append(f"Loss: {loss}")
    if summary.backend:
        fields.append(
            f"Backend: {summary.backend}"
            + (f" | job {summary.extra['job_id']}" if summary.extra.get("job_id") else "")
        )
    if summary.detail:
        fields.append(f"Detail: {summary.detail}")
    fields.append(f"Directory: {summary.run_dir}")
    return "\n".join(fields)


def _tail_file(path: Path, lines: int) -> list[str]:
    if lines <= 0:
        raise ValueError("lines must be positive")
    if not path.is_file():
        return []
    block_size = 8192
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        newline_count = 0
        while position > 0 and newline_count <= lines:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)
            chunks.append(block)
            newline_count += block.count(b"\n")
    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


def latest_log_paths(run_dir: str | os.PathLike[str]) -> dict[str, Path]:
    root = Path(run_dir).expanduser().resolve()
    attempts = root / "attempts"
    attempt_dirs = sorted(path for path in attempts.glob("*") if path.is_dir()) if attempts.is_dir() else []
    log_root = attempt_dirs[-1] if attempt_dirs else root
    return {"stdout": log_root / "stdout.log", "stderr": log_root / "stderr.log"}


def tail_logs(
    run_dir: str | os.PathLike[str], *, lines: int = 50, stream: str = "both"
) -> str:
    normalized = stream.casefold()
    if normalized not in {"stdout", "stderr", "both"}:
        raise ValueError("stream must be stdout, stderr, or both")
    paths = latest_log_paths(run_dir)
    selected = ("stdout", "stderr") if normalized == "both" else (normalized,)
    output: list[str] = []
    for name in selected:
        content = _tail_file(paths[name], lines)
        if normalized == "both":
            output.append(f"== {name}: {paths[name]} ==")
        output.extend(content or ["(no log output)"])
    return "\n".join(output)


def follow_log(
    path: str | os.PathLike[str], *, poll_interval: float = 0.5, start_at_end: bool = False
) -> Iterable[str]:
    """Yield appended log lines; callers decide when to stop iteration."""

    target = Path(path)
    position = target.stat().st_size if start_at_end and target.exists() else 0
    while True:
        if target.exists():
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(position)
                for line in handle:
                    yield line.rstrip("\r\n")
                position = handle.tell()
        time.sleep(poll_interval)


__all__ = [
    "PublishResult",
    "RunSummary",
    "TrackingError",
    "TrackingServerProcess",
    "build_mlflow_server_argv",
    "follow_log",
    "format_run_summary",
    "latest_log_paths",
    "publish_run",
    "serve_mlflow",
    "summarize_run",
    "tail_logs",
]
