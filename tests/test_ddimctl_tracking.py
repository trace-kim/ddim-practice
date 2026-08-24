from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ddimctl.tracking import (
    TrackingError,
    build_mlflow_server_argv,
    format_run_summary,
    publish_run,
    summarize_run,
    tail_logs,
)


class FakeMlflowClient:
    def __init__(self):
        self.experiment = None
        self.runs = []
        self.params = []
        self.metrics = []
        self.metric_batches = []
        self.artifacts = []
        self.tags = []
        self.terminated = []
        self.fail_next_metric_batch = False
        self.termination_message = None

    def get_experiment_by_name(self, name):
        return self.experiment

    def create_experiment(self, name):
        self.experiment = SimpleNamespace(experiment_id="exp-1", name=name)
        return "exp-1"

    def search_runs(self, **_kwargs):
        return self.runs

    def create_run(self, experiment_id, tags):
        run = SimpleNamespace(
            info=SimpleNamespace(run_id=f"run-{len(self.runs) + 1}"),
            data=SimpleNamespace(tags=dict(tags)),
        )
        self.runs.append(run)
        return run

    def log_param(self, run_id, key, value):
        self.params.append((run_id, key, value))

    def log_batch(self, run_id, metrics=(), params=(), tags=(), synchronous=None):
        assert not params
        assert not tags
        assert synchronous is True
        assert len(metrics) <= 1000
        if self.fail_next_metric_batch:
            self.fail_next_metric_batch = False
            raise RuntimeError("simulated metric batch failure")
        points = [
            (run_id, metric.key, metric.value, metric.timestamp, metric.step)
            for metric in metrics
        ]
        self.metric_batches.append(points)
        self.metrics.extend(points)

    def log_artifact(self, run_id, path, artifact_path):
        self.artifacts.append((run_id, Path(path).name, artifact_path))

    def log_artifacts(self, run_id, path, artifact_path):
        self.artifacts.append((run_id, Path(path).name, artifact_path))

    def set_tag(self, run_id, key, value):
        self.tags.append((run_id, key, value))
        for run in self.runs:
            if run.info.run_id == run_id:
                run.data.tags[key] = value

    def set_terminated(self, run_id, status):
        if self.termination_message is not None:
            print(self.termination_message)
        self.terminated.append((run_id, status))


class FakeMlflow:
    def __init__(self):
        self.uri = "file:///default"
        self.client = FakeMlflowClient()
        self.tracking = SimpleNamespace(MlflowClient=lambda **_kwargs: self.client)
        self.entities = SimpleNamespace(
            Metric=lambda key, value, timestamp, step: SimpleNamespace(
                key=key,
                value=value,
                timestamp=timestamp,
                step=step,
            )
        )

    def set_tracking_uri(self, uri):
        self.uri = uri

    def get_tracking_uri(self):
        return self.uri


def make_bundle(tmp_path: Path) -> Path:
    run = tmp_path / "20260823__sem__abc123"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "training": {"label": "SEM baseline", "max_steps": 10, "lr": 0.0002},
                "canonical_argv": [
                    "ddimctl", "train", "launch", "--max-steps", "10"
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "metrics": {"loss": 2.5, "note": "skip"}}),
                json.dumps({"step": 2, "metrics": {"loss": 1.5, "validation_loss": 1.8}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "state.json").write_text(
        json.dumps({"status": "completed", "global_step": 10}), encoding="utf-8"
    )
    samples = run / "samples"
    samples.mkdir()
    (samples / "step_2.txt").write_text("sample", encoding="utf-8")
    return run


def test_mlflow_publish_is_explicit_optional_and_idempotent(tmp_path):
    run = make_bundle(tmp_path)
    mlflow = FakeMlflow()

    first = publish_run(
        run,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="sem",
        mlflow_module=mlflow,
    )
    call_counts = (
        len(mlflow.client.params),
        len(mlflow.client.metrics),
        len(mlflow.client.artifacts),
    )
    second = publish_run(
        run,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="sem",
        mlflow_module=mlflow,
    )

    assert first.reused is False
    assert second.reused is True
    assert first.run_id == second.run_id
    assert len(mlflow.client.runs) == 1
    assert call_counts == (
        len(mlflow.client.params),
        len(mlflow.client.metrics),
        len(mlflow.client.artifacts),
    )
    assert (run / "tracker.json").is_file()
    assert {entry[1] for entry in mlflow.client.metrics} == {"loss", "validation_loss"}
    assert mlflow.client.runs[0].data.tags["mlflow.runName"] == "SEM baseline"
    assert '"--max-steps"' in mlflow.client.runs[0].data.tags["ddim.command"]

    with (run / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 3, "metrics": {"loss": 1.0}}) + "\n")
    changed = publish_run(
        run,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="sem",
        mlflow_module=mlflow,
    )
    assert changed.reused is False
    assert len(mlflow.client.runs) == 2


def test_mlflow_metrics_are_logged_in_bounded_batches_with_exact_points(tmp_path):
    run = make_bundle(tmp_path)
    long_key = "metric." + ("x" * 300)
    metrics = {f"metric_{index}": index + 0.25 for index in range(1000)}
    metrics[long_key] = 1000.25
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 7, "timestamp": 1234, "metrics": metrics}) + "\n",
        encoding="utf-8",
    )
    mlflow = FakeMlflow()

    result = publish_run(run, mlflow_module=mlflow)

    assert result.metrics_logged == 1001
    assert [len(batch) for batch in mlflow.client.metric_batches] == [1000, 1]
    assert mlflow.client.metrics[0] == ("run-1", "metric_0", 0.25, 1234000, 7)
    assert mlflow.client.metrics[-1] == (
        "run-1",
        long_key[:250],
        1000.25,
        1234000,
        7,
    )


def test_mlflow_retry_discovers_and_reuses_unfinished_run_without_marker(tmp_path):
    run = make_bundle(tmp_path)
    mlflow = FakeMlflow()
    mlflow.client.fail_next_metric_batch = True

    with pytest.raises(TrackingError, match="simulated metric batch failure"):
        publish_run(run, mlflow_module=mlflow)
    first_run_id = mlflow.client.runs[0].info.run_id
    (run / "tracker.json").unlink()

    result = publish_run(run, mlflow_module=mlflow)

    assert result.run_id == first_run_id
    assert result.reused is True
    assert len(mlflow.client.runs) == 1
    assert result.metrics_logged == 3
    assert mlflow.client.runs[0].data.tags["ddim.publication_complete"] == "true"


def test_mlflow_retry_refuses_ambiguous_unfinished_runs(tmp_path):
    run = make_bundle(tmp_path)
    mlflow = FakeMlflow()
    mlflow.client.fail_next_metric_batch = True

    with pytest.raises(TrackingError):
        publish_run(run, mlflow_module=mlflow)
    (run / "tracker.json").unlink()
    tags = dict(mlflow.client.runs[0].data.tags)
    mlflow.client.create_run("exp-1", tags=tags)

    with pytest.raises(TrackingError, match="multiple incomplete MLflow runs"):
        publish_run(run, mlflow_module=mlflow)
    assert len(mlflow.client.runs) == 2


def test_mlflow_termination_output_cannot_break_publication_on_cp949(tmp_path):
    class Cp949Stdout:
        encoding = "cp949"

        def write(self, value):
            value.encode(self.encoding)
            return len(value)

        def flush(self):
            return None

    run = make_bundle(tmp_path)
    mlflow = FakeMlflow()
    mlflow.client.termination_message = "\U0001f3f3 View run"

    with contextlib.redirect_stdout(Cp949Stdout()):
        result = publish_run(run, mlflow_module=mlflow)

    assert result.reused is False
    assert mlflow.client.terminated == [(result.run_id, "FINISHED")]
    assert json.loads((run / "tracker.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_mlflow_publish_refuses_a_mutating_active_bundle(tmp_path):
    run = make_bundle(tmp_path)
    (run / "state.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(TrackingError, match="terminal"):
        publish_run(run, mlflow_module=FakeMlflow())


def test_mlflow_tracking_uri_never_persists_embedded_credentials(tmp_path):
    run = make_bundle(tmp_path)
    with pytest.raises(TrackingError, match="credentials"):
        publish_run(
            run,
            tracking_uri="https://user:secret@mlflow.internal",
            mlflow_module=FakeMlflow(),
        )


def test_mlflow_server_defaults_to_loopback_and_rejects_accidental_exposure(tmp_path):
    argv = build_mlflow_server_argv(tmp_path, python_executable=sys.executable)

    assert argv[:4] == (str(Path(sys.executable).resolve()), "-m", "mlflow", "server")
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "sqlite:///" in argv[argv.index("--backend-store-uri") + 1]
    assert argv[argv.index("--artifacts-destination") + 1] == (
        tmp_path / "artifacts"
    ).resolve().as_uri()
    with pytest.raises(ValueError, match="loopback"):
        build_mlflow_server_argv(tmp_path, host="0.0.0.0")


def test_terminal_summary_and_latest_attempt_logs(tmp_path):
    run = make_bundle(tmp_path)
    (run / "state.json").write_text(
        json.dumps({"state": "running", "global_step": 2, "attempt": 2}),
        encoding="utf-8",
    )
    attempt = run / "attempts" / "002"
    attempt.mkdir(parents=True)
    (attempt / "backend.json").write_text(
        json.dumps({"backend": "slurm", "job_id": "123"}), encoding="utf-8"
    )
    (attempt / "stdout.log").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (attempt / "stderr.log").write_text("warning\n", encoding="utf-8")

    summary = summarize_run(run)
    rendered = format_run_summary(summary)
    logs = tail_logs(run, lines=2)

    assert summary.state == "running"
    assert summary.global_step == 2
    assert summary.max_steps == 10
    assert summary.backend == "slurm"
    assert summary.latest_loss == 1.5
    assert "20.0%" in rendered
    assert "two\nthree" in logs
    assert "warning" in logs
