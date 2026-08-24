from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ddimctl.backends import (
    CommandResult,
    ExternalHPCBackend,
    ForegroundBackend,
    LaunchRequest,
    SlurmBackend,
    SlurmResources,
    WindowsTaskBackend,
    detect_backend,
)


class ScriptedRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected command: {argv}")
        return self.responses.pop(0)


def result(stdout="", stderr="", returncode=0):
    return CommandResult(("fake",), returncode, stdout, stderr)


def request(tmp_path: Path, **changes) -> LaunchRequest:
    values = {
        "argv": (sys.executable, "-c", "print('hello world')"),
        "cwd": tmp_path,
        "run_dir": tmp_path / "run with spaces",
        "name": "SEM experiment",
    }
    values.update(changes)
    return LaunchRequest(**values)


def test_launch_request_normalizes_paths_and_rejects_bad_environment(tmp_path):
    prepared = request(tmp_path)
    assert prepared.cwd.is_absolute()
    assert prepared.run_dir.is_absolute()
    assert prepared.name == "SEM-experiment"
    with pytest.raises(ValueError, match="environment variable"):
        request(tmp_path, env={"BAD-NAME": "value"})


def test_foreground_captures_logs_and_preserves_exit_code(tmp_path):
    prepared = request(
        tmp_path,
        argv=(
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(3)",
        ),
    )
    backend = ForegroundBackend()
    submission = backend.submit(prepared)

    assert submission.state == "failed"
    assert submission.metadata["returncode"] == 3
    assert prepared.stdout_path.read_text(encoding="utf-8").strip() == "out"
    assert prepared.stderr_path.read_text(encoding="utf-8").strip() == "err"
    assert backend.status(submission).exit_code == 3


def test_windows_task_xml_is_durable_and_uses_safe_action_fields(tmp_path):
    prepared = request(tmp_path)
    xml = WindowsTaskBackend().render_task_xml(prepared, user_id="DOMAIN\\worker")

    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in xml
    assert f"<WorkingDirectory>{tmp_path}</WorkingDirectory>" in xml
    assert f"<Command>{Path(sys.executable).resolve()}</Command>" in xml
    assert "hello world" in xml


def test_windows_task_submit_and_status_use_argument_vectors(tmp_path):
    runner = ScriptedRunner(
        [
            result(),
            result(),
            result(stdout=json.dumps({"State": "Ready", "LastTaskResult": 0})),
        ]
    )
    backend = WindowsTaskBackend(
        runner=runner,
        schtasks_executable="C:/Windows/System32/schtasks.exe",
        powershell_executable="C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    )
    submission = backend.submit(request(tmp_path))
    status = backend.status(submission)

    assert submission.state == "submitted"
    assert status.state == "completed"
    assert runner.calls[0][0][1:3] == ("/Create", "/TN")
    assert runner.calls[1][0][1] == "/Run"
    assert "-EncodedCommand" in runner.calls[2][0]
    assert (request(tmp_path).stdout_path.parent / "windows-task.xml").is_file()


def test_slurm_script_has_explicit_resources_signal_and_quoted_command(tmp_path):
    prepared = request(
        tmp_path,
        argv=("/opt/venv/bin/python", "worker.py", "--label", "value with spaces"),
        env={"PYTHONUNBUFFERED": "1"},
        expected_gpu="H100",
    )
    resources = SlurmResources(
        partition="gpu",
        account="research",
        gpus=1,
        cpus_per_task=8,
        memory="64G",
        time_limit="04:00:00",
        signal_seconds=180,
        gpu_type="h100",
    )
    script = SlurmBackend.render_script(prepared, resources)

    assert "#SBATCH --gres=gpu:h100:1" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --signal=B:USR1@180" in script
    assert "#SBATCH --no-requeue" in script
    assert "'value with spaces'" in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert "expected GPU not found: H100" in script
    assert "exec srun --unbuffered /opt/venv/bin/python" in script


def test_slurm_submit_parses_cluster_suffix_and_status_uses_accounting(tmp_path):
    runner = ScriptedRunner(
        [
            result(stdout="12345;cluster-a\n"),
            result(stdout=""),
            result(stdout="12345|COMPLETED|0:0\n12345.batch|COMPLETED|0:0\n"),
            result(),
        ]
    )
    backend = SlurmBackend(
        runner=runner,
        sbatch_executable="/usr/bin/sbatch",
        squeue_executable="/usr/bin/squeue",
        sacct_executable="/usr/bin/sacct",
        scancel_executable="/usr/bin/scancel",
    )
    submission = backend.submit(request(tmp_path))
    status = backend.status(submission)
    stopped = backend.stop(submission)

    assert submission.job_id == "12345"
    assert status.state == "completed"
    assert status.exit_code == 0
    assert stopped.state == "interrupted"
    assert runner.calls[0][0][:2] == ("/usr/bin/sbatch", "--parsable")
    assert runner.calls[2][0][0] == "/usr/bin/sacct"
    assert runner.calls[3][0] == ("/usr/bin/scancel", "--signal=TERM", "12345")
    assert Path(submission.metadata["script"]).name == "job.sbatch"


def test_slurm_doctor_can_validate_resources_without_submitting(tmp_path, monkeypatch):
    responses = [
        result(stdout="gpu*|up|1-00:00:00|gpu:h100:4\n"),
        result(stdout="Job 12345 to start at 2026-08-24T00:00:00\n"),
    ]
    runner = ScriptedRunner(responses)
    backend = SlurmBackend(
        runner=runner,
        sbatch_executable="/usr/bin/sbatch",
        squeue_executable="/usr/bin/squeue",
        sacct_executable="/usr/bin/sacct",
        scancel_executable="/usr/bin/scancel",
    )
    monkeypatch.setattr("ddimctl.backends.shutil.which", lambda command: f"/usr/bin/{command}")

    report = backend.probe(
        {"slurm": {"partition": "gpu", "gpus": 1, "memory_gb": 64}},
        exercise=True,
    )

    assert report["available"] is True
    assert report["test_only"]["success"] is True
    assert runner.calls[1][0][1] == "--test-only"


def test_external_hpc_prepares_gpu_guarded_worker_and_probe(tmp_path):
    prepared = request(
        tmp_path,
        expected_gpu="H100",
        resources={"gpus": 1},
        argv=("/opt/venv/bin/python", "-m", "ddimctl.worker", "--manifest", "manifest.json"),
    )
    backend = ExternalHPCBackend()
    submission = backend.submit(prepared)
    worker = (prepared.run_dir / "worker.sh").read_text(encoding="utf-8")

    assert submission.state == "prepared"
    assert "nvidia-smi" in worker
    assert "expected GPU not found: H100" in worker
    assert "exec /opt/venv/bin/python -m ddimctl.worker" in worker
    assert (prepared.run_dir / "probe.sh").is_file()


def test_detect_backend_honors_profile_choice():
    assert detect_backend({"executor": "external_hpc"}) == "external-hpc"
    assert detect_backend({"execution_backend": "local"}) == "foreground"
