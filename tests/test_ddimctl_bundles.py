from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ddimctl.bundles import (
    _worker_bootstrap_source,
    atomic_write_json,
    create_run_bundle,
    create_source_snapshot,
    ConfigurationError,
    fingerprint_dataset,
    load_attempt_state,
    load_manifest,
    render_posix_command,
    render_powershell_command,
)
from ddimctl.schemas import ExecutorType, MachineProfile, RunStatus, TrainingSpec


def test_command_rendering_preserves_each_argument() -> None:
    argv = ["python", "train.py", "--label", "a name", "--path", "it's/here"]
    assert render_posix_command(argv) == "python train.py --label 'a name' --path 'it'\"'\"'s/here'"
    assert render_powershell_command(argv) == (
        "& 'python' 'train.py' '--label' 'a name' '--path' 'it''s/here'"
    )
    with pytest.raises(ValueError):
        render_posix_command(["python", "unsafe\nargument"])


def test_dataset_fingerprint_is_stable_and_detects_content_changes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "b.png").write_bytes(b"second")
    (dataset / "a.png").write_bytes(b"first")
    (dataset / "ignore.txt").write_text("not an image", encoding="utf-8")

    first = fingerprint_dataset(dataset, [".png"])
    second = fingerprint_dataset(dataset, [".png"])
    assert first.sha256 == second.sha256
    assert first.file_count == 2
    assert first.total_bytes == 11

    (dataset / "a.png").write_bytes(b"FIRST")
    changed = fingerprint_dataset(dataset, [".png"])
    assert changed.sha256 != first.sha256


def test_source_snapshot_is_deterministic_and_excludes_runtime_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("answer = 42\n", encoding="utf-8")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "secret.py").write_text("exclude = True\n", encoding="utf-8")
    one = create_source_snapshot(repo, tmp_path / "one.tar.gz")
    two = create_source_snapshot(repo, tmp_path / "two.tar.gz")
    assert one.sha256 == two.sha256
    assert one.file_count == 1
    with tarfile.open(tmp_path / "one.tar.gz", "r:gz") as archive:
        assert archive.getnames() == ["module.py"]


def test_run_bundle_is_complete_collision_safe_and_round_trips(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('training')\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "image.png").write_bytes(b"fake-image")
    profile = MachineProfile(
        machine_id="test-machine",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": dataset},
        timezone="Asia/Seoul",
        python_executable="python",
    )
    spec = TrainingSpec(
        label="smoke-test",
        max_steps=10,
        checkpoint_every=5,
        validation_every=5,
        sample_every=5,
    )
    created = datetime(2026, 8, 23, 14, 30, 15, tzinfo=ZoneInfo("Asia/Seoul"))
    argv = ("python", "-m", "ddimctl", "train", "launch", "--label", "smoke-test")

    run_root, manifest = create_run_bundle(repo, profile, spec, argv, now=created)
    assert run_root.parent.name == "2026-08-23"
    assert run_root.name.startswith("20260823T143015+0900__smoke-test__")
    expected = {
        "manifest.json",
        "resolved_config.yml",
        "argv.json",
        "command.ps1",
        "command.sh",
        "source.tar.gz",
        "environment.json",
        "dataset.json",
        "state.json",
        "metrics.jsonl",
        "tensorboard",
        "samples",
        "checkpoints",
        "attempts",
    }
    assert expected <= {path.name for path in run_root.iterdir()}
    assert load_manifest(run_root) == manifest
    assert load_attempt_state(run_root).status is RunStatus.PREPARED
    assert json.loads((run_root / "argv.json").read_text(encoding="utf-8")) == list(argv)
    assert (run_root / "attempts" / "001" / "stdout.log").is_file()
    assert (run_root / "source" / "train.py").is_file()
    environment = json.loads((run_root / "environment.json").read_text(encoding="utf-8"))
    assert environment["packages"]
    assert "version" in environment["torch"]
    assert isinstance(environment["torch"]["cuda_available"], bool)
    bootstrap = (run_root / "worker-bootstrap.py").read_text(encoding="utf-8")
    compile(bootstrap, "worker-bootstrap.py", "exec")
    assert "redirect_stdout(stdout_log)" in bootstrap
    assert "redirect_stderr(stderr_log)" in bootstrap

    with pytest.raises(FileExistsError):
        create_run_bundle(repo, profile, spec, argv, now=created)
    assert not list(run_root.parent.glob("*.preparing-*"))

    with (run_root / "source.tar.gz").open("ab") as handle:
        handle.write(b"tampered")
    from ddimctl.bundles import materialize_source_snapshot

    with pytest.raises(ConfigurationError, match="checksum mismatch"):
        materialize_source_snapshot(run_root)


@pytest.mark.parametrize(
    ("gpu_index", "inherited_visibility", "expected_visibility"),
    (
        (2, None, "2"),
        (1, "3, 5,7", "5"),
        (0, "GPU-allocated-by-scheduler", "GPU-allocated-by-scheduler"),
    ),
)
def test_worker_bootstrap_isolates_selected_gpu_before_importing_worker(
    tmp_path: Path,
    gpu_index: int,
    inherited_visibility: str | None,
    expected_visibility: str,
) -> None:
    run_root = tmp_path / f"run-{gpu_index}"
    package = run_root / "source" / "ddimctl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text(
        "import os\nprint('worker-visible=' + repr(os.environ.get('CUDA_VISIBLE_DEVICES')))\n",
        encoding="utf-8",
    )
    (run_root / "attempts" / "001").mkdir(parents=True)
    atomic_write_json(run_root / "manifest.json", {"machine": {"gpu_index": gpu_index}})
    bootstrap = run_root / "worker-bootstrap.py"
    bootstrap.write_text(_worker_bootstrap_source(), encoding="utf-8")

    environment = os.environ.copy()
    if inherited_visibility is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = inherited_visibility
    result = subprocess.run(
        [sys.executable, str(bootstrap), "--run", str(run_root), "--attempt", "1"],
        cwd=run_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    worker_log = (run_root / "attempts" / "001" / "stdout.log").read_text(encoding="utf-8")
    assert f"worker-visible={expected_visibility!r}" in worker_log
    assert "worker GPU isolation" in worker_log


def test_materialization_rejects_parent_traversal_before_extraction(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("pass\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "image.png").write_bytes(b"image")
    profile = MachineProfile(
        machine_id="safe-machine",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": dataset},
    )
    spec = TrainingSpec(max_steps=1, checkpoint_every=1, validation_every=1, sample_every=1)
    run_root, _ = create_run_bundle(
        repo,
        profile,
        spec,
        ("python", "worker-bootstrap.py", "--run", "placeholder", "--attempt", "1"),
    )

    unsafe = tmp_path / "unsafe-run"
    unsafe.mkdir()
    payload = b"escaped = True\n"
    archive = unsafe / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../escaped.py")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["source"].update(
        {"archive": "source.tar.gz", "sha256": digest, "file_count": 1, "total_bytes": len(payload)}
    )
    atomic_write_json(unsafe / "manifest.json", manifest)

    from ddimctl.bundles import materialize_source_snapshot

    with pytest.raises(ConfigurationError, match="unsafe source archive member"):
        materialize_source_snapshot(unsafe)
    assert not (tmp_path / "escaped.py").exists()


def test_atomic_json_replaces_without_leaving_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"value": 1})
    atomic_write_json(target, {"value": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(tmp_path.glob("*.tmp"))
