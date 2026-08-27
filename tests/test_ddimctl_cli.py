import sys
from pathlib import Path

import pytest
import typer

from ddimctl.cli import (
    DEFAULT_CONFIG,
    _canonical_argv,
    _launch_preflight,
    _load_plan,
    _reject_duplicate_scalar_options,
)
from ddimctl.schemas import ExecutorType, MachineProfile, TrainingSpec


def test_cli_rejects_duplicate_and_conflicting_boolean_spellings() -> None:
    with pytest.raises(typer.BadParameter, match="duplicate"):
        _reject_duplicate_scalar_options(["train", "launch", "--max-steps", "1", "--max-steps", "2"])
    with pytest.raises(typer.BadParameter, match="duplicate"):
        _reject_duplicate_scalar_options(
            ["train", "launch", "--cache-in-memory", "--no-cache-in-memory"]
        )


def test_cli_enforces_the_single_active_experiment_config(tmp_path: Path) -> None:
    alternate = tmp_path / "alternate.yml"
    alternate.write_text("training: {}\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="single active config"):
        _load_plan("not-needed", alternate, {})


def test_canonical_command_exposes_every_varying_setting() -> None:
    spec = TrainingSpec()
    argv = _canonical_argv("machine", DEFAULT_CONFIG, spec)
    for option in (
        "--label",
        "--dataset",
        "--image-size",
        "--model-ch",
        "--ch-mult",
        "--diffusion-steps",
        "--beta-start",
        "--beta-end",
        "--ema-rate",
        "--max-steps",
        "--batch-size",
        "--learning-rate",
        "--checkpoint-every",
        "--checkpoint-minutes",
        "--validation-every",
        "--sample-every",
        "--seed",
        "--reproducibility",
        "--num-workers",
    ):
        assert argv.count(option) == 1
    assert argv[-1] == "--yes"


def test_launch_preflight_accepts_selected_gpu_on_multi_gpu_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = MachineProfile(
        machine_id="four-gpu-host",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": tmp_path / "data"},
        python_executable=sys.executable,
        gpu_index=2,
        expected_gpu="Selected GPU",
    )
    report = {
        "available": True,
        "count": 4,
        "names": ["GPU 0", "GPU 1", "Selected GPU", "GPU 3"],
        "selected_index": 2,
        "selected_name": "Selected GPU",
        "selection_error": None,
        "cuda_visible_devices": None,
    }
    monkeypatch.setattr("ddimctl.cli._probe_configured_gpu", lambda _profile: report)

    _launch_preflight(profile)


def test_launch_preflight_checks_expected_name_on_selected_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = MachineProfile(
        machine_id="wrong-selected-gpu",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": tmp_path / "data"},
        python_executable=sys.executable,
        gpu_index=1,
        expected_gpu="H100",
    )
    report = {
        "available": True,
        "count": 4,
        "names": ["H100", "RTX A6000", "H100", "H100"],
        "selected_index": 1,
        "selected_name": "RTX A6000",
        "selection_error": None,
        "cuda_visible_devices": None,
    }
    monkeypatch.setattr("ddimctl.cli._probe_configured_gpu", lambda _profile: report)

    with pytest.raises(RuntimeError, match="selected GPU.*does not match"):
        _launch_preflight(profile)


def test_launch_preflight_rejects_out_of_range_gpu_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = MachineProfile(
        machine_id="bad-gpu-index",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": tmp_path / "data"},
        python_executable=sys.executable,
        gpu_index=4,
    )
    report = {
        "available": True,
        "count": 4,
        "names": ["GPU 0", "GPU 1", "GPU 2", "GPU 3"],
        "selected_index": None,
        "selected_name": None,
        "selection_error": "configured gpu_index 4 is out of range for 4 visible CUDA GPU(s)",
        "cuda_visible_devices": None,
    }
    monkeypatch.setattr("ddimctl.cli._probe_configured_gpu", lambda _profile: report)

    with pytest.raises(RuntimeError, match="gpu_index 4 is out of range"):
        _launch_preflight(profile)
