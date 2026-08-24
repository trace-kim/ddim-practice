from pathlib import Path

import pytest

from ddimctl.profiles import load_profile, profile_path, save_profile, user_config_dir
from ddimctl.schemas import ExecutorType, MachineProfile, SlurmResources


def test_standard_platform_configuration_paths() -> None:
    assert user_config_dir(
        environ={"APPDATA": "C:/Users/alice/AppData/Roaming"},
        platform="win32",
        home=Path("C:/Users/alice"),
    ) == Path("C:/Users/alice/AppData/Roaming/ddimctl")
    assert user_config_dir(
        environ={"XDG_CONFIG_HOME": "/configuration"},
        platform="linux",
        home=Path("/home/alice"),
    ) == Path("/configuration/ddimctl")


def test_profile_round_trip_is_atomic_and_strict(tmp_path: Path) -> None:
    profile = MachineProfile(
        machine_id="local-test",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": tmp_path / "data"},
        python_executable="python",
        expected_gpu="RTX A6000",
    )
    destination = save_profile(profile, tmp_path / "config")
    assert destination == profile_path("local-test", tmp_path / "config")
    assert load_profile("local-test", tmp_path / "config") == profile
    assert load_profile("local-test", tmp_path / "config").expected_gpu == "RTX A6000"
    assert not list(destination.parent.glob("*.tmp"))

    with pytest.raises(FileExistsError):
        save_profile(profile, tmp_path / "config", overwrite=False)


def test_profile_keeps_training_fields_out_and_validates_executor() -> None:
    with pytest.raises(ValueError, match="expected_gpu"):
        MachineProfile(
            machine_id="bad-gpu",
            executor=ExecutorType.FOREGROUND,
            runs_root=Path("/runs"),
            datasets={"sem": Path("/data")},
            expected_gpu="H100\nwrong",
        )
    with pytest.raises(ValueError, match="credentials"):
        MachineProfile(
            machine_id="bad-tracker",
            executor=ExecutorType.FOREGROUND,
            runs_root=Path("/runs"),
            datasets={"sem": Path("/data")},
            mlflow_tracking_uri="https://user:secret@mlflow.internal",
        )
    with pytest.raises(ValueError):
        MachineProfile(
            machine_id="hpc",
            executor=ExecutorType.SLURM,
            runs_root=Path("/runs"),
            datasets={"sem": Path("/data")},
        )
    with pytest.raises(ValueError):
        MachineProfile.model_validate(
            {
                "machine_id": "bad",
                "executor": ExecutorType.FOREGROUND,
                "runs_root": Path("/runs"),
                "datasets": {"sem": Path("/data")},
                "learning_rate": 0.1,
            }
        )

    valid = MachineProfile(
        machine_id="hpc",
        executor=ExecutorType.SLURM,
        runs_root=Path("/runs"),
        datasets={"sem": Path("/data")},
        slurm=SlurmResources(),
    )
    assert valid.slurm is not None
