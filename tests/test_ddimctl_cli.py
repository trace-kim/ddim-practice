from pathlib import Path

import pytest
import typer

from ddimctl.cli import (
    DEFAULT_CONFIG,
    _canonical_argv,
    _load_plan,
    _reject_duplicate_scalar_options,
)
from ddimctl.schemas import TrainingSpec


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
