from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from burst_diffusion.cli import app

runner = CliRunner()


def _make_sources(root: Path, count: int = 3) -> Path:
    source_dir = root / "sources"
    source_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        array = rng.integers(60, 196, size=(24, 20), dtype=np.uint8)
        Image.fromarray(array).save(source_dir / f"img_{index}.png")
    return source_dir


def _write_config(path: Path, dataset_dir: Path, run_dir: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  dataset_dir: {dataset_dir.as_posix()}",
                "  image_size: 16",
                "  channels: 1",
                "  val_fraction: 0.34",
                "schedule:",
                "  num_steps: 2",
                "model:",
                "  ch: 8",
                "  ch_mult: [1, 2]",
                "  num_res_blocks: 1",
                "  attn_resolutions: []",
                "training:",
                f"  run_dir: {run_dir.as_posix()}",
                "  batch_size: 2",
                "  max_steps: 2",
                "  log_every: 1",
                "  val_every: 2",
                "  val_images: 1",
                "  checkpoint_every: 2",
                "  device: cpu",
                "sampling: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "command",
    ["generate", "preview", "train", "sample", "evaluate", "repeatability", "provenance"],
)
def test_every_subcommand_has_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


def test_generate_preview_train_sample_evaluate_end_to_end(tmp_path: Path) -> None:
    source_dir = _make_sources(tmp_path)
    dataset_dir = tmp_path / "dataset"
    result = runner.invoke(
        app,
        [
            "generate",
            "--source-dir", str(source_dir),
            "--output-dir", str(dataset_dir),
            "--num-sources", "3",
            "--replicas", "3",
            "--noise-type", "poisson",
            "--peak", "30",
            "--max-side", "64",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "single-frame PSNR median" in result.output
    assert (dataset_dir / "stats.json").is_file()

    preview_path = tmp_path / "preview.png"
    result = runner.invoke(
        app,
        [
            "preview",
            "--dataset", str(dataset_dir),
            "--out", str(preview_path),
            "--avg-counts", "1,3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert preview_path.is_file()

    run_dir = tmp_path / "run"
    config_path = _write_config(tmp_path / "config.yml", dataset_dir, run_dir)
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    checkpoint = run_dir / "ckpt_latest.pt"
    assert checkpoint.is_file()

    result = runner.invoke(app, ["train", "--config", str(config_path), "--resume"])
    assert result.exit_code == 0, result.output

    samples_dir = tmp_path / "samples"
    result = runner.invoke(
        app,
        [
            "sample",
            "--checkpoint", str(checkpoint),
            "--dataset", str(dataset_dir),
            "--source-index", "0",
            "--replica", "0",
            "--out", str(samples_dir),
            "--trajectory",
        ],
    )
    assert result.exit_code == 0, result.output
    written = sorted(p.name for p in samples_dir.glob("*.png"))
    assert written == [
        "src00000_rep00000_average.png",
        "src00000_rep00000_input.png",
        "src00000_rep00000_prediction.png",
        "src00000_rep00000_trajectory.png",
    ]
    assert "PSNR vs clean [prediction]" in result.output

    eval_dir = tmp_path / "eval"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
            "--out", str(eval_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    results = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    assert set(results["methods"]) == {
        "single_frame", "avg_of_n", "one_shot", "iter_average", "iter_prediction",
    }


def test_sample_requires_exactly_one_input_mode(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["sample", "--checkpoint", "x.pt", "--out", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "exactly one of --input or --dataset" in result.output


def test_train_rejects_missing_resume_checkpoint(tmp_path: Path) -> None:
    source_dir = _make_sources(tmp_path)
    dataset_dir = tmp_path / "dataset"
    runner.invoke(
        app,
        [
            "generate",
            "--source-dir", str(source_dir),
            "--output-dir", str(dataset_dir),
            "--num-sources", "3",
            "--replicas", "3",
            "--peak", "30",
        ],
    )
    config_path = _write_config(tmp_path / "config.yml", dataset_dir, tmp_path / "run")
    result = runner.invoke(app, ["train", "--config", str(config_path), "--resume"])
    assert result.exit_code != 0
    assert "resume checkpoint not found" in result.output
