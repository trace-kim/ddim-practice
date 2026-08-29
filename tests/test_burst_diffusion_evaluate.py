from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from burst_diffusion.config import Config
from burst_diffusion.evaluate import METHOD_NAMES, evaluate
from burst_diffusion.train import CHECKPOINT_FORMAT
from burst_diffusion.unet import build_unet


def _write_burst(root: Path, *, size: tuple[int, int], num_sources: int = 4) -> Path:
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(num_sources):
        clean = rng.integers(80, 176, size=size, dtype=np.uint8)
        Image.fromarray(clean).save(burst / "clean" / f"{source_index:05d}.png")
        for replica_index in range(3):
            # Zero-mean +-1 offsets stay inside [0, 255]: no clipping bias, so
            # averaging must beat the single frame.
            noisy = (clean.astype(np.int16) + rng.integers(-30, 31, size)).clip(0, 255)
            name = f"noisy/{source_index:05d}_{replica_index:05d}.png"
            Image.fromarray(noisy.astype(np.uint8)).save(burst / name)
            rows.append(
                json.dumps(
                    {
                        "source_index": source_index,
                        "replica_index": replica_index,
                        "clean_path": f"clean/{source_index:05d}.png",
                        "noisy_path": name,
                    }
                )
            )
    (burst / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _config(dataset_dir: Path, run_dir: Path) -> Config:
    return Config.model_validate(
        {
            "data": {
                "dataset_dir": str(dataset_dir),
                "image_size": 16,
                "channels": 1,
                "val_fraction": 0.5,
            },
            "schedule": {"num_steps": 2},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(run_dir), "device": "cpu"},
            "sampling": {},
        }
    )


def _write_checkpoint(path: Path, config: Config) -> Path:
    torch.manual_seed(0)
    model = build_unet(config)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": 1,
        "config": config.model_dump(mode="json"),
        "model": model.state_dict(),
        "ema": None,
        "optimizer": {},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "factory": {"rng_state": np.random.default_rng(0).bit_generator.state},
    }
    torch.save(payload, path)
    return path


def _run(tmp_path: Path, *, size: tuple[int, int] = (20, 24), **kwargs) -> dict:
    dataset = _write_burst(tmp_path / "data", size=size)
    config = _config(dataset, tmp_path / "run")
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt", config)
    with pytest.warns(UserWarning, match="no EMA state"):
        return evaluate(
            config, checkpoint, out_dir=tmp_path / "eval", **kwargs
        )


def test_evaluate_reports_exactly_the_five_methods_with_finite_metrics(tmp_path: Path) -> None:
    results = _run(tmp_path)
    assert set(results["methods"].keys()) == set(METHOD_NAMES)
    assert results["split"] == "val"
    assert results["count"] == 2
    for method in results["methods"].values():
        assert np.isfinite(method["psnr_mean"])
        assert np.isfinite(method["ssim_mean"])
        assert len(method["psnr_per_image"]) == 2
        assert len(method["ssim_per_image"]) == 2
    assert (tmp_path / "eval" / "results.json").is_file()
    assert (tmp_path / "eval" / "comparison_grid.png").is_file()
    reloaded = json.loads((tmp_path / "eval" / "results.json").read_text(encoding="utf-8"))
    assert reloaded["methods"]["avg_of_n"]["psnr_mean"] == pytest.approx(
        results["methods"]["avg_of_n"]["psnr_mean"]
    )


def test_averaging_beats_the_single_frame_on_zero_mean_noise(tmp_path: Path) -> None:
    results = _run(tmp_path)
    assert (
        results["methods"]["avg_of_n"]["psnr_mean"]
        > results["methods"]["single_frame"]["psnr_mean"]
    )


def test_limit_and_train_split_are_respected(tmp_path: Path) -> None:
    results = _run(tmp_path, split="train", limit=1)
    assert results["split"] == "train"
    assert results["count"] == 1


def test_tile_mode_covers_the_full_image(tmp_path: Path) -> None:
    results = _run(tmp_path, tile=True)
    assert results["tile"] is True
    for method in results["methods"].values():
        assert np.isfinite(method["psnr_mean"])


def test_tile_mode_degenerates_to_the_crop_result_when_sizes_match(tmp_path: Path) -> None:
    tiled = _run(tmp_path, size=(16, 16), tile=True)
    cropped = _run(tmp_path / "again", size=(16, 16), tile=False)
    for name in METHOD_NAMES:
        assert tiled["methods"][name]["psnr_mean"] == pytest.approx(
            cropped["methods"][name]["psnr_mean"], abs=1e-9
        )


def test_mismatched_num_steps_is_rejected(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data", size=(20, 24))
    config = _config(dataset, tmp_path / "run")
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt", config)
    other = config.model_copy(deep=True)
    other.schedule.num_steps = 1
    with pytest.raises(ValueError, match="trained with num_steps=2"):
        evaluate(other, checkpoint, out_dir=tmp_path / "eval")
