from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from burst_diffusion.config import Config
from burst_diffusion.train import (
    CHECKPOINT_FORMAT,
    LATEST_CHECKPOINT_NAME,
    Trainer,
    load_checkpoint,
)

CHECKPOINT_KEYS = {
    "format",
    "step",
    "config",
    "model",
    "ema",
    "optimizer",
    "torch_rng",
    "cuda_rng",
    "factory",
}


def _write_burst(root: Path, *, num_sources: int = 3, replicas: int = 3) -> Path:
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(num_sources):
        clean = rng.integers(40, 216, size=(20, 24), dtype=np.uint8)
        Image.fromarray(clean).save(burst / "clean" / f"{source_index:05d}.png")
        for replica_index in range(replicas):
            noisy = np.clip(
                clean.astype(np.int16) + rng.integers(-30, 31, clean.shape), 0, 255
            ).astype(np.uint8)
            name = f"noisy/{source_index:05d}_{replica_index:05d}.png"
            Image.fromarray(noisy).save(burst / name)
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


def _config(dataset_dir: Path, run_dir: Path, **training_overrides) -> Config:
    training = {
        "run_dir": str(run_dir),
        "batch_size": 2,
        "max_steps": 3,
        "log_every": 1,
        "val_every": 2,
        "val_images": 2,
        "checkpoint_every": 2,
        "device": "cpu",
        "seed": 5,
    }
    training.update(training_overrides)
    return Config.model_validate(
        {
            "data": {
                "dataset_dir": str(dataset_dir),
                "image_size": 16,
                "channels": 1,
                "val_fraction": 0.34,
            },
            "schedule": {"num_steps": 2},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": training,
            "sampling": {},
        }
    )


def test_short_cpu_run_trains_logs_and_checkpoints(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    run_dir = tmp_path / "run"
    trainer = Trainer(_config(dataset, run_dir))
    checkpoint = trainer.run()

    assert trainer.step == 3
    assert checkpoint == run_dir / LATEST_CHECKPOINT_NAME
    assert checkpoint.is_file()
    assert (run_dir / "ckpt_0000002.pt").is_file()
    assert not list(run_dir.glob("*.tmp"))
    assert list((run_dir / "tb").glob("events.out.tfevents.*"))
    assert (run_dir / "config.yml").is_file()

    payload = load_checkpoint(checkpoint)
    assert set(payload.keys()) == CHECKPOINT_KEYS
    assert payload["step"] == 3
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["ema"] is not None
    total = sum(value.numel() for value in payload["model"].values())
    assert total > 0
    assert all(torch.isfinite(value).all() for value in payload["model"].values())


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")

    full = Trainer(_config(dataset, tmp_path / "run_full", max_steps=4))
    full.run()

    partial = Trainer(_config(dataset, tmp_path / "run_split", max_steps=2))
    checkpoint = partial.run()
    with pytest.warns(UserWarning, match="different config"):
        resumed = Trainer(
            _config(dataset, tmp_path / "run_split", max_steps=4), resume_from=checkpoint
        )
    assert resumed.step == 2
    resumed.run()
    assert resumed.step == 4

    full_state = full.model.state_dict()
    resumed_state = resumed.model.state_dict()
    for name, value in full_state.items():
        assert torch.allclose(value, resumed_state[name], atol=1e-6), name
    for name, value in full.ema.state_dict().items():
        assert torch.allclose(value, resumed.ema.state_dict()[name], atol=1e-6), name

    next_full = full.factory.sample_batch()
    next_resumed = resumed.factory.sample_batch()
    assert torch.equal(next_full.x_t, next_resumed.x_t)
    assert torch.equal(next_full.t, next_resumed.t)
    assert torch.equal(next_full.eps, next_resumed.eps)


def test_stop_file_ends_the_run_early_with_a_checkpoint(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    run_dir = tmp_path / "run"
    trainer = Trainer(_config(dataset, run_dir, max_steps=500))
    (run_dir / "stop").touch()
    checkpoint = trainer.run()
    assert trainer.step == 0
    assert checkpoint.is_file()
    assert load_checkpoint(checkpoint)["step"] == 0


def test_leftover_stop_file_is_cleared_at_startup(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stop").touch()
    trainer = Trainer(_config(dataset, run_dir))
    assert not (run_dir / "stop").exists()
    trainer.run()
    assert trainer.step == 3


def test_training_without_ema_stores_none_and_resume_with_ema_rejects_it(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    trainer = Trainer(_config(dataset, tmp_path / "run", ema=False, max_steps=2))
    checkpoint = trainer.run()
    assert load_checkpoint(checkpoint)["ema"] is None
    with pytest.warns(UserWarning, match="different config"):
        with pytest.raises(ValueError, match="no EMA state"):
            Trainer(_config(dataset, tmp_path / "run", ema=True), resume_from=checkpoint)


def test_load_checkpoint_rejects_foreign_payloads(tmp_path: Path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"format": 99}, path)
    with pytest.raises(ValueError, match="unrecognized checkpoint format"):
        load_checkpoint(path)
