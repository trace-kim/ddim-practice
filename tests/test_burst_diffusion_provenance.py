from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from burst_diffusion.config import Config
from burst_diffusion.provenance import (
    PROVENANCE_NAME,
    dataset_fingerprint,
    file_sha256,
    write_provenance,
)
from burst_diffusion.train import CHECKPOINT_FORMAT


def _write_burst(root: Path, *, num_sources: int = 4, duplicate: bool = False) -> Path:
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(num_sources):
        # With duplicate=True sources 2,3 repeat the content of 0,1.
        content_index = source_index % 2 if duplicate else source_index
        clean = np.full((20, 24), 80 + content_index * 20, dtype=np.uint8)
        clean[:, ::3] = 200
        Image.fromarray(clean).save(burst / "clean" / f"{source_index:05d}.png")
        for replica_index in range(3):
            noisy = np.clip(
                clean.astype(np.int16) + rng.integers(-20, 21, clean.shape), 0, 255
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


def _config(dataset_dir: Path, run_dir: Path, **data_overrides) -> Config:
    data = {
        "dataset_dir": str(dataset_dir),
        "image_size": 16,
        "channels": 1,
        "val_fraction": 0.25,
    }
    data.update(data_overrides)
    return Config.model_validate(
        {
            "data": data,
            "schedule": {"num_steps": 2},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(run_dir), "device": "cpu"},
            "sampling": {},
        }
    )


def test_provenance_records_code_data_environment_and_checkpoint(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    config = _config(dataset, tmp_path / "run")
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"format": CHECKPOINT_FORMAT, "step": 4242}, checkpoint)

    path = write_provenance(
        tmp_path / "run", config, config_path="configs/x.yml",
        checkpoint=checkpoint, command="python -m burst_diffusion train --config configs/x.yml",
    )
    assert path.name == PROVENANCE_NAME
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["config"]["schedule"]["num_steps"] == 2
    assert len(record["config_sha256"]) == 64
    assert record["checkpoint"]["step"] == 4242
    assert record["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert record["environment"]["torch"] == torch.__version__
    assert record["command"].startswith("python -m burst_diffusion")
    assert record["dataset"]["sources"] == 4
    assert record["dataset"]["distinct_contents"] == 4
    assert set(record["dataset"]["split"]) == {"train", "val", "test"}
    assert "git" in record


def test_dataset_fingerprint_detects_a_content_change(tmp_path: Path) -> None:
    """The digest is the guard that would catch a dataset quietly changing
    under a run -- including gaining duplicate content (report S9.1)."""
    first = dataset_fingerprint(_config(_write_burst(tmp_path / "a"), tmp_path / "run"))
    same = dataset_fingerprint(_config(_write_burst(tmp_path / "b"), tmp_path / "run"))
    assert first["content_digest_sha256"] == same["content_digest_sha256"]

    with pytest.warns(UserWarning, match="duplicated clean image"):
        duplicated = dataset_fingerprint(
            _config(_write_burst(tmp_path / "c", duplicate=True), tmp_path / "run")
        )
    assert duplicated["content_digest_sha256"] != first["content_digest_sha256"]
    assert duplicated["sources"] == 4
    assert duplicated["distinct_contents"] == 2
    assert duplicated["duplicate_groups"]


def test_provenance_records_the_locked_test_split(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data", num_sources=8)
    config = _config(dataset, tmp_path / "run", val_fraction=0.25, test_fraction=0.25)
    record = json.loads(
        write_provenance(tmp_path / "run", config).read_text(encoding="utf-8")
    )
    split = record["dataset"]["split"]
    assert len(split["test"]) == 2 and len(split["val"]) == 2
    assert not set(split["test"]) & set(split["train"])
    assert not set(split["test"]) & set(split["val"])
