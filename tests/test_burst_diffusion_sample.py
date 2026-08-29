from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from burst_diffusion.config import Config
from burst_diffusion.sample import (
    Sampler,
    load_input_image,
    save_model_image,
    save_trajectory_sheet,
)
from burst_diffusion.train import CHECKPOINT_FORMAT
from burst_diffusion.unet import build_unet


class _ConstantNet(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


class _IdentityNet(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x


def _sampler(net: torch.nn.Module, num_steps: int = 4) -> Sampler:
    return Sampler(net, num_steps=num_steps, device=torch.device("cpu"))


def test_constant_predictions_yield_the_closed_form_average() -> None:
    num_steps = 4
    value = 0.5
    sampler = _sampler(_ConstantNet(value), num_steps)
    x_start = torch.full((2, 1, 8, 8), -0.25)
    result = sampler.run(x_start)
    # Running average of x_T plus T constant "frames": (x + T*c) / (T+1).
    expected = (-0.25 + num_steps * value) / (num_steps + 1)
    assert torch.allclose(result.average, torch.full_like(x_start, expected), atol=1e-6)
    assert torch.allclose(result.prediction, torch.full_like(x_start, value), atol=1e-6)


def test_accelerated_schedule_matches_the_full_rollout_for_a_shared_prediction() -> None:
    sampler = _sampler(_ConstantNet(0.3), num_steps=6)
    x_start = torch.randn(1, 1, 8, 8).clamp(-1, 1)
    full = sampler.run(x_start)
    fast = sampler.run(x_start, schedule=[6, 1])
    assert torch.allclose(full.average, fast.average, atol=1e-6)


def test_identity_predictions_leave_the_input_unchanged() -> None:
    sampler = _sampler(_IdentityNet())
    x_start = torch.randn(3, 1, 8, 8).clamp(-1, 1)
    result = sampler.run(x_start)
    assert torch.allclose(result.average, x_start, atol=1e-6)
    assert torch.allclose(result.prediction, x_start, atol=1e-6)


def test_outputs_are_clamped_and_trajectory_tracks_every_step() -> None:
    sampler = _sampler(_ConstantNet(5.0), num_steps=4)
    x_start = torch.zeros(1, 1, 8, 8)
    result = sampler.run(x_start, keep_trajectory=True)
    assert result.average.max() <= 1.0
    assert result.prediction.max() <= 1.0
    assert result.trajectory is not None
    assert len(result.trajectory) == 4
    for step in result.trajectory:
        assert step.shape == (1, 1, 8, 8)
        assert step.device.type == "cpu"
    no_history = sampler.run(x_start)
    assert no_history.trajectory is None


def test_invalid_schedules_and_shapes_are_rejected() -> None:
    sampler = _sampler(_IdentityNet())
    x = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="strictly decreasing"):
        sampler.run(x, schedule=[4, 4, 1])
    with pytest.raises(ValueError, match=r"levels must be in \[1, 4\]"):
        sampler.run(x, schedule=[5, 1])
    with pytest.raises(ValueError, match="at least one level"):
        sampler.run(x, schedule=[])
    with pytest.raises(ValueError, match=r"x_start must be \[B, C, H, W\]"):
        sampler.run(torch.zeros(1, 8, 8))


def _write_checkpoint(tmp_path: Path, *, ema_value: float | None) -> tuple[Path, Config, dict]:
    config = Config.model_validate(
        {
            "data": {"dataset_dir": "data/x", "image_size": 16, "channels": 1},
            "schedule": {"num_steps": 2},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": "runs/x", "device": "cpu"},
            "sampling": {},
        }
    )
    torch.manual_seed(0)
    model = build_unet(config)
    ema_state = None
    if ema_value is not None:
        ema_state = {
            name: torch.full_like(param, ema_value)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": 7,
        "config": config.model_dump(mode="json"),
        "model": model.state_dict(),
        "ema": ema_state,
        "optimizer": {},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "factory": {"rng_state": np.random.default_rng(0).bit_generator.state},
    }
    path = tmp_path / "ckpt.pt"
    torch.save(payload, path)
    return path, config, model.state_dict()


def test_from_checkpoint_uses_the_ema_shadow_when_asked(tmp_path: Path) -> None:
    path, _, live_state = _write_checkpoint(tmp_path, ema_value=0.123)
    ema_sampler = Sampler.from_checkpoint(path, device="cpu", use_ema=True)
    for name, param in ema_sampler.model.named_parameters():
        assert torch.allclose(param.data, torch.full_like(param.data, 0.123)), name
    live_sampler = Sampler.from_checkpoint(path, device="cpu", use_ema=False)
    for name, param in live_sampler.model.named_parameters():
        assert torch.allclose(param.data, live_state[name]), name
    assert ema_sampler.num_steps == 2
    assert ema_sampler.config is not None


def test_from_checkpoint_without_ema_warns_and_falls_back(tmp_path: Path) -> None:
    path, _, live_state = _write_checkpoint(tmp_path, ema_value=None)
    with pytest.warns(UserWarning, match="no EMA state"):
        sampler = Sampler.from_checkpoint(path, device="cpu", use_ema=True)
    for name, param in sampler.model.named_parameters():
        assert torch.allclose(param.data, live_state[name]), name


def test_input_image_is_center_cropped_never_resized(tmp_path: Path) -> None:
    array = np.arange(20 * 30, dtype=np.float64).reshape(20, 30) % 256
    path = tmp_path / "input.png"
    Image.fromarray(array.astype(np.uint8)).save(path)
    tensor = load_input_image(path, image_size=16, channels=1)
    assert tensor.shape == (1, 1, 16, 16)
    expected = array[2:18, 7:23] / 255.0 * 2.0 - 1.0
    np.testing.assert_allclose(tensor[0, 0].numpy(), expected.astype(np.float32), atol=1e-6)
    with pytest.raises(ValueError, match="crops only, no resizing"):
        load_input_image(path, image_size=32, channels=1)


def test_model_images_and_trajectory_sheets_round_trip_to_png(tmp_path: Path) -> None:
    tensor = torch.linspace(-1, 1, 16).reshape(1, 4, 4)
    out = save_model_image(tensor, tmp_path / "img.png")
    saved = np.asarray(Image.open(out), dtype=np.float64)
    np.testing.assert_allclose(
        saved, np.rint(((tensor[0].numpy() + 1) / 2) * 255), atol=0
    )
    trajectory = [torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4)]
    sheet = save_trajectory_sheet(trajectory, tmp_path / "traj.png")
    assert Image.open(sheet).size == (8, 4)  # two 4x4 tiles side by side
    with pytest.raises(ValueError, match="trajectory is empty"):
        save_trajectory_sheet([], tmp_path / "empty.png")
