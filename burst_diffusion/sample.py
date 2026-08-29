"""Iterative burst-diffusion sampling (the DDIM analog).

Start from a noisy measurement ``x_T`` (a single burst frame, or an average of
a few real frames) and walk a strictly decreasing schedule of noise levels: at
each step the network predicts a plausible fresh frame ``eps_hat`` and the
cumulative-average update folds it in as if it were ``m(t_next) - m(t)`` real
acquisitions. Two outputs fall out of the loop:

- ``average``: the final running average (the spec-faithful recursion result;
  still carries ``1/(T+1)`` of the original real-frame noise), and
- ``prediction``: the last ``eps_hat``, whose MSE-optimal value is
  ``E[clean | current average]`` -- theoretically the cleanest estimate.

Evaluation reports both. Everything stays on one device; the trajectory is
only materialized on request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .config import Config
from .schedule import sample_step, sampling_schedule
from .train import load_checkpoint, resolve_device
from .unet import build_unet


@dataclass
class SampleResult:
    average: torch.Tensor  # [B, C, H, W] float32 in [-1, 1], on CPU
    prediction: torch.Tensor  # [B, C, H, W] float32 in [-1, 1], on CPU
    trajectory: list[torch.Tensor] | None  # per schedule step, on CPU


class Sampler:
    def __init__(self, model: torch.nn.Module, *, num_steps: int, device: torch.device):
        self.model = model.to(device).eval()
        self.num_steps = num_steps
        self.device = device
        self.config: Config | None = None

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: str = "auto", use_ema: bool = True
    ) -> "Sampler":
        resolved = resolve_device(device)
        payload = load_checkpoint(path, map_location=resolved)
        config = Config.model_validate(payload["config"])
        model = build_unet(config)
        model.load_state_dict(payload["model"])
        if use_ema:
            if payload["ema"] is None:
                import warnings

                warnings.warn(
                    f"use_ema=True but {path} has no EMA state; using the live weights",
                    stacklevel=2,
                )
            else:
                named = dict(model.named_parameters())
                for name, value in payload["ema"].items():
                    named[name].data.copy_(value)
        sampler = cls(model, num_steps=config.schedule.num_steps, device=resolved)
        sampler.config = config
        return sampler

    def _validated_schedule(self, schedule: Sequence[int] | None) -> list[int]:
        if schedule is None:
            return sampling_schedule(self.num_steps, None)
        levels = [int(level) for level in schedule]
        if not levels:
            raise ValueError("schedule must contain at least one level")
        if any(not 1 <= level <= self.num_steps for level in levels):
            raise ValueError(f"schedule levels must be in [1, {self.num_steps}], got {levels}")
        if any(a <= b for a, b in zip(levels, levels[1:])):
            raise ValueError(f"schedule must be strictly decreasing, got {levels}")
        return levels

    def run(
        self,
        x_start: torch.Tensor,
        *,
        schedule: Sequence[int] | None = None,
        keep_trajectory: bool = False,
    ) -> SampleResult:
        if x_start.dim() != 4:
            raise ValueError(f"x_start must be [B, C, H, W], got shape {tuple(x_start.shape)}")
        levels = self._validated_schedule(schedule)
        x = x_start.to(device=self.device, dtype=torch.float32)
        batch = x.shape[0]
        trajectory: list[torch.Tensor] | None = [] if keep_trajectory else None
        eps_hat = x
        with torch.no_grad():
            for t, t_next in zip(levels, list(levels[1:]) + [0]):
                t_tensor = torch.full((batch,), float(t), device=self.device)
                eps_hat = self.model(x, t_tensor)
                x = sample_step(x, eps_hat, t, t_next, self.num_steps)
                if trajectory is not None:
                    trajectory.append(x.clamp(-1.0, 1.0).cpu())
        return SampleResult(
            average=x.clamp(-1.0, 1.0).cpu(),
            prediction=eps_hat.clamp(-1.0, 1.0).cpu(),
            trajectory=trajectory,
        )


def load_input_image(path: str | Path, *, image_size: int, channels: int) -> torch.Tensor:
    """Load a PNG/TIFF measurement as a ``[1, C, S, S]`` tensor in [-1, 1].

    The image is center-CROPPED to ``image_size`` (never resized: resampling a
    noisy frame partially denoises it and changes the noise statistics the
    model was trained on).
    """
    with Image.open(path) as image:
        converted = image.convert("L" if channels == 1 else "RGB")
        array = np.asarray(converted, dtype=np.float32)
    height, width = array.shape[:2]
    if height < image_size or width < image_size:
        raise ValueError(
            f"input {path} is {height}x{width}, smaller than the model resolution "
            f"{image_size}; provide a larger image (crops only, no resizing)"
        )
    top = (height - image_size) // 2
    left = (width - image_size) // 2
    crop = array[top : top + image_size, left : left + image_size]
    normalized = crop / 255.0 * 2.0 - 1.0
    if normalized.ndim == 2:
        normalized = normalized[None, :, :]
    else:
        normalized = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return torch.from_numpy(normalized)[None]


def save_model_image(tensor: torch.Tensor, path: str | Path) -> Path:
    """Save a ``[C, H, W]`` tensor in [-1, 1] as an 8-bit PNG."""
    if tensor.dim() != 3:
        raise ValueError(f"expected [C, H, W], got shape {tuple(tensor.shape)}")
    array01 = ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
    array = np.rint(array01 * 255.0).astype(np.uint8)
    image = Image.fromarray(array[0]) if array.shape[0] == 1 else Image.fromarray(
        np.ascontiguousarray(array.transpose(1, 2, 0))
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination


def save_trajectory_sheet(
    trajectory: Sequence[torch.Tensor], path: str | Path, *, batch_index: int = 0
) -> Path:
    """Save one batch item's trajectory as a horizontal contact sheet PNG."""
    if not trajectory:
        raise ValueError("trajectory is empty")
    tiles = [step[batch_index] for step in trajectory]
    strip = torch.cat(tiles, dim=-1)
    return save_model_image(strip, path)
