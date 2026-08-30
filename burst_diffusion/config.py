"""Typed configuration for burst_diffusion: YAML -> validated pydantic models.

Unknown keys are rejected everywhere (``extra="forbid"``) so typos fail fast.
Cross-field validators enforce the U-Net's structural constraints up front
(spatial divisibility, GroupNorm divisibility, reachable attention
resolutions) instead of failing deep inside torch.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictModel):
    dataset_dir: Path
    image_size: int = Field(default=64, ge=8)
    channels: Literal[1, 3] = 1
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    split_seed: int = Field(default=2019, ge=0)

    @model_validator(mode="after")
    def _check_holdout_fractions(self) -> "DataConfig":
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"data.val_fraction + data.test_fraction must be < 1, got "
                f"{self.val_fraction} + {self.test_fraction}"
            )
        return self


class ScheduleConfig(_StrictModel):
    num_steps: int = Field(default=15, ge=1)
    target_mode: Literal["fresh", "included"] = "fresh"


class ModelConfig(_StrictModel):
    ch: int = Field(default=64, ge=4)
    out_ch: int | None = Field(default=None, ge=1)
    ch_mult: list[int] = Field(default=[1, 2, 2, 2], min_length=1)
    num_res_blocks: int = Field(default=2, ge=1)
    attn_resolutions: list[int] = Field(default=[16])
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    resamp_with_conv: bool = True
    num_groups: int | None = Field(default=None, ge=1)

    @property
    def effective_num_groups(self) -> int:
        return self.num_groups if self.num_groups is not None else min(32, self.ch)


class RolloutConfig(_StrictModel):
    """Opt-in self-rollout finetuning stage (method doc S5, Q&A Q6).

    A ``fraction`` of every batch is replaced by the sampler's own pseudo-average
    states (generated on the fly) as network INPUTS, conditioned on the nominal
    sampler level; the targets for those samples remain REAL fresh frames.
    ``fraction: 0`` (or omitting the block) is bit-identical to baseline training.
    """

    fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    use_ema: bool = True


class TrainingConfig(_StrictModel):
    run_dir: Path
    batch_size: int = Field(default=16, ge=1)
    max_steps: int = Field(default=30000, ge=1)
    lr: float = Field(default=2.0e-4, gt=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_eps: float = Field(default=1.0e-8, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    ema: bool = True
    ema_rate: float = Field(default=0.999, ge=0.0, lt=1.0)
    antithetic: bool = True
    seed: int = Field(default=0, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_every: int = Field(default=50, ge=1)
    val_every: int = Field(default=1000, ge=1)
    val_images: int = Field(default=8, ge=1)
    checkpoint_every: int = Field(default=2000, ge=1)
    rollout: RolloutConfig | None = None


class SamplingConfig(_StrictModel):
    output_mode: Literal["average", "prediction", "both"] = "both"
    num_sample_steps: int | None = Field(default=None, ge=1)


class Config(_StrictModel):
    data: DataConfig
    schedule: ScheduleConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig

    @property
    def effective_out_ch(self) -> int:
        return self.model.out_ch if self.model.out_ch is not None else self.data.channels

    @property
    def rollout_active(self) -> bool:
        rollout = self.training.rollout
        return rollout is not None and rollout.fraction > 0.0

    @model_validator(mode="after")
    def _check_structure(self) -> "Config":
        image_size = self.data.image_size
        num_levels = len(self.model.ch_mult)
        divisor = 2 ** (num_levels - 1)
        if image_size % divisor != 0:
            raise ValueError(
                f"data.image_size ({image_size}) must be divisible by "
                f"2**(len(model.ch_mult)-1) = {divisor} so every "
                "downsample/upsample level lines up"
            )
        for mult in self.model.ch_mult:
            if mult < 1:
                raise ValueError(f"model.ch_mult entries must be >= 1, got {mult}")
        groups = self.model.effective_num_groups
        if self.model.ch % groups != 0:
            raise ValueError(
                f"model.num_groups ({groups}) must divide model.ch "
                f"({self.model.ch}); every layer width is a multiple of ch"
            )
        level_resolutions = {image_size >> level for level in range(num_levels)}
        for resolution in self.model.attn_resolutions:
            if resolution not in level_resolutions:
                raise ValueError(
                    f"model.attn_resolutions entry {resolution} is not one of the "
                    f"reachable level resolutions {sorted(level_resolutions, reverse=True)}"
                )
            if resolution > 16:
                warnings.warn(
                    f"attention at resolution {resolution} costs O((H*W)^2) memory; "
                    "resolutions above 16 are rarely affordable",
                    stacklevel=2,
                )
        if self.rollout_active:
            if self.schedule.num_steps < 2:
                raise ValueError(
                    "training.rollout needs schedule.num_steps >= 2: with T = 1 the "
                    "sampler has no intermediate pseudo-average states to train on"
                )
            if self.training.rollout.use_ema and not self.training.ema:
                raise ValueError(
                    "training.rollout.use_ema requires training.ema: inference samples "
                    "with the EMA weights, so rollout states should be generated with "
                    "them too (set rollout.use_ema: false to use the live weights)"
                )
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}: {config_path}")
    return Config.model_validate(raw)
