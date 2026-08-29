"""Burst dataset access: locate, cache, and batch noisy-burst training data.

Design notes (see docs in the package README):

- No ``DataLoader``. The repo has a documented worker-restart trap on small
  datasets, and burst sampling needs random access to *all* frames of a source
  at once. Frames are decoded once into an in-RAM uint8 cache
  (:class:`BurstCache`) and batches are assembled by a seeded
  :class:`BatchFactory` -- deterministic, resumable, and trivially testable.
- Noisy frames are NEVER resized (resampling partially denoises and correlates
  the noise). Batches use aligned random crops: one window per sample, shared
  by every frame of that source.
- The training target is a FRESH frame excluded from the averaged subset; an
  included target makes the MSE-optimal network the identity (exchangeability
  gives E[frame | mean of subset] = the mean itself). ``target_mode="included"``
  exists only as a documented ablation and warns at construction.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .schedule import frames_at, min_replicas

_SIXTEEN_BIT_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I")


def resolve_burst_dir(dataset_dir: str | Path) -> Path:
    """Return the directory that holds ``manifest.jsonl`` + ``clean/`` + ``noisy/``.

    Accepts either the burst directory itself or a dataset root produced by
    ``generate_burst_dataset`` (which nests the pipeline output under
    ``burst/``).
    """
    root = Path(dataset_dir)
    for candidate in (root, root / "burst"):
        if (candidate / "manifest.jsonl").is_file():
            return candidate
    raise FileNotFoundError(
        f"no manifest.jsonl under {root} or {root / 'burst'}; "
        "expected a burst dataset produced by 'python -m burst_diffusion generate' "
        "or noising_pipeline.create_noisy_dataset"
    )


@dataclass
class BurstSource:
    """One clean image with all of its noisy burst frames, cached as uint8."""

    source_index: int
    clean: np.ndarray
    frames: list[np.ndarray]


@dataclass(frozen=True)
class SampleInfo:
    """Provenance of one training sample, exposed for tests and debugging."""

    source_index: int
    t: int
    crop_yx: tuple[int, int]
    subset_replicas: tuple[int, ...]
    target_replica: int


@dataclass
class TrainingBatch:
    x_t: torch.Tensor  # [B, C, S, S] float32 in [-1, 1]
    t: torch.Tensor  # [B] float32 in {1..T}
    eps: torch.Tensor  # [B, C, S, S] float32 in [-1, 1]


@dataclass
class ValidationBatch:
    x_t: torch.Tensor
    t: torch.Tensor
    eps: torch.Tensor
    clean: torch.Tensor  # [B, C, S, S] float32 in [-1, 1]


class BurstCache:
    """Load a burst dataset's manifest and cache every frame in RAM as uint8."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        channels: int = 1,
        min_replicas: int = 2,
        min_size: int = 1,
        val_fraction: float = 0.1,
        split_seed: int = 2019,
    ):
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {channels}")
        if min_replicas < 1:
            raise ValueError(f"min_replicas must be >= 1, got {min_replicas}")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
        self.burst_dir = resolve_burst_dir(dataset_dir)
        self.channels = channels

        grouped = self._parse_manifest(self.burst_dir / "manifest.jsonl")

        self._warned_color_conversion = False
        kept: list[BurstSource] = []
        self.dropped_replicas: list[int] = []
        self.dropped_size: list[int] = []
        for source_index in sorted(grouped):
            clean_path, replicas = grouped[source_index]
            if len(replicas) < min_replicas:
                self.dropped_replicas.append(source_index)
                continue
            clean = self._load_uint8(self.burst_dir / clean_path)
            if min(clean.shape[0], clean.shape[1]) < min_size:
                self.dropped_size.append(source_index)
                continue
            frames = []
            for _, noisy_path in sorted(replicas):
                frame = self._load_uint8(self.burst_dir / noisy_path)
                if frame.shape != clean.shape:
                    raise ValueError(
                        f"replica shape {frame.shape} != clean shape {clean.shape} "
                        f"for source {source_index}: {noisy_path}"
                    )
                frames.append(frame)
            kept.append(BurstSource(source_index=source_index, clean=clean, frames=frames))
        if self.dropped_replicas:
            warnings.warn(
                f"dropped {len(self.dropped_replicas)} source(s) with fewer than "
                f"{min_replicas} replicas (e.g. an interrupted generation run): "
                f"{self.dropped_replicas[:8]}",
                stacklevel=2,
            )
        if self.dropped_size:
            warnings.warn(
                f"dropped {len(self.dropped_size)} source(s) smaller than {min_size} px: "
                f"{self.dropped_size[:8]}",
                stacklevel=2,
            )
        if not kept:
            raise RuntimeError(
                f"no usable sources in {self.burst_dir}: "
                f"{len(self.dropped_replicas)} dropped for replica count, "
                f"{len(self.dropped_size)} dropped for size"
            )

        order = np.random.RandomState(split_seed).permutation(len(kept))
        val_count = 0
        if val_fraction > 0.0:
            if len(kept) > 1:
                val_count = max(1, min(len(kept) - 1, round(val_fraction * len(kept))))
            else:
                warnings.warn(
                    "only one usable source; keeping it for training and leaving "
                    "the validation split empty",
                    stacklevel=2,
                )
        val_positions = set(order[len(kept) - val_count :].tolist())
        self.train_sources = [kept[i] for i in range(len(kept)) if i not in val_positions]
        self.val_sources = [kept[i] for i in range(len(kept)) if i in val_positions]

    @staticmethod
    def _parse_manifest(manifest_path: Path) -> dict[int, tuple[str, list[tuple[int, str]]]]:
        grouped: dict[int, tuple[str, list[tuple[int, str]]]] = {}
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                source_index = int(row["source_index"])
                replica_index = int(row["replica_index"])
                clean_path = str(row["clean_path"])
                noisy_path = str(row["noisy_path"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                if line_number == len(lines):
                    warnings.warn(
                        f"skipping truncated final manifest line {line_number} "
                        f"(interrupted generation?): {manifest_path}",
                        stacklevel=3,
                    )
                    continue
                raise ValueError(
                    f"malformed manifest line {line_number} in {manifest_path}: {error}"
                ) from error
            if source_index in grouped and grouped[source_index][0] != clean_path:
                raise ValueError(
                    f"conflicting clean_path entries for source {source_index} in {manifest_path}"
                )
            grouped.setdefault(source_index, (clean_path, []))[1].append(
                (replica_index, noisy_path)
            )
        if not grouped:
            raise RuntimeError(f"manifest has no usable rows: {manifest_path}")
        return grouped

    def _load_uint8(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            if image.mode in _SIXTEEN_BIT_MODES:
                array = np.asarray(image, dtype=np.float64)
                array = np.clip(array / 257.0, 0.0, 255.0)
                gray = np.rint(array).astype(np.uint8)
                return gray if self.channels == 1 else np.stack([gray] * 3, axis=-1)
            if self.channels == 1 and image.mode != "L" and not self._warned_color_conversion:
                self._warned_color_conversion = True
                warnings.warn(
                    f"converting {image.mode} frames to grayscale averages the color "
                    "channels and partially denoises independently-noised data; prefer "
                    "generating grayscale bursts (generate ... --grayscale)",
                    stacklevel=3,
                )
            converted = image.convert("L" if self.channels == 1 else "RGB")
            return np.asarray(converted).copy()

    def summary(self) -> dict:
        ram_bytes = sum(
            source.clean.nbytes + sum(frame.nbytes for frame in source.frames)
            for source in self.train_sources + self.val_sources
        )
        return {
            "burst_dir": str(self.burst_dir),
            "train_sources": len(self.train_sources),
            "val_sources": len(self.val_sources),
            "val_source_indices": [source.source_index for source in self.val_sources],
            "dropped_replicas": list(self.dropped_replicas),
            "dropped_size": list(self.dropped_size),
            "min_frames": min(
                len(source.frames) for source in self.train_sources + self.val_sources
            ),
            "ram_bytes": ram_bytes,
        }


def _to_model_range(crop: np.ndarray) -> np.ndarray:
    return crop.astype(np.float32) / 255.0 * 2.0 - 1.0


def _to_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[None, :, :]
    return np.ascontiguousarray(image.transpose(2, 0, 1))


class BatchFactory:
    """Seeded assembler of training/validation batches from a :class:`BurstCache`."""

    def __init__(
        self,
        cache: BurstCache,
        *,
        num_steps: int,
        image_size: int,
        batch_size: int,
        target_mode: str = "fresh",
        antithetic: bool = True,
        seed: int = 0,
    ):
        if target_mode not in ("fresh", "included"):
            raise ValueError(f"target_mode must be 'fresh' or 'included', got {target_mode!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not cache.train_sources:
            raise ValueError("cache has no training sources")
        required = min_replicas(num_steps)
        for source in cache.train_sources + cache.val_sources:
            if len(source.frames) < required:
                raise ValueError(
                    f"source {source.source_index} has {len(source.frames)} frames but "
                    f"num_steps={num_steps} needs at least {required} (T averaged + 1 fresh "
                    "target); lower schedule.num_steps or regenerate with more replicas"
                )
            if min(source.clean.shape[0], source.clean.shape[1]) < image_size:
                raise ValueError(
                    f"source {source.source_index} is {source.clean.shape[:2]} but crops "
                    f"need at least {image_size}x{image_size}"
                )
        if target_mode == "included":
            warnings.warn(
                "target_mode='included' is a documented-degenerate ablation: the target "
                "frame is inside the averaged subset, so the MSE-optimal network is the "
                "identity. Use 'fresh' for real training.",
                stacklevel=2,
            )
        self.cache = cache
        self.num_steps = num_steps
        self.image_size = image_size
        self.batch_size = batch_size
        self.target_mode = target_mode
        self.antithetic = antithetic
        self._rng = np.random.default_rng(seed)

    def _sample_t_values(self) -> np.ndarray:
        if not self.antithetic:
            return self._rng.integers(1, self.num_steps + 1, size=self.batch_size)
        half = math.ceil(self.batch_size / 2)
        drawn = self._rng.integers(1, self.num_steps + 1, size=half)
        mirrored = self.num_steps + 1 - drawn
        return np.concatenate([drawn, mirrored])[: self.batch_size]

    def sample_batch(
        self, *, return_info: bool = False
    ) -> TrainingBatch | tuple[TrainingBatch, list[SampleInfo]]:
        t_values = self._sample_t_values()
        size = self.image_size
        x_list: list[np.ndarray] = []
        eps_list: list[np.ndarray] = []
        info: list[SampleInfo] = []
        for t in t_values:
            t = int(t)
            source = self.cache.train_sources[
                int(self._rng.integers(len(self.cache.train_sources)))
            ]
            height, width = source.clean.shape[:2]
            top = int(self._rng.integers(0, height - size + 1))
            left = int(self._rng.integers(0, width - size + 1))
            permutation = self._rng.permutation(len(source.frames))
            frame_count = frames_at(t, self.num_steps)
            subset = permutation[:frame_count]
            target = int(subset[0]) if self.target_mode == "included" else int(permutation[frame_count])

            window = np.s_[top : top + size, left : left + size]
            accumulator = np.zeros(source.frames[0][window].shape, dtype=np.float32)
            for replica in subset:
                accumulator += source.frames[int(replica)][window]
            average = accumulator / (frame_count * 255.0) * 2.0 - 1.0
            x_list.append(_to_chw(average))
            eps_list.append(_to_chw(_to_model_range(source.frames[target][window])))
            if return_info:
                info.append(
                    SampleInfo(
                        source_index=source.source_index,
                        t=t,
                        crop_yx=(top, left),
                        subset_replicas=tuple(int(r) for r in subset),
                        target_replica=target,
                    )
                )
        batch = TrainingBatch(
            x_t=torch.from_numpy(np.stack(x_list)),
            t=torch.as_tensor(t_values, dtype=torch.float32),
            eps=torch.from_numpy(np.stack(eps_list)),
        )
        if return_info:
            return batch, info
        return batch

    def val_batch(self, *, level: int, count: int) -> ValidationBatch:
        """Deterministic validation batch: center crops, fixed frame subsets."""
        if not 1 <= level <= self.num_steps:
            raise ValueError(f"level must be in [1, {self.num_steps}], got {level}")
        if not self.cache.val_sources:
            raise ValueError("cache has no validation sources")
        size = self.image_size
        frame_count = frames_at(level, self.num_steps)
        x_list: list[np.ndarray] = []
        eps_list: list[np.ndarray] = []
        clean_list: list[np.ndarray] = []
        for index in range(count):
            source = self.cache.val_sources[index % len(self.cache.val_sources)]
            height, width = source.clean.shape[:2]
            top = (height - size) // 2
            left = (width - size) // 2
            window = np.s_[top : top + size, left : left + size]
            accumulator = np.zeros(source.frames[0][window].shape, dtype=np.float32)
            for replica in range(frame_count):
                accumulator += source.frames[replica][window]
            average = accumulator / (frame_count * 255.0) * 2.0 - 1.0
            x_list.append(_to_chw(average))
            eps_list.append(_to_chw(_to_model_range(source.frames[frame_count][window])))
            clean_list.append(_to_chw(_to_model_range(source.clean[window])))
        return ValidationBatch(
            x_t=torch.from_numpy(np.stack(x_list)),
            t=torch.full((count,), float(level), dtype=torch.float32),
            eps=torch.from_numpy(np.stack(eps_list)),
            clean=torch.from_numpy(np.stack(clean_list)),
        )

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng_state"]
