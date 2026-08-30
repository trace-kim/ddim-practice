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

import hashlib
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


def content_key(array: np.ndarray) -> str:
    """Content hash of a clean image (shape-tagged, so different sizes never collide).

    The identity used for group-aware splitting and for generation-time
    deduplication: two sources with the same key are the same picture and must
    never land on opposite sides of a train/validation split.
    """
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


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


@dataclass(frozen=True)
class RolloutPairInfo:
    """Provenance of one self-rollout training pair, for tests and debugging."""

    source_index: int
    crop_yx: tuple[int, int]
    seed_replica: int
    target_replica: int
    stop_level: int


@dataclass
class RolloutPairBatch:
    """Raw material for self-rollout finetuning (method doc S5).

    The data layer stays torch-model-free: it supplies the real seed frame the
    sampler will start from, the REAL fresh target frame (never a model
    output), and the nominal level at which the trajectory state will be
    harvested; ``burst_diffusion.rollout`` turns seeds into pseudo-average
    inputs. Seed and target are crops of the SAME window, and the target
    replica differs from the seed replica -- the seed's noise is inside the
    pseudo-average, so a seed target would re-create the Theorem 2 degeneracy.
    """

    seed: torch.Tensor  # [R, C, S, S] float32 in [-1, 1]: real frame, level T
    target: torch.Tensor  # [R, C, S, S] float32 in [-1, 1]: real fresh frame
    stop_level: torch.Tensor  # [R] int64 in {1..T-1}: nominal harvest level


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
        test_fraction: float = 0.0,
        split_seed: int = 2019,
    ):
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {channels}")
        if min_replicas < 1:
            raise ValueError(f"min_replicas must be >= 1, got {min_replicas}")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
        if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction + test_fraction must be < 1, got "
                f"{val_fraction} + {test_fraction}"
            )
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

        # Split by CONTENT GROUP, never by source index: image corpora ship the
        # same picture under several filenames, and splitting indices then puts
        # byte-identical content on both sides (fresh noise, seen scene), which
        # silently invalidates every held-out number. Sources whose clean images
        # hash equal move together. With all-distinct content each group is a
        # singleton and this reduces exactly to the plain index permutation.
        groups: dict[str, list[int]] = {}
        for position, source in enumerate(kept):
            groups.setdefault(content_key(source.clean), []).append(position)
        group_members = list(groups.values())
        self.duplicate_groups = [
            [kept[position].source_index for position in members]
            for members in group_members
            if len(members) > 1
        ]
        if self.duplicate_groups:
            warnings.warn(
                f"{len(self.duplicate_groups)} duplicated clean image(s) among "
                f"{len(kept)} sources (e.g. {self.duplicate_groups[:3]}): duplicates are "
                "kept together on one side of the split, so the validation split holds "
                "no content seen in training. Prefer regenerating the dataset with "
                "content-deduplicated sources.",
                stacklevel=2,
            )

        # Groups are consumed from the END of the permutation: test first, then
        # val. Anchoring test at the end keeps it FIXED when val_fraction
        # changes, so a locked test split stays untouched across development;
        # with test_fraction=0 val consumes the tail, exactly as before.
        order = list(reversed(np.random.RandomState(split_seed).permutation(len(group_members)).tolist()))
        if val_fraction > 0.0 and len(group_members) < 2:
            warnings.warn(
                "only one usable content group; keeping it for training and leaving "
                "the validation split empty",
                stacklevel=2,
            )
        cursor = 0
        holdouts: dict[str, set[int]] = {}
        for name, fraction in (("test", test_fraction), ("val", val_fraction)):
            positions: set[int] = set()
            if fraction > 0.0 and len(group_members) > 1:
                target = max(1, round(fraction * len(kept)))
                while cursor < len(order) and len(positions) < target:
                    members = group_members[order[cursor]]
                    taken = sum(len(g) for g in holdouts.values()) + len(positions)
                    if taken + len(members) > len(kept) - 1:
                        break  # never leave the training split empty
                    positions.update(members)
                    cursor += 1
            holdouts[name] = positions
        test_positions, val_positions = holdouts["test"], holdouts["val"]
        held = test_positions | val_positions
        self.train_sources = [kept[i] for i in range(len(kept)) if i not in held]
        self.val_sources = [kept[i] for i in range(len(kept)) if i in val_positions]
        self.test_sources = [kept[i] for i in range(len(kept)) if i in test_positions]

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

    @property
    def all_sources(self) -> list[BurstSource]:
        return self.train_sources + self.val_sources + self.test_sources

    def sources_for_split(self, split: str) -> list[BurstSource]:
        """Sources of ``'train'``, ``'val'``, or ``'test'``."""
        try:
            return {
                "train": self.train_sources,
                "val": self.val_sources,
                "test": self.test_sources,
            }[split]
        except KeyError:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got {split!r}"
            ) from None

    def summary(self) -> dict:
        ram_bytes = sum(
            source.clean.nbytes + sum(frame.nbytes for frame in source.frames)
            for source in self.all_sources
        )
        return {
            "burst_dir": str(self.burst_dir),
            "train_sources": len(self.train_sources),
            "val_sources": len(self.val_sources),
            "test_sources": len(self.test_sources),
            "val_source_indices": [source.source_index for source in self.val_sources],
            "test_source_indices": [source.source_index for source in self.test_sources],
            "dropped_replicas": list(self.dropped_replicas),
            "dropped_size": list(self.dropped_size),
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "min_frames": min(len(source.frames) for source in self.all_sources),
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
        for source in cache.all_sources:
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

    def _sample_t_values(self, count: int) -> np.ndarray:
        if not self.antithetic:
            return self._rng.integers(1, self.num_steps + 1, size=count)
        half = math.ceil(count / 2)
        drawn = self._rng.integers(1, self.num_steps + 1, size=half)
        mirrored = self.num_steps + 1 - drawn
        return np.concatenate([drawn, mirrored])[:count]

    def sample_batch(
        self, *, count: int | None = None, return_info: bool = False
    ) -> TrainingBatch | tuple[TrainingBatch, list[SampleInfo]]:
        count = self.batch_size if count is None else count
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        t_values = self._sample_t_values(count)
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

    def _sample_stop_levels(self, count: int) -> np.ndarray:
        """Nominal harvest levels, uniform over {1..T-1} (the inference sampler
        visits each level exactly once per run; T itself is the raw real frame,
        already covered by ordinary training). Antithetic pairing mirrors
        within the same range: ``u <-> T - u``."""
        if not self.antithetic:
            return self._rng.integers(1, self.num_steps, size=count)
        half = math.ceil(count / 2)
        drawn = self._rng.integers(1, self.num_steps, size=half)
        mirrored = self.num_steps - drawn
        return np.concatenate([drawn, mirrored])[:count]

    def rollout_pair_batch(
        self, *, count: int, return_info: bool = False
    ) -> RolloutPairBatch | tuple[RolloutPairBatch, list[RolloutPairInfo]]:
        """Seed frames + REAL fresh targets + harvest levels for self-rollout.

        Model outputs never appear here: the target side of every pair is a
        real frame drawn from the cache, from a replica different from the
        seed's (see :class:`RolloutPairBatch`).
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if self.num_steps < 2:
            raise ValueError(
                f"rollout pairs need num_steps >= 2, got {self.num_steps}: with T = 1 "
                "the sampler has no intermediate pseudo-average states"
            )
        stop_levels = self._sample_stop_levels(count)
        size = self.image_size
        seed_list: list[np.ndarray] = []
        target_list: list[np.ndarray] = []
        info: list[RolloutPairInfo] = []
        for stop_level in stop_levels:
            source = self.cache.train_sources[
                int(self._rng.integers(len(self.cache.train_sources)))
            ]
            height, width = source.clean.shape[:2]
            top = int(self._rng.integers(0, height - size + 1))
            left = int(self._rng.integers(0, width - size + 1))
            permutation = self._rng.permutation(len(source.frames))
            seed_replica = int(permutation[0])
            target_replica = int(permutation[1])

            window = np.s_[top : top + size, left : left + size]
            seed_list.append(_to_chw(_to_model_range(source.frames[seed_replica][window])))
            target_list.append(_to_chw(_to_model_range(source.frames[target_replica][window])))
            if return_info:
                info.append(
                    RolloutPairInfo(
                        source_index=source.source_index,
                        crop_yx=(top, left),
                        seed_replica=seed_replica,
                        target_replica=target_replica,
                        stop_level=int(stop_level),
                    )
                )
        batch = RolloutPairBatch(
            seed=torch.from_numpy(np.stack(seed_list)),
            target=torch.from_numpy(np.stack(target_list)),
            stop_level=torch.as_tensor(stop_levels, dtype=torch.int64),
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
