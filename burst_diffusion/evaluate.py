"""Evaluation: model outputs vs the classical baselines, per validation image.

Five methods are measured against the clean reference (PSNR + SSIM):

- ``single_frame``: burst frame 0 as-is -- the measurement everything starts from.
- ``avg_of_n``: plain average of ALL burst frames. A *reference baseline*, not
  a ceiling: a learned prior can legitimately beat N-frame averaging (the
  standard Noise2Noise result). Beating it from ONE frame is the headline.
- ``one_shot``: the network prediction at t=T from frame 0 (single forward pass).
- ``iter_average`` / ``iter_prediction``: the two outputs of the full iterative
  sampler started from frame 0.

Default geometry is a deterministic center crop at the training resolution;
``tile=True`` evaluates full images through overlapping tiles blended by
uniform averaging (model calls only see training-resolution crops).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .data import BurstCache, BurstSource
from .metrics import psnr, ssim
from .sample import Sampler
from .schedule import min_replicas, sampling_schedule

METHOD_NAMES = ("single_frame", "avg_of_n", "one_shot", "iter_average", "iter_prediction")
_TILE_OVERLAP = 32


def _to_hwc01(array: np.ndarray) -> np.ndarray:
    """uint8 [H, W] or [H, W, C] -> float64 [H, W, C] in [0, 1]."""
    image = array.astype(np.float64) / 255.0
    if image.ndim == 2:
        image = image[:, :, None]
    return image


def _to_model_tensor(image01: np.ndarray) -> torch.Tensor:
    chw = np.ascontiguousarray(image01.transpose(2, 0, 1)).astype(np.float32)
    return torch.from_numpy(chw * 2.0 - 1.0)[None]


def _from_model_tensor(tensor: torch.Tensor) -> np.ndarray:
    array = ((tensor[0].clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
    return np.ascontiguousarray(array.transpose(1, 2, 0)).astype(np.float64)


def _tile_positions(extent: int, size: int, stride: int) -> list[int]:
    positions = list(range(0, extent - size + 1, stride))
    if positions[-1] != extent - size:
        positions.append(extent - size)
    return positions


def _run_model_outputs(
    sampler: Sampler, measurement01: np.ndarray, schedule: list[int]
) -> dict[str, np.ndarray]:
    x_start = _to_model_tensor(measurement01)
    one_shot = sampler.run(x_start, schedule=[sampler.num_steps])
    full = sampler.run(x_start, schedule=schedule)
    return {
        "one_shot": _from_model_tensor(one_shot.prediction),
        "iter_average": _from_model_tensor(full.average),
        "iter_prediction": _from_model_tensor(full.prediction),
    }


def _tiled_model_outputs(
    sampler: Sampler, measurement01: np.ndarray, schedule: list[int], size: int
) -> dict[str, np.ndarray]:
    height, width = measurement01.shape[:2]
    stride = max(1, size - min(_TILE_OVERLAP, size // 2))
    accumulators = {name: np.zeros_like(measurement01) for name in
                    ("one_shot", "iter_average", "iter_prediction")}
    weights = np.zeros((height, width, 1), dtype=np.float64)
    for top in _tile_positions(height, size, stride):
        for left in _tile_positions(width, size, stride):
            window = np.s_[top : top + size, left : left + size]
            outputs = _run_model_outputs(sampler, measurement01[window], schedule)
            for name, tile in outputs.items():
                accumulators[name][window] += tile
            weights[window] += 1.0
    return {name: accumulator / weights for name, accumulator in accumulators.items()}


def _evaluate_source(
    source: BurstSource,
    sampler: Sampler,
    *,
    image_size: int,
    schedule: list[int],
    tile: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if tile:
        clean01 = _to_hwc01(source.clean)
        measurement01 = _to_hwc01(source.frames[0])
        frames01 = [_to_hwc01(frame) for frame in source.frames]
    else:
        height, width = source.clean.shape[:2]
        top = (height - image_size) // 2
        left = (width - image_size) // 2
        window = np.s_[top : top + image_size, left : left + image_size]
        clean01 = _to_hwc01(source.clean[window])
        measurement01 = _to_hwc01(source.frames[0][window])
        frames01 = [_to_hwc01(frame[window]) for frame in source.frames]

    outputs = {
        "single_frame": measurement01,
        "avg_of_n": np.mean(frames01, axis=0),
    }
    if tile:
        outputs.update(_tiled_model_outputs(sampler, measurement01, schedule, image_size))
    else:
        outputs.update(_run_model_outputs(sampler, measurement01, schedule))
    return outputs, clean01


def _write_comparison_grid(
    rows: list[dict[str, np.ndarray]],
    cleans: list[np.ndarray],
    scores: list[dict[str, float]],
    path: Path,
    *,
    max_tile: int = 192,
) -> None:
    caption_height, pad = 16, 4
    font = ImageFont.load_default()
    columns = list(METHOD_NAMES) + ["clean"]

    def crop(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        top = max(0, (height - max_tile) // 2)
        left = max(0, (width - max_tile) // 2)
        cropped = image[top : top + min(height, max_tile), left : left + min(width, max_tile)]
        # Small tiles get integer nearest-neighbor upscaling so pixels stay
        # faithful and the caption text fits under each column.
        scale = max(1, -(-128 // min(cropped.shape[:2])))
        if scale > 1:
            cropped = cropped.repeat(scale, axis=0).repeat(scale, axis=1)
        return cropped

    tiles = []
    for row_index, row in enumerate(rows):
        row_tiles = []
        for name in columns:
            image = crop(cleans[row_index] if name == "clean" else row[name])
            caption = name if name == "clean" else f"{name} | {scores[row_index][name]:.1f} dB"
            row_tiles.append((caption, image))
        tiles.append(row_tiles)

    tile_height, tile_width = tiles[0][0][1].shape[:2]
    cell_width, cell_height = tile_width + pad, tile_height + caption_height + pad
    canvas = Image.new(
        "RGB",
        (len(columns) * cell_width + pad, len(tiles) * cell_height + pad),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row_tiles in enumerate(tiles):
        for column_index, (caption, image) in enumerate(row_tiles):
            x = pad + column_index * cell_width
            y = pad + row_index * cell_height
            array = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            tile_image = Image.fromarray(array[:, :, 0] if array.shape[2] == 1 else array)
            canvas.paste(tile_image.convert("RGB"), (x, y))
            draw.text((x, y + tile_height + 2), caption, fill=(230, 230, 230), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def evaluate(
    config: Config,
    checkpoint: str | Path,
    *,
    split: str = "val",
    limit: int | None = None,
    out_dir: str | Path,
    sample_steps: int | None = None,
    tile: bool = False,
    device: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run the five-baseline evaluation; writes results.json + a comparison grid."""
    if split not in ("val", "train"):
        raise ValueError(f"split must be 'val' or 'train', got {split!r}")
    sampler = Sampler.from_checkpoint(
        checkpoint, device=device if device is not None else config.training.device
    )
    if sampler.num_steps != config.schedule.num_steps:
        raise ValueError(
            f"checkpoint was trained with num_steps={sampler.num_steps} but the config "
            f"says {config.schedule.num_steps}"
        )
    cache = BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=min_replicas(config.schedule.num_steps),
        min_size=config.data.image_size,
        val_fraction=config.data.val_fraction,
        split_seed=config.data.split_seed,
    )
    sources = cache.val_sources if split == "val" else cache.train_sources
    if not sources:
        raise ValueError(f"no sources in the {split!r} split")
    if limit is not None:
        sources = sources[:limit]
    schedule = sampling_schedule(
        config.schedule.num_steps,
        sample_steps if sample_steps is not None else config.sampling.num_sample_steps,
    )

    per_image: dict[str, dict[str, list[float]]] = {
        name: {"psnr": [], "ssim": []} for name in METHOD_NAMES
    }
    grid_rows: list[dict[str, np.ndarray]] = []
    grid_cleans: list[np.ndarray] = []
    grid_scores: list[dict[str, float]] = []
    for index, source in enumerate(sources):
        outputs, clean01 = _evaluate_source(
            source, sampler,
            image_size=config.data.image_size, schedule=schedule, tile=tile,
        )
        scores: dict[str, float] = {}
        for name in METHOD_NAMES:
            scores[name] = psnr(clean01, outputs[name])
            per_image[name]["psnr"].append(scores[name])
            per_image[name]["ssim"].append(ssim(clean01, outputs[name]))
        if len(grid_rows) < 4:
            grid_rows.append(outputs)
            grid_cleans.append(clean01)
            grid_scores.append(scores)
        if progress_callback is not None:
            progress_callback(index + 1, len(sources))

    results = {
        "checkpoint": str(checkpoint),
        "dataset_dir": str(config.data.dataset_dir),
        "split": split,
        "count": len(sources),
        "source_indices": [source.source_index for source in sources],
        "sample_steps": len(schedule),
        "tile": tile,
        "methods": {
            name: {
                "psnr_mean": float(np.mean(values["psnr"])),
                "ssim_mean": float(np.mean(values["ssim"])),
                "psnr_per_image": values["psnr"],
                "ssim_per_image": values["ssim"],
            }
            for name, values in per_image.items()
        },
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_comparison_grid(grid_rows, grid_cleans, grid_scores, destination / "comparison_grid.png")
    return results
