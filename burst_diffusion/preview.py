"""Preview grids for burst datasets: the visual "noisy enough" check.

Renders, for one source, the clean image next to individual noisy frames
(top row) and running averages over growing frame counts (bottom row), each
captioned with its PSNR vs clean. The averaging row should visibly converge
toward the clean image -- the premise the whole method rests on. Pure PIL;
no matplotlib dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import resolve_burst_dir
from .metrics import psnr

_CAPTION_HEIGHT = 16
_PAD = 4


def _load_grayscale_or_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float64) / 255.0


def _center_crop(array: np.ndarray, max_tile: int) -> np.ndarray:
    height, width = array.shape[:2]
    if height <= max_tile and width <= max_tile:
        return array
    top = max(0, (height - max_tile) // 2)
    left = max(0, (width - max_tile) // 2)
    return array[top : top + min(height, max_tile), left : left + min(width, max_tile)]


def _to_tile(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    return Image.fromarray(np.rint(clipped * 255.0).astype(np.uint8))


def make_preview_grid(
    dataset_dir: str | Path,
    *,
    source_index: int = 0,
    out_path: str | Path,
    avg_counts: tuple[int, ...] = (1, 2, 4, 8, 16),
    max_tile: int = 256,
) -> Path:
    """Render the clean/frames/averages grid for one source; returns ``out_path``."""
    burst_dir = resolve_burst_dir(dataset_dir)
    clean_path = burst_dir / "clean" / f"{source_index:05d}.png"
    if not clean_path.is_file():
        raise FileNotFoundError(f"no clean image for source {source_index}: {clean_path}")
    replica_paths = sorted((burst_dir / "noisy").glob(f"{source_index:05d}_*.png"))
    if not replica_paths:
        raise FileNotFoundError(f"no noisy replicas for source {source_index} under {burst_dir}")

    counts: list[int] = []
    for count in avg_counts:
        clamped = min(count, len(replica_paths))
        if clamped >= 1 and clamped not in counts:
            counts.append(clamped)

    clean = _center_crop(_load_grayscale_or_rgb(clean_path), max_tile)
    crop_shape = clean.shape

    singles: list[np.ndarray] = []
    running_sum = np.zeros_like(clean)
    averages: dict[int, np.ndarray] = {}
    for number, path in enumerate(replica_paths[: max(counts)], start=1):
        frame = _center_crop(_load_grayscale_or_rgb(path), max_tile)
        if frame.shape != crop_shape:
            raise ValueError(f"replica shape {frame.shape} != clean shape {crop_shape}: {path}")
        if len(singles) < len(counts):
            singles.append(frame)
        running_sum += frame
        if number in counts:
            averages[number] = running_sum / number

    top_row = [("clean", clean)] + [
        (f"frame {index} | {psnr(clean, frame):.1f} dB", frame)
        for index, frame in enumerate(singles)
    ]
    bottom_row = [("", None)] + [
        (f"avg n={count} | {psnr(clean, averages[count]):.1f} dB", averages[count])
        for count in counts
    ]

    tile_height, tile_width = crop_shape[0], crop_shape[1]
    columns = max(len(top_row), len(bottom_row))
    cell_width = tile_width + _PAD
    cell_height = tile_height + _CAPTION_HEIGHT + _PAD
    canvas = Image.new(
        "RGB", (columns * cell_width + _PAD, 2 * cell_height + _PAD), color=(24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate((top_row, bottom_row)):
        for column_index, (caption, array) in enumerate(row):
            x = _PAD + column_index * cell_width
            y = _PAD + row_index * cell_height
            if array is not None:
                canvas.paste(_to_tile(array).convert("RGB"), (x, y))
            if caption:
                draw.text((x, y + tile_height + 2), caption, fill=(230, 230, 230), font=font)

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
    return destination
