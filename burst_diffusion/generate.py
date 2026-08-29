"""Synthetic burst-dataset generation: stage clean sources, add noise, verify.

Wraps ``noising_pipeline.create_noisy_dataset`` (the sanctioned data tool) with
the preprocessing and verification the burst-diffusion method needs:

- Clean sources are grayscaled and downscaled BEFORE noising (converting or
  resizing *noisy* frames would partially denoise them and correlate the
  noise), then pre-scaled into ``[margin, 1 - margin]`` so the pipeline's
  [0, 1] clipping rarely bites (clipped noise is biased and averaging cannot
  remove that bias).
- After generation, per-source statistics are measured and written to
  ``stats.json``: single-frame PSNR (the "noisy enough" check, target roughly
  10-18 dB), avg-of-N PSNR vs clean, and the residual bias of the N-frame
  average. Out-of-band values raise warnings rather than failing.

Output layout under ``output_dir``::

    _sources/00000.png   staged prepped clean images (margin applied)
    sources.json         provenance: staged file -> original path + settings
    burst/               noising_pipeline output (clean/, noisy/, manifest.jsonl)
    stats.json           measured noise statistics + warnings
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from noising_pipeline import create_noisy_dataset

from .metrics import psnr

SOURCE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SINGLE_FRAME_PSNR_BAND = (10.0, 18.0)
MIN_AVERAGE_PSNR = 26.0
MAX_ABS_BIAS = 0.01
# Per-step defaults mirrored from noising_pipeline (its table is private).
_NOISE_DEFAULTS: dict[str, dict[str, float]] = {
    "gaussian": {"mean": 0.0, "std": 0.01},
    "poisson": {"peak": 10000.0},
    "salt_pepper": {"amount": 0.001, "salt_ratio": 0.5},
}


def _find_source_images(source_dir: Path) -> list[Path]:
    files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    ]
    files.sort(key=lambda p: (p.relative_to(source_dir).as_posix().casefold(),
                              p.relative_to(source_dir).as_posix()))
    if not files:
        raise FileNotFoundError(f"no images with suffixes {SOURCE_SUFFIXES} under {source_dir}")
    return files


def select_sources(source_dir: str | Path, num_sources: int, seed: int) -> list[Path]:
    """Seeded, order-stable subset of the images under ``source_dir``."""
    if num_sources < 1:
        raise ValueError(f"num_sources must be >= 1, got {num_sources}")
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"source_dir is not a directory: {root}")
    files = _find_source_images(root)
    if num_sources > len(files):
        warnings.warn(
            f"requested {num_sources} sources but only {len(files)} images exist "
            f"under {root}; using all of them",
            stacklevel=2,
        )
        num_sources = len(files)
    order = np.random.RandomState(seed).permutation(len(files))[:num_sources]
    selected = [files[index] for index in sorted(order)]
    return selected


def _load_clean_array(path: Path, *, grayscale: bool) -> np.ndarray:
    """Decode any PIL-openable image to uint8 [H, W] (grayscale) or [H, W, 3]."""
    with Image.open(path) as image:
        mode = image.mode
        if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
            array = np.asarray(image, dtype=np.float64)
            scale = 65535.0 if array.max() > 255 else 255.0
            array = np.clip(array / scale * 255.0, 0.0, 255.0)
            gray = np.rint(array).astype(np.uint8)
            if grayscale:
                return gray
            return np.stack([gray] * 3, axis=-1)
        converted = image.convert("L" if grayscale else "RGB")
        return np.asarray(converted).copy()


def _stage_clean_image(
    source: Path, destination: Path, *, grayscale: bool, max_side: int | None, margin: float
) -> tuple[int, int]:
    array = _load_clean_array(source, grayscale=grayscale)
    height, width = array.shape[:2]
    if max_side is not None and max(height, width) > max_side:
        scale = max_side / max(height, width)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = Image.fromarray(array).resize(new_size, resample=Image.LANCZOS)
        array = np.asarray(image).copy()
    normalized = array.astype(np.float64) / 255.0
    prescaled = margin + (1.0 - 2.0 * margin) * normalized
    staged = np.rint(prescaled * 255.0).astype(np.uint8)
    Image.fromarray(staged).save(destination, format="PNG")
    return staged.shape[0], staged.shape[1]


def _effective_noise_params(
    noise_types: Sequence[str], noise_params: Mapping[str, Mapping[str, float]] | None, steps: int
) -> dict[str, dict[str, float]]:
    effective: dict[str, dict[str, float]] = {}
    for name in dict.fromkeys(noise_types):
        params = dict(_NOISE_DEFAULTS.get(name, {}))
        if noise_params and name in noise_params:
            params.update(noise_params[name])
        if name == "gaussian":
            params = {"mean": params["mean"] * steps, "std": params["std"] * math.sqrt(steps)}
        elif name == "poisson":
            params = {"peak": params["peak"] / steps}
        elif name == "salt_pepper":
            params = {
                "amount": 1.0 - (1.0 - params["amount"]) ** steps,
                "salt_ratio": params["salt_ratio"],
            }
        effective[name] = params
    return effective


def _guard_output_dir(output_dir: Path, source_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_source = source_dir.resolve()
    if resolved_output == resolved_source or resolved_source in resolved_output.parents:
        raise ValueError(f"output_dir {output_dir} must not live inside source_dir {source_dir}")
    if resolved_output in resolved_source.parents:
        raise ValueError(f"source_dir {source_dir} must not live inside output_dir {output_dir}")
    home = Path.home().resolve()
    if resolved_output in (home, Path(resolved_output.anchor)):
        raise ValueError(f"refusing to use protected directory as output_dir: {output_dir}")
    cwd = Path.cwd().resolve()
    if resolved_output == cwd or resolved_output in cwd.parents:
        raise ValueError(f"refusing to overwrite the working directory or an ancestor: {output_dir}")


def _measure_stats(
    burst_dir: Path,
    *,
    num_sources: int,
    replicas: int,
) -> tuple[list[dict], dict]:
    per_source: list[dict] = []
    single_psnrs_all: list[float] = []
    average_psnrs: list[float] = []
    biases: list[float] = []
    for source_index in range(num_sources):
        clean_path = burst_dir / "clean" / f"{source_index:05d}.png"
        with Image.open(clean_path) as image:
            clean = np.asarray(image, dtype=np.float64) / 255.0
        running_sum = np.zeros_like(clean)
        single_psnrs: list[float] = []
        count = 0
        for replica_index in range(replicas):
            noisy_path = burst_dir / "noisy" / f"{source_index:05d}_{replica_index:05d}.png"
            with Image.open(noisy_path) as image:
                noisy = np.asarray(image, dtype=np.float64) / 255.0
            running_sum += noisy
            single_psnrs.append(psnr(clean, noisy))
            count += 1
        average = running_sum / count
        average_psnr = psnr(clean, average)
        bias = float(average.mean() - clean.mean())
        per_source.append(
            {
                "source_index": source_index,
                "single_frame_psnr_mean": float(np.mean(single_psnrs)),
                "single_frame_psnr_min": float(np.min(single_psnrs)),
                "avg_of_n_psnr": average_psnr,
                "bias": bias,
            }
        )
        single_psnrs_all.extend(single_psnrs)
        average_psnrs.append(average_psnr)
        biases.append(bias)
    aggregate = {
        "single_frame_psnr_median": float(np.median(single_psnrs_all)),
        "single_frame_psnr_p10": float(np.percentile(single_psnrs_all, 10)),
        "single_frame_psnr_p90": float(np.percentile(single_psnrs_all, 90)),
        "avg_of_n_psnr_median": float(np.median(average_psnrs)),
        "bias_abs_median": float(np.median(np.abs(biases))),
        "bias_mean": float(np.mean(biases)),
    }
    return per_source, aggregate


def _collect_warnings(
    aggregate: dict,
    *,
    noise_types: Sequence[str],
    effective_params: Mapping[str, Mapping[str, float]],
    margin: float,
) -> list[str]:
    messages: list[str] = []
    low, high = SINGLE_FRAME_PSNR_BAND
    median_single = aggregate["single_frame_psnr_median"]
    if median_single > high:
        messages.append(
            f"median single-frame PSNR {median_single:.1f} dB is above {high:.0f} dB: "
            "the frames may be too clean for a convincing denoising benchmark; "
            "increase the noise (lower poisson peak / raise gaussian std / raise steps)"
        )
    elif median_single < low:
        messages.append(
            f"median single-frame PSNR {median_single:.1f} dB is below {low:.0f} dB: "
            "extremely noisy; expect slow convergence"
        )
    if aggregate["avg_of_n_psnr_median"] < MIN_AVERAGE_PSNR:
        messages.append(
            f"median avg-of-N PSNR {aggregate['avg_of_n_psnr_median']:.1f} dB is below "
            f"{MIN_AVERAGE_PSNR:.0f} dB: averaging does not recover the clean image well "
            "(clip bias or too few replicas); raise margin, lower the noise, or add replicas"
        )
    if aggregate["bias_abs_median"] > MAX_ABS_BIAS:
        messages.append(
            f"median |bias| {aggregate['bias_abs_median']:.4f} exceeds {MAX_ABS_BIAS}: "
            "the N-frame average is systematically shifted from clean (clipped noise); "
            "raise margin or reduce the noise strength"
        )
    if "salt_pepper" in noise_types:
        messages.append(
            "salt_pepper noise is not zero-mean: E[noisy] = (1-amount)*clean + "
            "amount*salt_ratio, so the MSE/Noise2Noise training objective converges to a "
            "biased limit; prefer gaussian/poisson for quantitative runs"
        )
    gaussian = effective_params.get("gaussian")
    if gaussian is not None and gaussian["std"] > margin / 2.0:
        messages.append(
            f"effective gaussian std {gaussian['std']:.3f} exceeds margin/2 = {margin / 2:.3f}: "
            "clipping at 0/1 will bias the noise; raise margin or lower std"
        )
    return messages


def generate_burst_dataset(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    num_sources: int,
    replicas: int,
    noise_type: str | Sequence[str],
    steps: int = 1,
    noise_params: Mapping[str, Mapping[str, float]] | None = None,
    margin: float = 0.15,
    grayscale: bool = True,
    max_side: int | None = 512,
    select_seed: int = 0,
    noise_seed: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Generate a verified synthetic burst dataset; returns the stats.json path."""
    if replicas < 1:
        raise ValueError(f"replicas must be >= 1, got {replicas}")
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin}")
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    _guard_output_dir(output_root, source_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output_dir already exists: {output_root} (pass overwrite=True to replace it)"
            )
        shutil.rmtree(output_root)

    selected = select_sources(source_root, num_sources, select_seed)
    staging_dir = output_root / "_sources"
    staging_dir.mkdir(parents=True)
    records = []
    for index, source in enumerate(selected):
        staged_name = f"{index:05d}.png"
        height, width = _stage_clean_image(
            source, staging_dir / staged_name,
            grayscale=grayscale, max_side=max_side, margin=margin,
        )
        records.append(
            {
                "staged": staged_name,
                "original": source.relative_to(source_root).as_posix(),
                "staged_size": [height, width],
            }
        )

    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    settings = {
        "source_dir": str(source_root),
        "num_sources": len(selected),
        "replicas": replicas,
        "noise_type": noise_types,
        "steps": steps,
        "noise_params": {k: dict(v) for k, v in noise_params.items()} if noise_params else None,
        "margin": margin,
        "grayscale": grayscale,
        "max_side": max_side,
        "select_seed": select_seed,
        "noise_seed": noise_seed,
    }
    (output_root / "sources.json").write_text(
        json.dumps({"settings": settings, "sources": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    create_noisy_dataset(
        source_dir=staging_dir,
        output_dir=output_root / "burst",
        n=replicas,
        steps=steps,
        noise_type=noise_type,
        noise_params=noise_params,
        seed=noise_seed,
        overwrite=True,
        progress=progress,
    )

    per_source, aggregate = _measure_stats(
        output_root / "burst", num_sources=len(selected), replicas=replicas
    )
    effective_params = _effective_noise_params(noise_types, noise_params, steps)
    messages = _collect_warnings(
        aggregate, noise_types=noise_types, effective_params=effective_params, margin=margin
    )
    for message in messages:
        warnings.warn(message, stacklevel=2)

    stats = {
        "settings": settings,
        "effective_noise_params": effective_params,
        "aggregate": aggregate,
        "per_source": per_source,
        "warnings": messages,
    }
    stats_path = output_root / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats_path
