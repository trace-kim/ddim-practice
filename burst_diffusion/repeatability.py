"""Repeatability & metrology evaluation: the *precision* of each denoising method.

PSNR/SSIM (``evaluate``) measure accuracy against clean; metrology cares about
repeatability: acquire the same area again and ask how much the output -- and
the dimensional measurements extracted from it -- moves. The sampler is
deterministic, so feeding it K different real burst frames ("seeds") of the
same source isolates exactly that: seed-to-seed variation is the measurement
noise each method transmits.

Methods (per source, deterministic center crop at the training resolution):

- ``single_frame``: seed frame j as-is (j = 0..K-1) -- the raw measurement.
- ``avg_of_{m}``: disjoint m-frame averages -- ``N // m`` independent
  realizations each, the classical precision-vs-throughput ladder.
  ``avg_of_N`` has a single realization, so it contributes accuracy and bias
  but no precision estimate.
- ``one_shot@{arm}`` / ``iter_average@{arm}`` / ``iter_prediction@{arm}``:
  model outputs started from seed frame j, one set per checkpoint arm.

Reported per method:

- pixel repeatability: per-pixel sample std across realizations (c4-debiased
  so different realization counts stay comparable);
- accuracy: PSNR/SSIM vs clean, mean over realizations;
- CD (critical dimension): edge pairs measured by the standard CD-SEM recipe
  (band-averaged profile, 50% threshold between robust profile extremes,
  subpixel linear-interpolated crossings) on fixed sites selected from the
  clean image; pooled sigma across (source, site), bias vs the clean-image
  CD, and measurement success rate;
- registration: feature-center precision from the same sites (site-level
  placement) plus global sub-pixel image shift vs clean via upsampled
  cross-correlation (Guizar-Sicairos style), precision and bias.

Pooling follows metrology convention: per-site/per-source means are removed,
sums of squares and degrees of freedom accumulate across sites and sources,
and the pooled sigma is Bessel + c4 corrected. numpy/PIL/torch only (the repo
deliberately has no scipy/skimage dependency).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .data import BurstCache, BurstSource
from .evaluate import _from_model_tensor, _to_hwc01, _to_model_tensor
from .metrics import psnr, ssim
from .sample import Sampler
from .schedule import min_replicas, sampling_schedule

MODEL_METHOD_NAMES = ("one_shot", "iter_average", "iter_prediction")
_SIGMA_MAP_FULL_SCALE = 0.2  # sigma (in [0,1] intensity) rendered as white
_PROFILE_PERCENTILES = (10.0, 90.0)  # robust lo/hi for the 50% threshold


# ---------------------------------------------------------------------------
# statistics helpers


def _c4(n: int) -> float:
    """E[sample std] / true sigma for a normal sample of size ``n`` (>= 2).

    Dividing a sample std by c4 debiases it, which matters when comparing
    methods whose realization counts differ (c4(2) ~ 0.798: a two-realization
    std underestimates sigma by ~20%).
    """
    if n < 2:
        raise ValueError(f"c4 needs a sample size >= 2, got {n}")
    return math.sqrt(2.0 / (n - 1)) * math.exp(
        math.lgamma(n / 2.0) - math.lgamma((n - 1) / 2.0)
    )


def _pooled_sigma(sum_squares: float, dof: int) -> float | None:
    """c4-debiased pooled sigma from accumulated within-group sums of squares."""
    if dof < 1:
        return None
    return math.sqrt(sum_squares / dof) / _c4(dof + 1)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) > 0 else None


def _median_or_none(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) > 0 else None


# ---------------------------------------------------------------------------
# global registration: sub-pixel translation via upsampled cross-correlation


def estimate_shift(
    reference: np.ndarray, moved: np.ndarray, *, upsample: int = 50
) -> tuple[float, float]:
    """Sub-pixel translation ``(dy, dx)`` such that ``moved ~ reference shifted by (dy, dx)``.

    Windowed FFT cross-correlation with a matrix-multiply DFT refinement of the
    peak on a ``1/upsample``-pixel grid (Guizar-Sicairos et al., 2008). Both
    inputs are 2D float arrays of the same shape.
    """
    if reference.shape != moved.shape or reference.ndim != 2:
        raise ValueError(
            f"expected two 2D arrays of equal shape, got {reference.shape} and {moved.shape}"
        )
    if upsample < 1:
        raise ValueError(f"upsample must be >= 1, got {upsample}")
    height, width = reference.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum_ref = np.fft.fft2((reference - reference.mean()) * window)
    spectrum_mov = np.fft.fft2((moved - moved.mean()) * window)
    cross = spectrum_ref * np.conj(spectrum_mov)

    correlation = np.fft.ifft2(cross)
    peak_flat = int(np.argmax(np.abs(correlation)))
    peak_y, peak_x = np.unravel_index(peak_flat, correlation.shape)
    # The correlation peak sits at MINUS the shift; wrap to signed coordinates.
    peak_y = peak_y - height if peak_y > height // 2 else peak_y
    peak_x = peak_x - width if peak_x > width // 2 else peak_x

    # Refine on a +-0.75 px neighborhood with an explicit DFT evaluated only
    # there (cheap: [n, H] @ [H, W] @ [W, n] with n ~ 1.5 * upsample).
    offsets = np.linspace(-0.75, 0.75, int(round(1.5 * upsample)) + 1)
    grid_y = peak_y + offsets
    grid_x = peak_x + offsets
    freq_y = np.fft.fftfreq(height)
    freq_x = np.fft.fftfreq(width)
    dft_rows = np.exp(2j * np.pi * np.outer(grid_y, freq_y))
    dft_cols = np.exp(2j * np.pi * np.outer(freq_x, grid_x))
    fine = np.abs(dft_rows @ cross @ dft_cols)
    fine_flat = int(np.argmax(fine))
    fine_y, fine_x = np.unravel_index(fine_flat, fine.shape)
    return -float(grid_y[fine_y]), -float(grid_x[fine_x])


# ---------------------------------------------------------------------------
# CD (critical dimension) measurement: threshold-crossing profile metrology


@dataclass(frozen=True)
class CDSite:
    """One fixed measurement site, selected on the clean image.

    ``orientation`` "h" measures across columns (a vertically oriented feature)
    with the band spanning rows ``[band_start, band_stop)``; "v" is the
    transpose. ``left_edge``/``right_edge`` are the clean image's subpixel
    crossing positions; every realization is measured by searching for the
    same-direction crossing nearest to them. ``polarity`` +1 means the feature
    is brighter than the threshold between the edges, -1 darker.
    """

    orientation: str
    band_start: int
    band_stop: int
    left_edge: float
    right_edge: float
    polarity: int
    clean_cd: float
    clean_center: float


def _gray2d(image01: np.ndarray) -> np.ndarray:
    """[H, W, C] (or [H, W]) float -> channel-averaged [H, W] float."""
    return image01.mean(axis=2) if image01.ndim == 3 else image01


def _band_profile(
    image2d: np.ndarray, band_start: int, band_stop: int, smooth: int
) -> np.ndarray:
    """Mean of rows [band_start, band_stop) with light reflect-padded smoothing.

    Band-averaging is the CD-SEM practice of averaging scan lines inside the
    measurement box; the identical smoothing is applied to clean and noisy
    profiles so the estimator (and its small symmetric bias) cancels in
    comparisons.
    """
    profile = image2d[band_start:band_stop].mean(axis=0)
    if smooth > 1:
        half = smooth // 2
        padded = np.pad(profile, half, mode="reflect")
        profile = np.convolve(padded, np.full(smooth, 1.0 / smooth), mode="valid")
    return profile


def _crossings(profile: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel threshold crossings: positions plus rising (below->above) flags."""
    delta = profile - threshold
    indices = np.nonzero(delta[:-1] * delta[1:] < 0.0)[0]
    positions = indices + delta[indices] / (delta[indices] - delta[indices + 1])
    rising = delta[indices] < 0.0
    return positions.astype(np.float64), rising


def measure_site(
    image2d: np.ndarray, site: CDSite, *, tolerance: float, smooth: int
) -> tuple[float, float] | None:
    """Measure ``(cd, center)`` at a fixed site, or ``None`` when an edge is lost.

    The threshold is re-derived per measurement from robust extremes of the
    profile inside the site window (CD-SEM's per-waveform 50% threshold), and
    each edge is the same-direction crossing nearest to the clean position
    within ``tolerance`` pixels.
    """
    if site.orientation == "v":
        image2d = image2d.T
    profile = _band_profile(image2d, site.band_start, site.band_stop, smooth)
    window_lo = max(0, int(math.floor(site.left_edge - tolerance)))
    window_hi = min(len(profile), int(math.ceil(site.right_edge + tolerance)) + 1)
    window = profile[window_lo:window_hi]
    lo, hi = np.percentile(window, _PROFILE_PERCENTILES)
    if hi <= lo:
        return None
    threshold = 0.5 * (lo + hi)
    positions, rising = _crossings(profile, threshold)
    if positions.size == 0:
        return None

    def _nearest(target: float, want_rising: bool) -> float | None:
        candidates = positions[(rising == want_rising) & (np.abs(positions - target) <= tolerance)]
        if candidates.size == 0:
            return None
        return float(candidates[np.argmin(np.abs(candidates - target))])

    bright = site.polarity > 0
    left = _nearest(site.left_edge, want_rising=bright)
    right = _nearest(site.right_edge, want_rising=not bright)
    if left is None or right is None or right <= left:
        return None
    return right - left, 0.5 * (left + right)


def find_cd_sites(
    clean2d: np.ndarray,
    *,
    band_height: int = 16,
    smooth: int = 3,
    tolerance: float = 4.0,
    max_sites: int = 8,
    min_width: float = 3.0,
    min_contrast: float = 0.08,
) -> list[CDSite]:
    """Select up to ``max_sites`` measurable edge pairs from the clean image.

    Non-overlapping bands in both orientations are profiled; in each band the
    highest-contrast threshold-crossing segment with a sane width and enough
    border margin becomes a candidate. Each candidate is validated by running
    :func:`measure_site` on the clean image itself, so ``clean_cd`` /
    ``clean_center`` come from the *identical* estimator applied everywhere.
    """
    scored: list[tuple[float, CDSite]] = []
    for orientation in ("h", "v"):
        image2d = clean2d if orientation == "h" else clean2d.T
        height, length = image2d.shape
        for band_start in range(0, height - band_height + 1, band_height):
            profile = _band_profile(image2d, band_start, band_start + band_height, smooth)
            lo, hi = np.percentile(profile, _PROFILE_PERCENTILES)
            if hi - lo < min_contrast:
                continue
            threshold = 0.5 * (lo + hi)
            positions, rising = _crossings(profile, threshold)
            best: tuple[float, CDSite] | None = None
            for index in range(len(positions) - 1):
                left, right = positions[index], positions[index + 1]
                if rising[index] == rising[index + 1]:  # defensive: must alternate
                    continue
                width = right - left
                if not min_width <= width <= length / 2.0:
                    continue
                if left < tolerance + 1.0 or right > length - tolerance - 2.0:
                    continue
                interior = profile[int(math.ceil(left + 0.5)) : int(math.floor(right - 0.5)) + 1]
                if interior.size == 0:
                    continue
                depth = abs(float(np.median(interior)) - threshold)
                if depth < 0.25 * (hi - lo):
                    continue
                candidate = CDSite(
                    orientation=orientation,
                    band_start=band_start,
                    band_stop=band_start + band_height,
                    left_edge=float(left),
                    right_edge=float(right),
                    polarity=1 if rising[index] else -1,
                    clean_cd=float(width),
                    clean_center=float(0.5 * (left + right)),
                )
                if best is None or depth > best[0]:
                    best = (depth, candidate)
            if best is not None:
                scored.append(best)

    scored.sort(key=lambda item: -item[0])
    sites: list[CDSite] = []
    for _, candidate in scored[:max_sites]:
        measured = measure_site(clean2d, candidate, tolerance=tolerance, smooth=smooth)
        if measured is None:
            continue
        sites.append(
            CDSite(
                orientation=candidate.orientation,
                band_start=candidate.band_start,
                band_stop=candidate.band_stop,
                left_edge=candidate.left_edge,
                right_edge=candidate.right_edge,
                polarity=candidate.polarity,
                clean_cd=measured[0],
                clean_center=measured[1],
            )
        )
    return sites


# ---------------------------------------------------------------------------
# realization generation


def _center_crop01(source: BurstSource, image_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = source.clean.shape[:2]
    top = (height - image_size) // 2
    left = (width - image_size) // 2
    window = np.s_[top : top + image_size, left : left + image_size]
    clean01 = _to_hwc01(source.clean[window])
    frames01 = [_to_hwc01(frame[window]) for frame in source.frames]
    return clean01, frames01


def _classical_realizations(
    frames01: list[np.ndarray], num_seeds: int, avg_counts: Sequence[int]
) -> dict[str, list[np.ndarray]]:
    methods = {"single_frame": frames01[:num_seeds]}
    for count in avg_counts:
        groups = len(frames01) // count
        methods[f"avg_of_{count}"] = [
            np.mean(frames01[group * count : (group + 1) * count], axis=0)
            for group in range(groups)
        ]
    return methods


def _model_realizations(
    sampler: Sampler, seeds01: list[np.ndarray], schedule: list[int], max_batch: int
) -> dict[str, list[np.ndarray]]:
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_METHOD_NAMES}
    for start in range(0, len(seeds01), max_batch):
        chunk = seeds01[start : start + max_batch]
        batch = torch.cat([_to_model_tensor(seed) for seed in chunk], dim=0)
        one_shot = sampler.run(batch, schedule=[sampler.num_steps])
        full = sampler.run(batch, schedule=schedule)
        for index in range(len(chunk)):
            item = np.s_[index : index + 1]
            outputs["one_shot"].append(_from_model_tensor(one_shot.prediction[item]))
            outputs["iter_average"].append(_from_model_tensor(full.average[item]))
            outputs["iter_prediction"].append(_from_model_tensor(full.prediction[item]))
    return outputs


# ---------------------------------------------------------------------------
# per-method accumulation and summary


def _new_accumulator() -> dict:
    return {
        "psnr": [],
        "ssim": [],
        "realizations": [],
        "pixel_sigma_means": [],
        "pixel_sigma_p95s": [],
        "shift_ss": 0.0,
        "shift_dof": 0,
        "shift_mean_vectors": [],
        "cd_ss": 0.0,
        "cd_dof": 0,
        "center_ss": 0.0,
        "center_dof": 0,
        "cd_biases": [],
        "cd_valid": 0,
        "cd_total": 0,
        # Scene-level (per-source) statistics. Sites inside one source share the
        # scene, the frames and the model outputs, so they are NOT independent
        # experimental units -- site-pooled sigma over-weights whichever scene
        # happens to yield the most sites. The independent unit is the source.
        "cd_scene_sigmas": [],
        "shift_scene_sigmas": [],
    }


def _accumulate_method(
    accumulator: dict,
    realizations01: list[np.ndarray],
    clean01: np.ndarray,
    sites: Sequence[CDSite],
    *,
    tolerance: float,
    smooth: int,
    include_shift: bool,
) -> tuple[np.ndarray | None, dict]:
    """Fold one source's realizations into a method accumulator.

    ``include_shift=False`` skips the global-registration statistics for this
    source (the area failed the registrability gate); the exclusion applies to
    every method identically, so comparisons stay fair. Returns the per-pixel
    sigma map (channel-averaged; ``None`` for a single realization) plus a
    JSON-ready detail record.
    """
    count = len(realizations01)
    accumulator["realizations"].append(count)
    for realization in realizations01:
        accumulator["psnr"].append(psnr(clean01, realization))
        accumulator["ssim"].append(ssim(clean01, realization))

    sigma_map: np.ndarray | None = None
    if count >= 2:
        stack = np.stack(realizations01)
        sigma = stack.std(axis=0, ddof=1) / _c4(count)
        accumulator["pixel_sigma_means"].append(float(sigma.mean()))
        accumulator["pixel_sigma_p95s"].append(float(np.percentile(sigma, 95.0)))
        sigma_map = sigma.mean(axis=-1)

    shifts: np.ndarray | None = None
    if include_shift:
        clean2d = _gray2d(clean01)
        shifts = np.array(
            [estimate_shift(clean2d, _gray2d(realization)) for realization in realizations01]
        )
        mean_vector = shifts.mean(axis=0)
        accumulator["shift_mean_vectors"].append(mean_vector)
        if count >= 2:
            deviations = shifts - mean_vector
            scene_ss = float((deviations**2).sum())
            accumulator["shift_ss"] += scene_ss
            accumulator["shift_dof"] += 2 * (count - 1)
            accumulator["shift_scene_sigmas"].append(
                _pooled_sigma(scene_ss, 2 * (count - 1))
            )

    scene_cd_ss, scene_cd_dof = 0.0, 0
    cd_detail: list[dict] = []
    for site in sites:
        values = np.full(count, np.nan)
        centers = np.full(count, np.nan)
        for index, realization in enumerate(realizations01):
            measured = measure_site(
                _gray2d(realization), site, tolerance=tolerance, smooth=smooth
            )
            if measured is not None:
                values[index], centers[index] = measured
        valid = ~np.isnan(values)
        accumulator["cd_total"] += count
        accumulator["cd_valid"] += int(valid.sum())
        if valid.sum() >= 1:
            accumulator["cd_biases"].append(float(values[valid].mean() - site.clean_cd))
        if valid.sum() >= 2:
            kept_values = values[valid]
            kept_centers = centers[valid]
            site_ss = float(((kept_values - kept_values.mean()) ** 2).sum())
            accumulator["cd_ss"] += site_ss
            accumulator["cd_dof"] += int(valid.sum()) - 1
            scene_cd_ss += site_ss
            scene_cd_dof += int(valid.sum()) - 1
            accumulator["center_ss"] += float(((kept_centers - kept_centers.mean()) ** 2).sum())
            accumulator["center_dof"] += int(valid.sum()) - 1
        cd_detail.append(
            {
                "cd_values": [None if math.isnan(v) else float(v) for v in values],
                "centers": [None if math.isnan(v) else float(v) for v in centers],
            }
        )

    scene_cd_sigma = _pooled_sigma(scene_cd_ss, scene_cd_dof)
    if scene_cd_sigma is not None:
        accumulator["cd_scene_sigmas"].append(scene_cd_sigma)

    detail = {
        "psnr": [float(value) for value in accumulator["psnr"][-count:]],
        "shifts": (
            None if shifts is None else [[float(dy), float(dx)] for dy, dx in shifts]
        ),
        "pixel_sigma_mean": accumulator["pixel_sigma_means"][-1] if count >= 2 else None,
        "cd_scene_sigma_px": scene_cd_sigma,
        "cd_sites": cd_detail,
    }
    return sigma_map, detail


def _summarize_method(accumulator: dict, sites_total: int) -> dict:
    shift_vectors = np.array(accumulator["shift_mean_vectors"])
    pixel_available = len(accumulator["pixel_sigma_means"]) > 0
    cd_sigma = _pooled_sigma(accumulator["cd_ss"], accumulator["cd_dof"])
    return {
        "realizations_per_source": accumulator["realizations"],
        "accuracy": {
            "psnr_mean": _mean_or_none(accumulator["psnr"]),
            "psnr_std": (
                float(np.std(accumulator["psnr"], ddof=1))
                if len(accumulator["psnr"]) >= 2
                else None
            ),
            "ssim_mean": _mean_or_none(accumulator["ssim"]),
        },
        "pixel_repeatability": (
            {
                "sigma_mean": _mean_or_none(accumulator["pixel_sigma_means"]),
                "sigma_p95_mean": _mean_or_none(accumulator["pixel_sigma_p95s"]),
            }
            if pixel_available
            else None
        ),
        "cd": {
            "sites": sites_total,
            "measurements": accumulator["cd_total"],
            "success_rate": (
                accumulator["cd_valid"] / accumulator["cd_total"]
                if accumulator["cd_total"] > 0
                else None
            ),
            "pooled_sigma_px": cd_sigma,
            "pooled_3sigma_px": None if cd_sigma is None else 3.0 * cd_sigma,
            # Scene-level: the source is the independent unit, so this is the
            # figure to compare methods on (site-pooled values over-weight
            # site-rich scenes). Reported alongside, never instead.
            "scene_sigmas_px": list(accumulator["cd_scene_sigmas"]),
            "scene_median_sigma_px": _median_or_none(accumulator["cd_scene_sigmas"]),
            "scene_median_3sigma_px": (
                None
                if not accumulator["cd_scene_sigmas"]
                else 3.0 * float(np.median(accumulator["cd_scene_sigmas"]))
            ),
            "scenes": len(accumulator["cd_scene_sigmas"]),
            "bias_mean_px": _mean_or_none(accumulator["cd_biases"]),
            "bias_abs_mean_px": _mean_or_none([abs(b) for b in accumulator["cd_biases"]]),
            "center_pooled_sigma_px": _pooled_sigma(
                accumulator["center_ss"], accumulator["center_dof"]
            ),
        },
        "registration": {
            "shift_sigma_px": _pooled_sigma(accumulator["shift_ss"], accumulator["shift_dof"]),
            "scene_median_sigma_px": _median_or_none(accumulator["shift_scene_sigmas"]),
            "shift_bias_px": (
                float(np.linalg.norm(shift_vectors.mean(axis=0)))
                if shift_vectors.size
                else None
            ),
            "shift_bias_abs_mean_px": (
                float(np.mean(np.linalg.norm(shift_vectors, axis=1)))
                if shift_vectors.size
                else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# report writers


def _write_sigma_map_grid(
    sigma_maps: dict[str, list[tuple[int, np.ndarray]]],
    method_names: Sequence[str],
    path: Path,
    *,
    upscale: int = 3,
) -> None:
    rows = [name for name in method_names if sigma_maps.get(name)]
    if not rows:
        return
    caption_height, pad = 26, 4
    font = ImageFont.load_default()
    columns = max(len(sigma_maps[name]) for name in rows)
    tile_side = sigma_maps[rows[0]][0][1].shape[0] * upscale
    cell_width = tile_side + pad
    cell_height = tile_side + caption_height + pad
    canvas = Image.new(
        "RGB", (columns * cell_width + pad, len(rows) * cell_height + pad), color=(24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, name in enumerate(rows):
        for column_index, (source_index, sigma_map) in enumerate(sigma_maps[name]):
            scaled = np.clip(sigma_map / _SIGMA_MAP_FULL_SCALE * 255.0, 0.0, 255.0)
            tile = scaled.astype(np.uint8).repeat(upscale, axis=0).repeat(upscale, axis=1)
            x = pad + column_index * cell_width
            y = pad + row_index * cell_height
            canvas.paste(Image.fromarray(tile).convert("RGB"), (x, y))
            draw.text((x, y + tile_side + 2), name, fill=(230, 230, 230), font=font)
            draw.text(
                (x, y + tile_side + 13),
                f"src {source_index} | sigma={sigma_map.mean() * 1e3:.2f}e-3",
                fill=(230, 230, 230),
                font=font,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _format(value: float | None, spec: str, scale: float = 1.0) -> str:
    return "-" if value is None else format(value * scale, spec)


def _write_summary_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Repeatability & metrology summary",
        "",
        f"dataset: `{results['dataset_dir']}` | split: {results['split']} | "
        f"sources: {results['count']} | seeds/source: {results['num_seeds']} | "
        f"CD sites: {results['cd_sites_total']} "
        f"(sources without sites: {results['sources_without_sites']}) | "
        f"registrable sources: {results['registration_sources']}/{results['count']} "
        f"(gate {results['registration_gate_px']} px)",
        "",
        "Sigmas are c4-debiased; units are intensity in [0,1] for pixel sigma and",
        "pixels for CD/registration. **`CD 3sig scene` is the headline CD figure**:",
        "the median of per-scene pooled 3-sigma values, because sites inside one",
        "scene share frames and model outputs and are not independent units, so the",
        "site-pooled column over-weights site-rich scenes. Bias is reported both",
        "signed (systematic offset; opposite-sign sites cancel) and absolute (typical",
        "per-site magnitude) -- they answer different questions and can differ by an",
        "order of magnitude.",
        "",
        "| method | K/src | PSNR dB | pixel sigma x1e-3 | CD 3sig scene px | CD 3sig sites px "
        "| CD bias px | CD abs-bias px | CD success | center sigma px | shift sigma px "
        "| shift bias px |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, method in results["methods"].items():
        realizations = method["realizations_per_source"]
        k_text = str(realizations[0]) if len(set(realizations)) == 1 else "var"
        pixel = method["pixel_repeatability"]
        cd = method["cd"]
        registration = method["registration"]
        lines.append(
            "| {name} | {k} | {psnr} | {pixel} | {cdscene} | {cd3} | {cdbias} | {cdabs} "
            "| {success} | {center} | {shift} | {shiftbias} |".format(
                name=name,
                k=k_text,
                psnr=_format(method["accuracy"]["psnr_mean"], ".2f"),
                pixel=_format(None if pixel is None else pixel["sigma_mean"], ".2f", 1e3),
                cdscene=_format(cd["scene_median_3sigma_px"], ".3f"),
                cd3=_format(cd["pooled_3sigma_px"], ".3f"),
                cdbias=_format(cd["bias_mean_px"], "+.3f"),
                cdabs=_format(cd["bias_abs_mean_px"], ".3f"),
                success=_format(cd["success_rate"], ".1%"),
                center=_format(cd["center_pooled_sigma_px"], ".3f"),
                shift=_format(registration["shift_sigma_px"], ".3f"),
                shiftbias=_format(registration["shift_bias_px"], ".3f"),
            )
        )
    lines.append("")
    lines.append(
        f"Per-scene CD 3-sigma values back the scene column "
        f"(n = {results['methods'][next(iter(results['methods']))]['cd']['scenes']} scenes "
        "with >= 1 measurable site); see `repeatability.json` -> "
        "`methods.<name>.cd.scene_sigmas_px`."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# orchestration


def repeatability(
    config: Config,
    checkpoints: dict[str, str | Path],
    *,
    out_dir: str | Path,
    split: str = "val",
    limit: int | None = None,
    num_seeds: int = 10,
    avg_counts: Sequence[int] = (2, 4, 8, 16),
    sample_steps: int | None = None,
    device: str | None = None,
    band_height: int = 16,
    smooth: int = 3,
    edge_tolerance: float = 4.0,
    max_sites: int = 8,
    max_batch: int = 10,
    registration_gate_px: float = 0.5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run the repeatability evaluation; writes repeatability.json + summary.md.

    ``checkpoints`` maps an arm label to a checkpoint path; model methods are
    reported as ``{method}@{arm}``. Classical methods are computed once.

    Registration statistics are gated per source: burst frames are pixel-
    aligned by the pipeline's premise, so the clean image registered against
    its own all-frame average must come out at ~0 shift. A source whose gate
    shift exceeds ``registration_gate_px`` has no usable registration content
    (e.g. a near-featureless crop where the correlation peak wanders), and is
    excluded from the shift statistics of every method identically.
    """
    if split not in ("val", "train", "test"):
        raise ValueError(f"split must be 'val', 'train', or 'test', got {split!r}")
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
    if not checkpoints:
        raise ValueError("provide at least one checkpoint arm")

    samplers: dict[str, Sampler] = {}
    for arm, path in checkpoints.items():
        sampler = Sampler.from_checkpoint(
            path, device=device if device is not None else config.training.device
        )
        if sampler.num_steps != config.schedule.num_steps:
            raise ValueError(
                f"checkpoint {path} was trained with num_steps={sampler.num_steps} but "
                f"the config says {config.schedule.num_steps}"
            )
        samplers[arm] = sampler

    cache = BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=min_replicas(config.schedule.num_steps),
        min_size=config.data.image_size,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        split_seed=config.data.split_seed,
    )
    sources = cache.sources_for_split(split)
    if not sources:
        raise ValueError(f"no sources in the {split!r} split")
    if limit is not None:
        sources = sources[:limit]

    min_frames = min(len(source.frames) for source in sources)
    if num_seeds > min_frames:
        warnings.warn(
            f"num_seeds={num_seeds} exceeds the {min_frames} frames available on every "
            f"source; clamping to {min_frames}",
            stacklevel=2,
        )
        num_seeds = min_frames
    effective_avg_counts = []
    for count in avg_counts:
        if count < 2 or count > min_frames:
            warnings.warn(
                f"skipping avg_of_{count}: needs 2 <= m <= {min_frames} frames", stacklevel=2
            )
            continue
        effective_avg_counts.append(int(count))

    schedule = sampling_schedule(
        config.schedule.num_steps,
        sample_steps if sample_steps is not None else config.sampling.num_sample_steps,
    )
    method_names = ["single_frame"] + [f"avg_of_{count}" for count in effective_avg_counts]
    for arm in samplers:
        method_names += [f"{name}@{arm}" for name in MODEL_METHOD_NAMES]

    accumulators = {name: _new_accumulator() for name in method_names}
    sigma_maps: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in method_names}
    per_source: list[dict] = []
    sites_total = 0
    sources_without_sites = 0

    registration_sources = 0
    for index, source in enumerate(sources):
        clean01, frames01 = _center_crop01(source, config.data.image_size)
        sites = find_cd_sites(
            _gray2d(clean01),
            band_height=min(band_height, config.data.image_size),
            smooth=smooth,
            tolerance=edge_tolerance,
            max_sites=max_sites,
        )
        sites_total += len(sites)
        if not sites:
            sources_without_sites += 1

        gate_shift = estimate_shift(
            _gray2d(clean01), _gray2d(np.mean(frames01, axis=0))
        )
        registrable = math.hypot(*gate_shift) <= registration_gate_px
        registration_sources += int(registrable)

        realizations = _classical_realizations(frames01, num_seeds, effective_avg_counts)
        seeds01 = frames01[:num_seeds]
        for arm, sampler in samplers.items():
            for name, outputs in _model_realizations(
                sampler, seeds01, schedule, max_batch
            ).items():
                realizations[f"{name}@{arm}"] = outputs

        source_detail: dict = {
            "source_index": source.source_index,
            "sites": [asdict(site) for site in sites],
            "registration_gate_shift": [float(value) for value in gate_shift],
            "registrable": registrable,
            "methods": {},
        }
        for name in method_names:
            sigma_map, detail = _accumulate_method(
                accumulators[name],
                realizations[name],
                clean01,
                sites,
                tolerance=edge_tolerance,
                smooth=smooth,
                include_shift=registrable,
            )
            source_detail["methods"][name] = detail
            if index < 4 and sigma_map is not None:
                sigma_maps[name].append((source.source_index, sigma_map))
        per_source.append(source_detail)
        if progress_callback is not None:
            progress_callback(index + 1, len(sources))

    results = {
        "dataset_dir": str(config.data.dataset_dir),
        "split": split,
        "count": len(sources),
        "source_indices": [source.source_index for source in sources],
        "num_seeds": num_seeds,
        "avg_counts": effective_avg_counts,
        "sample_steps": len(schedule),
        "checkpoints": {arm: str(path) for arm, path in checkpoints.items()},
        "cd_sites_total": sites_total,
        "sources_without_sites": sources_without_sites,
        "registration_sources": registration_sources,
        "registration_gate_px": registration_gate_px,
        "edge_tolerance_px": edge_tolerance,
        "band_height": band_height,
        "methods": {
            name: _summarize_method(accumulators[name], sites_total) for name in method_names
        },
        "per_source": per_source,
    }

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "repeatability.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary_markdown(results, destination / "summary.md")
    _write_sigma_map_grid(sigma_maps, method_names, destination / "sigma_maps.png")
    return results
