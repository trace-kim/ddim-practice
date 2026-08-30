from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from burst_diffusion.config import Config
from burst_diffusion.repeatability import (
    _c4,
    _pooled_sigma,
    estimate_shift,
    find_cd_sites,
    measure_site,
    repeatability,
)
from burst_diffusion.train import CHECKPOINT_FORMAT
from burst_diffusion.unet import build_unet


# ---------------------------------------------------------------------------
# statistics


def test_c4_matches_the_closed_form_for_two_samples() -> None:
    assert _c4(2) == pytest.approx(math.sqrt(2.0 / math.pi))


def test_pooled_sigma_is_bessel_and_c4_corrected() -> None:
    # Two groups of [0, 1]: each contributes SS = 0.5 with 1 dof.
    groups = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    sum_squares = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    dof = sum(len(g) - 1 for g in groups)
    expected = math.sqrt(sum_squares / dof) / _c4(dof + 1)
    assert _pooled_sigma(sum_squares, dof) == pytest.approx(expected)
    assert _pooled_sigma(0.0, 0) is None


# ---------------------------------------------------------------------------
# global registration


def _smooth_random_image(size: int = 64, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.normal(size=(size, size)))
    freq_y = np.fft.fftfreq(size)[:, None]
    freq_x = np.fft.fftfreq(size)[None, :]
    lowpass = np.exp(-(freq_y**2 + freq_x**2) / (2.0 * 0.05**2))
    return np.real(np.fft.ifft2(spectrum * lowpass))


def _fourier_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    height, width = image.shape
    freq_y = np.fft.fftfreq(height)[:, None]
    freq_x = np.fft.fftfreq(width)[None, :]
    phase = np.exp(-2j * np.pi * (freq_y * dy + freq_x * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(image) * phase))


def test_shift_estimation_recovers_a_known_subpixel_translation() -> None:
    reference = _smooth_random_image()
    moved = _fourier_shift(reference, 0.30, -0.45)
    dy, dx = estimate_shift(reference, moved)
    assert dy == pytest.approx(0.30, abs=0.05)
    assert dx == pytest.approx(-0.45, abs=0.05)


def test_shift_estimation_is_zero_for_identical_images() -> None:
    reference = _smooth_random_image(seed=3)
    dy, dx = estimate_shift(reference, reference.copy())
    assert abs(dy) < 0.03 and abs(dx) < 0.03


# ---------------------------------------------------------------------------
# CD measurement


def _bar_image(size: int, left: float, right: float, *, low: float = 0.2, high: float = 0.8) -> np.ndarray:
    """Vertical bright bar with area-sampled (anti-aliased) subpixel edges."""
    x = np.arange(size, dtype=np.float64)
    coverage = np.clip(np.minimum(x + 1.0, right) - np.maximum(x, left), 0.0, 1.0)
    return np.tile(low + (high - low) * coverage, (size, 1))


def test_cd_sites_and_measurement_recover_bar_width() -> None:
    image = _bar_image(64, 20.3, 40.7)
    sites = find_cd_sites(image)
    assert sites, "the bar must be detected as a measurement site"
    site = sites[0]
    assert site.orientation == "h"
    assert site.polarity == 1
    # Width is convention-free (edge offsets cancel); the center carries the
    # half-pixel sample-index convention.
    assert site.clean_cd == pytest.approx(20.4, abs=0.5)
    assert site.clean_center == pytest.approx(30.0, abs=0.35)
    measured = measure_site(image, site, tolerance=4.0, smooth=3)
    assert measured is not None
    assert measured[0] == pytest.approx(site.clean_cd, abs=1e-9)


def test_cd_measurement_tracks_a_quarter_pixel_shift() -> None:
    base = _bar_image(64, 20.3, 40.7)
    shifted = _bar_image(64, 20.55, 40.95)
    sites = find_cd_sites(base)
    assert sites
    site = sites[0]
    measured_base = measure_site(base, site, tolerance=4.0, smooth=3)
    measured_shifted = measure_site(shifted, site, tolerance=4.0, smooth=3)
    assert measured_base is not None and measured_shifted is not None
    assert measured_shifted[1] - measured_base[1] == pytest.approx(0.25, abs=0.08)
    assert measured_shifted[0] - measured_base[0] == pytest.approx(0.0, abs=0.08)


def test_cd_measurement_fails_gracefully_without_a_feature() -> None:
    image = _bar_image(64, 20.3, 40.7)
    sites = find_cd_sites(image)
    assert sites
    flat = np.full_like(image, 0.5)
    assert measure_site(flat, sites[0], tolerance=4.0, smooth=3) is None
    assert find_cd_sites(flat) == []


# ---------------------------------------------------------------------------
# end-to-end


def _write_bar_burst(root: Path, *, num_sources: int = 4, replicas: int = 4) -> Path:
    """Structured burst dataset: 20x24 images with a bar the center crop sees."""
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(num_sources):
        # Bar columns [9.5, 14.5] land at [5.5, 10.5] inside the 16px center crop.
        profile = _bar_image(24, 9.5, 14.5)[0]
        clean = np.rint(np.tile(profile, (20, 1)) * 255.0).astype(np.uint8)
        Image.fromarray(clean).save(burst / "clean" / f"{source_index:05d}.png")
        for replica_index in range(replicas):
            noisy = (clean.astype(np.int16) + rng.integers(-30, 31, clean.shape)).clip(0, 255)
            name = f"noisy/{source_index:05d}_{replica_index:05d}.png"
            Image.fromarray(noisy.astype(np.uint8)).save(burst / name)
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


def _config(dataset_dir: Path, run_dir: Path) -> Config:
    return Config.model_validate(
        {
            "data": {
                "dataset_dir": str(dataset_dir),
                "image_size": 16,
                "channels": 1,
                "val_fraction": 0.5,
            },
            "schedule": {"num_steps": 2},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(run_dir), "device": "cpu"},
            "sampling": {},
        }
    )


def _write_checkpoint(path: Path, config: Config) -> Path:
    torch.manual_seed(0)
    model = build_unet(config)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": 1,
        "config": config.model_dump(mode="json"),
        "model": model.state_dict(),
        "ema": None,
        "optimizer": {},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "factory": {"rng_state": np.random.default_rng(0).bit_generator.state},
    }
    torch.save(payload, path)
    return path


def _run(tmp_path: Path, out_name: str = "rep") -> dict:
    dataset = _write_bar_burst(tmp_path / "data")
    config = _config(dataset, tmp_path / "run")
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt", config)
    with pytest.warns(UserWarning, match="no EMA state"):
        return repeatability(
            config,
            {"model": checkpoint},
            out_dir=tmp_path / out_name,
            num_seeds=2,
            avg_counts=(2,),
            edge_tolerance=3.0,
        )


def test_repeatability_end_to_end_reports_every_method(tmp_path: Path) -> None:
    results = _run(tmp_path)
    expected = {
        "single_frame",
        "avg_of_2",
        "one_shot@model",
        "iter_average@model",
        "iter_prediction@model",
    }
    assert set(results["methods"].keys()) == expected
    assert results["num_seeds"] == 2
    assert results["cd_sites_total"] >= results["count"]  # the bar is found per source
    for method in results["methods"].values():
        assert method["accuracy"]["psnr_mean"] is not None
        assert np.isfinite(method["accuracy"]["psnr_mean"])
        assert method["pixel_repeatability"] is not None
    assert (tmp_path / "rep" / "repeatability.json").is_file()
    assert (tmp_path / "rep" / "summary.md").is_file()
    assert (tmp_path / "rep" / "sigma_maps.png").is_file()
    # The written JSON must be valid (no bare NaN) and reload to the same numbers.
    reloaded = json.loads((tmp_path / "rep" / "repeatability.json").read_text(encoding="utf-8"))
    assert reloaded["methods"]["single_frame"]["accuracy"]["psnr_mean"] == pytest.approx(
        results["methods"]["single_frame"]["accuracy"]["psnr_mean"]
    )


def test_averaging_improves_pixel_and_cd_repeatability(tmp_path: Path) -> None:
    results = _run(tmp_path)
    single = results["methods"]["single_frame"]
    averaged = results["methods"]["avg_of_2"]
    assert (
        averaged["pixel_repeatability"]["sigma_mean"]
        < single["pixel_repeatability"]["sigma_mean"]
    )
    # Both classical methods measure the bar on every realization.
    assert single["cd"]["success_rate"] == pytest.approx(1.0)
    assert averaged["cd"]["success_rate"] == pytest.approx(1.0)
    assert single["cd"]["pooled_sigma_px"] is not None
    assert averaged["cd"]["pooled_sigma_px"] is not None


def test_repeatability_is_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path, out_name="rep_a")
    second = _run(tmp_path / "again", out_name="rep_b")
    for name in first["methods"]:
        assert first["methods"][name]["accuracy"]["psnr_mean"] == pytest.approx(
            second["methods"][name]["accuracy"]["psnr_mean"]
        )
        first_cd = first["methods"][name]["cd"]["pooled_sigma_px"]
        second_cd = second["methods"][name]["cd"]["pooled_sigma_px"]
        if first_cd is None:
            assert second_cd is None
        else:
            assert first_cd == pytest.approx(second_cd)


def test_misaligned_frames_fail_the_registration_gate_for_every_method(tmp_path: Path) -> None:
    """Frames shifted 3 px against clean violate the pixel-alignment premise:
    the per-source registrability gate (clean vs the all-frame average) must
    exclude such sources from the shift statistics of ALL methods identically,
    while pixel repeatability stays reported."""
    burst = tmp_path / "data" / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(4):
        profile = _bar_image(24, 9.5, 14.5)[0]
        clean = np.rint(np.tile(profile, (20, 1)) * 255.0).astype(np.uint8)
        Image.fromarray(clean).save(burst / "clean" / f"{source_index:05d}.png")
        moved = np.roll(clean, 3, axis=1)
        for replica_index in range(4):
            noisy = (moved.astype(np.int16) + rng.integers(-10, 11, clean.shape)).clip(0, 255)
            name = f"noisy/{source_index:05d}_{replica_index:05d}.png"
            Image.fromarray(noisy.astype(np.uint8)).save(burst / name)
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

    config = _config(tmp_path / "data", tmp_path / "run")
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt", config)
    with pytest.warns(UserWarning, match="no EMA state"):
        results = repeatability(
            config,
            {"model": checkpoint},
            out_dir=tmp_path / "rep",
            num_seeds=2,
            avg_counts=(2,),
            edge_tolerance=3.0,
        )
    assert results["registration_sources"] == 0
    for method in results["methods"].values():
        assert method["registration"]["shift_sigma_px"] is None
        assert method["registration"]["shift_bias_px"] is None
        assert method["pixel_repeatability"] is not None


def test_single_realization_methods_report_no_precision(tmp_path: Path) -> None:
    dataset = _write_bar_burst(tmp_path / "data", replicas=3)
    config = _config(dataset, tmp_path / "run")
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt", config)
    with pytest.warns(UserWarning, match="no EMA state"):
        results = repeatability(
            config,
            {"model": checkpoint},
            out_dir=tmp_path / "rep",
            num_seeds=1,
            avg_counts=(3,),
            edge_tolerance=3.0,
        )
    lonely = results["methods"]["avg_of_3"]
    assert lonely["realizations_per_source"] == [1, 1]
    assert lonely["pixel_repeatability"] is None
    assert lonely["cd"]["pooled_sigma_px"] is None
    assert lonely["registration"]["shift_sigma_px"] is None
    # Accuracy and bias are still reported from the single realization.
    assert lonely["accuracy"]["psnr_mean"] is not None
    assert lonely["cd"]["bias_mean_px"] is not None
