from __future__ import annotations

import numpy as np
import pytest

from burst_diffusion.metrics import PSNR_CAP_DB, psnr, ssim


def test_psnr_matches_the_known_constant_difference_value() -> None:
    a = np.zeros((13, 17), dtype=np.float64)
    b = np.full((13, 17), 0.1, dtype=np.float64)
    # MSE = 0.01 with data_range 1.0 -> exactly 20 dB.
    assert psnr(a, b) == pytest.approx(20.0, abs=1e-9)
    assert psnr(b, a) == pytest.approx(20.0, abs=1e-9)


def test_psnr_is_capped_for_identical_inputs() -> None:
    image = np.random.default_rng(0).random((9, 7, 3))
    assert psnr(image, image) == PSNR_CAP_DB


def test_psnr_respects_data_range() -> None:
    a = np.zeros((8, 8))
    b = np.full((8, 8), 25.5)
    assert psnr(a, b, data_range=255.0) == pytest.approx(20.0, abs=1e-9)


def test_psnr_rejects_mismatched_shapes_and_bad_range() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr(np.zeros((4, 4)), np.zeros((4, 5)))
    with pytest.raises(ValueError, match="data_range must be positive"):
        psnr(np.zeros((4, 4)), np.zeros((4, 4)), data_range=0.0)
    with pytest.raises(ValueError, match=r"must be \[H, W\] or \[H, W, C\]"):
        psnr(np.zeros(4), np.zeros(4))


def test_ssim_is_one_for_identical_and_symmetric() -> None:
    image = np.random.default_rng(1).random((21, 19))
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-12)
    noisy = np.clip(image + np.random.default_rng(2).normal(0, 0.1, image.shape), 0, 1)
    assert ssim(image, noisy) == pytest.approx(ssim(noisy, image), abs=1e-12)


def test_ssim_decreases_as_noise_grows_and_stays_in_range() -> None:
    rng = np.random.default_rng(3)
    base = rng.random((32, 32))
    previous = 1.0
    for std in (0.05, 0.2, 0.5):
        noisy = base + rng.normal(0.0, std, base.shape)
        value = ssim(base, noisy)
        assert -1.0 <= value <= 1.0
        assert value < previous
        previous = value


def test_ssim_rejects_images_smaller_than_the_window() -> None:
    with pytest.raises(ValueError, match="smaller than the 11x11 window"):
        ssim(np.zeros((8, 8)), np.zeros((8, 8)))
    with pytest.raises(ValueError, match="window must be an odd integer"):
        ssim(np.zeros((16, 16)), np.zeros((16, 16)), window=4)
