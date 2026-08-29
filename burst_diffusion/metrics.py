"""Image quality metrics: PSNR and Gaussian-window SSIM.

Implemented with numpy/torch only (the repo deliberately has no
scikit-image/scipy dependency). SSIM follows Wang et al. 2004 with the
skimage-compatible defaults: 11x11 Gaussian window, sigma 1.5, K1=0.01,
K2=0.03. Inputs are CPU arrays (numpy, or anything ``np.asarray`` accepts)
shaped ``[H, W]`` or ``[H, W, C]``, compared over a shared ``data_range``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

PSNR_CAP_DB = 100.0


def _as_float_image(array: object, name: str) -> np.ndarray:
    image = np.asarray(array, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"{name} must be [H, W] or [H, W, C], got shape {image.shape}")
    return image


def psnr(a: object, b: object, *, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB, capped at ``PSNR_CAP_DB`` for identical inputs."""
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    image_a = _as_float_image(a, "a")
    image_b = _as_float_image(b, "b")
    if image_a.shape != image_b.shape:
        raise ValueError(f"shape mismatch: {image_a.shape} vs {image_b.shape}")
    mse = float(np.mean((image_a - image_b) ** 2))
    if mse == 0.0:
        return PSNR_CAP_DB
    value = 10.0 * math.log10((data_range * data_range) / mse)
    return min(value, PSNR_CAP_DB)


def _gaussian_window(window: int, sigma: float) -> torch.Tensor:
    offsets = torch.arange(window, dtype=torch.float64) - (window - 1) / 2.0
    kernel_1d = torch.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.reshape(1, 1, window, window)


def ssim(
    a: object,
    b: object,
    *,
    data_range: float = 1.0,
    window: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean structural similarity over all channels (Gaussian-weighted windows)."""
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be an odd integer >= 3, got {window}")
    image_a = _as_float_image(a, "a")
    image_b = _as_float_image(b, "b")
    if image_a.shape != image_b.shape:
        raise ValueError(f"shape mismatch: {image_a.shape} vs {image_b.shape}")
    height, width, _ = image_a.shape
    if min(height, width) < window:
        raise ValueError(
            f"image sides {height}x{width} are smaller than the {window}x{window} window"
        )

    # Channels ride the batch dimension: [C, 1, H, W] with a single [1,1,k,k] kernel.
    x = torch.from_numpy(np.ascontiguousarray(image_a.transpose(2, 0, 1)))[:, None]
    y = torch.from_numpy(np.ascontiguousarray(image_b.transpose(2, 0, 1)))[:, None]
    kernel = _gaussian_window(window, sigma)

    mu_x = torch.nn.functional.conv2d(x, kernel)
    mu_y = torch.nn.functional.conv2d(y, kernel)
    var_x = torch.nn.functional.conv2d(x * x, kernel) - mu_x * mu_x
    var_y = torch.nn.functional.conv2d(y * y, kernel) - mu_y * mu_y
    cov_xy = torch.nn.functional.conv2d(x * y, kernel) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float((numerator / denominator).mean().item())
