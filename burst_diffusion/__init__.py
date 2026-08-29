"""Burst-averaging diffusion-style denoiser: train a network to reconstruct
clean images from noisy bursts, where the DDPM "epsilon" is a real noisy frame
and the timestep corresponds to how many frames are averaged.

Standalone package: it never imports the legacy DDIM code (``models``,
``runners``, ``functions``, ``datasets``) or ``ddimctl``. It may import
``noising_pipeline``, the sanctioned synthetic-data tool. See README.md in
this directory for the method, its assumptions, and the full workflow.
"""

from .config import Config, load_config
from .data import BatchFactory, BurstCache, resolve_burst_dir
from .ema import EMAHelper, ema_parameters
from .evaluate import evaluate
from .generate import generate_burst_dataset
from .metrics import psnr, ssim
from .preview import make_preview_grid
from .sample import Sampler, SampleResult
from .schedule import frames_at, min_replicas, sample_step, sampling_schedule
from .train import Trainer, load_checkpoint, save_checkpoint
from .unet import UNet, build_unet

__all__ = [
    "BatchFactory",
    "BurstCache",
    "Config",
    "EMAHelper",
    "SampleResult",
    "Sampler",
    "Trainer",
    "UNet",
    "build_unet",
    "ema_parameters",
    "evaluate",
    "frames_at",
    "generate_burst_dataset",
    "load_checkpoint",
    "load_config",
    "make_preview_grid",
    "min_replicas",
    "psnr",
    "resolve_burst_dir",
    "sample_step",
    "sampling_schedule",
    "save_checkpoint",
    "ssim",
]
