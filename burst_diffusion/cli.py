"""Command-line interface: ``python -m burst_diffusion <command>``.

Commands cover the full experiment loop: ``generate`` (synthetic burst dataset
+ noise verification), ``preview`` (visual "noisy enough" grid), ``train``,
``sample`` (denoise one measurement), and ``evaluate`` (five-baseline report).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="burst_diffusion",
    help="Burst-averaging diffusion-style denoiser (standalone pipeline).",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _configure_logging(
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Log progress to stderr."),
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s - %(name)s - %(message)s",
    )


def _parse_int_list(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise typer.BadParameter(f"expected comma-separated integers, got {raw!r}") from error
    if not values:
        raise typer.BadParameter("expected at least one integer")
    return values


@app.command()
def generate(
    source_dir: Path = typer.Option(..., help="Directory of clean source images (searched recursively)."),
    output_dir: Path = typer.Option(..., help="Dataset output directory."),
    num_sources: int = typer.Option(96, min=1, help="Clean images to use."),
    replicas: int = typer.Option(16, min=1, help="Noisy frames per clean image (N >= T+1)."),
    noise_type: list[str] = typer.Option(["poisson"], help="Noise distribution(s), applied in order."),
    steps: int = typer.Option(1, min=1, help="Corruption-strength multiplier (fused)."),
    peak: Optional[float] = typer.Option(None, help="Poisson peak (photon count at intensity 1)."),
    gaussian_std: Optional[float] = typer.Option(None, help="Gaussian std per step."),
    gaussian_mean: Optional[float] = typer.Option(None, help="Gaussian mean per step."),
    sp_amount: Optional[float] = typer.Option(None, help="Salt-and-pepper corrupted fraction."),
    sp_salt_ratio: Optional[float] = typer.Option(None, help="Salt share of corrupted pixels."),
    margin: float = typer.Option(0.15, min=0.0, max=0.49, help="Clean pre-scale headroom against clipping bias."),
    grayscale: bool = typer.Option(True, "--grayscale/--rgb", help="Convert clean sources before noising."),
    max_side: Optional[int] = typer.Option(512, help="Downscale clean sources above this size (never the noisy frames)."),
    select_seed: int = typer.Option(0, help="Seed for the source subset."),
    noise_seed: int = typer.Option(0, help="Seed for the noise draws."),
    overwrite: bool = typer.Option(False, help="Replace an existing output directory."),
) -> None:
    """Generate a synthetic burst dataset and verify it is noisy enough."""
    from .generate import generate_burst_dataset

    noise_params: dict[str, dict[str, float]] = {}
    if peak is not None:
        noise_params.setdefault("poisson", {})["peak"] = peak
    if gaussian_std is not None:
        noise_params.setdefault("gaussian", {})["std"] = gaussian_std
    if gaussian_mean is not None:
        noise_params.setdefault("gaussian", {})["mean"] = gaussian_mean
    if sp_amount is not None:
        noise_params.setdefault("salt_pepper", {})["amount"] = sp_amount
    if sp_salt_ratio is not None:
        noise_params.setdefault("salt_pepper", {})["salt_ratio"] = sp_salt_ratio

    stats_path = generate_burst_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        num_sources=num_sources,
        replicas=replicas,
        noise_type=list(noise_type),
        steps=steps,
        noise_params=noise_params or None,
        margin=margin,
        grayscale=grayscale,
        max_side=max_side,
        select_seed=select_seed,
        noise_seed=noise_seed,
        overwrite=overwrite,
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    aggregate = stats["aggregate"]
    typer.echo(
        f"single-frame PSNR median {aggregate['single_frame_psnr_median']:.2f} dB | "
        f"avg-of-N PSNR median {aggregate['avg_of_n_psnr_median']:.2f} dB | "
        f"|bias| median {aggregate['bias_abs_median']:.5f}"
    )
    for message in stats["warnings"]:
        typer.echo(f"WARNING: {message}", err=True)
    typer.echo(f"stats written to {stats_path}")


@app.command()
def preview(
    dataset: Path = typer.Option(..., help="Burst dataset directory."),
    source_index: int = typer.Option(0, min=0, help="Which source to render."),
    out: Path = typer.Option(..., help="Output PNG path."),
    avg_counts: str = typer.Option("1,2,4,8,16", help="Comma-separated averaging counts."),
    max_tile: int = typer.Option(256, min=16, help="Center-crop tiles to this size."),
) -> None:
    """Render the clean / single-frames / running-averages grid for one source."""
    from .preview import make_preview_grid

    path = make_preview_grid(
        dataset,
        source_index=source_index,
        out_path=out,
        avg_counts=_parse_int_list(avg_counts),
        max_tile=max_tile,
    )
    typer.echo(f"preview written to {path}")


@app.command()
def train(
    config: Path = typer.Option(..., help="YAML config path."),
    resume: bool = typer.Option(False, help="Resume from <run_dir>/ckpt_latest.pt."),
    resume_from: Optional[Path] = typer.Option(None, help="Resume from a specific checkpoint."),
) -> None:
    """Train a burst-diffusion model as described by a config file."""
    from .config import load_config
    from .train import LATEST_CHECKPOINT_NAME, Trainer

    loaded = load_config(config)
    checkpoint: Path | None = resume_from
    if resume and checkpoint is None:
        checkpoint = Path(loaded.training.run_dir) / LATEST_CHECKPOINT_NAME
    if checkpoint is not None and not checkpoint.is_file():
        raise typer.BadParameter(f"resume checkpoint not found: {checkpoint}")
    trainer = Trainer(loaded, resume_from=checkpoint)
    final = trainer.run()
    typer.echo(f"finished at step {trainer.step}; latest checkpoint: {final}")


@app.command()
def sample(
    checkpoint: Path = typer.Option(..., help="Checkpoint (.pt) to sample from."),
    out: Path = typer.Option(..., help="Output directory for PNGs."),
    input: Optional[Path] = typer.Option(None, help="A noisy measurement image (center-cropped)."),
    dataset: Optional[Path] = typer.Option(None, help="Burst dataset to pull a frame from."),
    source_index: int = typer.Option(0, min=0, help="Dataset source to denoise."),
    replica: int = typer.Option(0, min=0, help="Which burst frame is the measurement."),
    steps: Optional[int] = typer.Option(None, help="Accelerated sampling step count (default: all T)."),
    trajectory: bool = typer.Option(False, help="Also save the per-step trajectory sheet."),
    ema: bool = typer.Option(True, "--ema/--no-ema", help="Use the EMA weights."),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """Denoise one measurement with the iterative sampler."""
    from .data import resolve_burst_dir
    from .metrics import psnr
    from .sample import (
        Sampler,
        load_input_image,
        save_model_image,
        save_trajectory_sheet,
    )
    from .schedule import sampling_schedule

    if (input is None) == (dataset is None):
        raise typer.BadParameter("provide exactly one of --input or --dataset")
    sampler = Sampler.from_checkpoint(checkpoint, device=device, use_ema=ema)
    assert sampler.config is not None
    image_size = sampler.config.data.image_size
    channels = sampler.config.data.channels

    clean_path: Path | None = None
    if input is not None:
        measurement_path = input
        stem = input.stem
    else:
        burst_dir = resolve_burst_dir(dataset)
        measurement_path = burst_dir / "noisy" / f"{source_index:05d}_{replica:05d}.png"
        if not measurement_path.is_file():
            raise typer.BadParameter(f"no such burst frame: {measurement_path}")
        candidate = burst_dir / "clean" / f"{source_index:05d}.png"
        clean_path = candidate if candidate.is_file() else None
        stem = f"src{source_index:05d}_rep{replica:05d}"

    x_start = load_input_image(measurement_path, image_size=image_size, channels=channels)
    schedule = sampling_schedule(sampler.num_steps, steps)
    result = sampler.run(x_start, schedule=schedule, keep_trajectory=trajectory)

    save_model_image(x_start[0], out / f"{stem}_input.png")
    save_model_image(result.average[0], out / f"{stem}_average.png")
    save_model_image(result.prediction[0], out / f"{stem}_prediction.png")
    written = 3
    if trajectory and result.trajectory:
        save_trajectory_sheet(result.trajectory, out / f"{stem}_trajectory.png")
        written += 1
    typer.echo(f"wrote {written} PNG(s) to {out}")

    if clean_path is not None:
        clean = load_input_image(clean_path, image_size=image_size, channels=channels)
        clean01 = ((clean[0] + 1) / 2).numpy().transpose(1, 2, 0)
        for label, tensor in (
            ("input", x_start[0]),
            ("average", result.average[0]),
            ("prediction", result.prediction[0]),
        ):
            value01 = ((tensor.clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0)
            typer.echo(f"PSNR vs clean [{label}]: {psnr(clean01, value01):.2f} dB")


@app.command()
def evaluate(
    config: Path = typer.Option(..., help="YAML config path (dataset + schedule)."),
    checkpoint: Path = typer.Option(..., help="Checkpoint (.pt) to evaluate."),
    out: Path = typer.Option(..., help="Output directory for results.json + grid."),
    split: str = typer.Option("val", help="val | train | test (test is the locked holdout: report once)"),
    limit: Optional[int] = typer.Option(None, help="Evaluate at most this many sources."),
    steps: Optional[int] = typer.Option(None, help="Accelerated sampling step count."),
    tile: bool = typer.Option(False, help="Full-image evaluation via overlapping tiles."),
    device: Optional[str] = typer.Option(None, help="auto | cpu | cuda (default: config)."),
) -> None:
    """Compare the model against the classical baselines on held-out sources."""
    from .config import load_config
    from .evaluate import evaluate as run_evaluation

    results = run_evaluation(
        load_config(config),
        checkpoint,
        split=split,
        limit=limit,
        out_dir=out,
        sample_steps=steps,
        tile=tile,
        device=device,
    )
    for name, method in results["methods"].items():
        typer.echo(
            f"{name:>15}: PSNR {method['psnr_mean']:6.2f} dB | SSIM {method['ssim_mean']:.4f}"
        )
    typer.echo(f"results written to {Path(out) / 'results.json'}")


@app.command()
def repeatability(
    config: Path = typer.Option(..., help="YAML config path (dataset + schedule)."),
    checkpoint: list[str] = typer.Option(
        ...,
        help="Checkpoint arm as NAME=PATH (repeatable; a bare PATH is named 'model').",
    ),
    out: Path = typer.Option(..., help="Output directory for repeatability.json + summary.md."),
    split: str = typer.Option("val", help="val | train | test (test is the locked holdout: report once)"),
    limit: Optional[int] = typer.Option(None, help="Evaluate at most this many sources."),
    seeds: int = typer.Option(10, min=1, help="Fresh frames (seeds) per source."),
    steps: Optional[int] = typer.Option(None, help="Accelerated sampling step count."),
    device: Optional[str] = typer.Option(None, help="auto | cpu | cuda (default: config)."),
) -> None:
    """Measure per-method repeatability plus CD / registration metrology."""
    from .config import load_config
    from .repeatability import repeatability as run_repeatability

    checkpoints: dict[str, Path] = {}
    for entry in checkpoint:
        arm, _, path = entry.partition("=")
        if not path:
            arm, path = "model", entry
        if arm in checkpoints:
            raise typer.BadParameter(f"duplicate checkpoint arm {arm!r}")
        checkpoints[arm] = Path(path)

    results = run_repeatability(
        load_config(config),
        checkpoints,
        out_dir=out,
        split=split,
        limit=limit,
        num_seeds=seeds,
        sample_steps=steps,
        device=device,
        progress_callback=lambda done, total: typer.echo(f"source {done}/{total}", err=True),
    )
    for name, method in results["methods"].items():
        pixel = method["pixel_repeatability"]
        pixel_text = "-" if pixel is None else f"{pixel['sigma_mean'] * 1e3:6.2f}e-3"
        cd3 = method["cd"]["pooled_3sigma_px"]
        cd_text = "-" if cd3 is None else f"{cd3:6.3f} px"
        shift = method["registration"]["shift_sigma_px"]
        shift_text = "-" if shift is None else f"{shift:6.3f} px"
        typer.echo(
            f"{name:>26}: PSNR {method['accuracy']['psnr_mean']:6.2f} dB | "
            f"pixel sigma {pixel_text} | CD 3sigma {cd_text} | shift sigma {shift_text}"
        )
    typer.echo(f"results written to {Path(out) / 'repeatability.json'}")
