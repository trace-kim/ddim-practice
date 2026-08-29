"""Training loop for burst diffusion.

Follows the legacy DDIM recipe where it matters (Adam with beta2=0.999,
gradient clipping, per-step EMA) but with a mean- rather than sum-reduced MSE
(patch-size invariant; Adam is approximately loss-scale invariant), keyed
atomic checkpoints that restore *all* RNG state for exact resume, and no
DataLoader (see data.py).

Expectation setting: because the target is always a noisy frame, the training
loss converges to roughly the single-frame noise variance in model space
(4x the [0, 1]-space variance) plus estimation error -- a PLATEAU IS EXPECTED,
not a sign of divergence. Progress is measured by the validation PSNR of the
prediction against the clean image (``val/psnr_pred_t*`` in TensorBoard).
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .data import BatchFactory, BurstCache, ValidationBatch
from .ema import EMAHelper, ema_parameters
from .metrics import psnr
from .schedule import min_replicas
from .unet import build_unet

logger = logging.getLogger("burst_diffusion.train")

CHECKPOINT_FORMAT = 1
LATEST_CHECKPOINT_NAME = "ckpt_latest.pt"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is 'cuda' but CUDA is not available")
    return torch.device(requested)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: EMAHelper | None,
    optimizer: torch.optim.Optimizer,
    factory: BatchFactory,
    step: int,
    config: Config,
) -> None:
    """Atomically write a keyed checkpoint (tensors + primitives only)."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "config": config.model_dump(mode="json"),
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "factory": factory.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unrecognized checkpoint format in {path}; expected a burst_diffusion "
            f"format-{CHECKPOINT_FORMAT} checkpoint"
        )
    return payload


class Trainer:
    def __init__(self, config: Config, *, resume_from: str | Path | None = None):
        self.config = config
        self.device = resolve_device(config.training.device)
        self.run_dir = Path(config.training.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = self.run_dir / "stop"
        if self.stop_file.exists():
            logger.info("removing leftover stop file %s", self.stop_file)
            self.stop_file.unlink()

        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)

        self.cache = BurstCache(
            config.data.dataset_dir,
            channels=config.data.channels,
            min_replicas=min_replicas(config.schedule.num_steps),
            min_size=config.data.image_size,
            val_fraction=config.data.val_fraction,
            split_seed=config.data.split_seed,
        )
        summary = self.cache.summary()
        logger.info(
            "dataset: %d train / %d val sources, min %d frames, %.0f MB cached",
            summary["train_sources"],
            summary["val_sources"],
            summary["min_frames"],
            summary["ram_bytes"] / 1e6,
        )
        self.factory = BatchFactory(
            self.cache,
            num_steps=config.schedule.num_steps,
            image_size=config.data.image_size,
            batch_size=config.training.batch_size,
            target_mode=config.schedule.target_mode,
            antithetic=config.training.antithetic,
            seed=config.training.seed,
        )
        self.model = build_unet(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.lr,
            betas=(config.training.beta1, 0.999),
            eps=config.training.adam_eps,
            weight_decay=config.training.weight_decay,
        )
        self.ema: EMAHelper | None = None
        if config.training.ema:
            self.ema = EMAHelper(mu=config.training.ema_rate)
            self.ema.register(self.model)
        self.step = 0

        if resume_from is not None:
            self._restore(Path(resume_from))

        (self.run_dir / "config.yml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )

    def _restore(self, checkpoint_path: Path) -> None:
        payload = load_checkpoint(checkpoint_path, map_location=self.device)
        stored_config = Config.model_validate(payload["config"])
        if stored_config != self.config:
            warnings.warn(
                f"checkpoint {checkpoint_path} was written with a different config; "
                "resuming with the CURRENT config",
                stacklevel=2,
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.ema is not None:
            if payload["ema"] is None:
                raise ValueError("config enables EMA but the checkpoint has no EMA state")
            self.ema.load_state_dict(
                {name: value.to(self.device) for name, value in payload["ema"].items()}
            )
        torch.set_rng_state(payload["torch_rng"].cpu())
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
        self.factory.load_state_dict(payload["factory"])
        self.step = int(payload["step"])
        logger.info("resumed from %s at step %d", checkpoint_path, self.step)

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.run_dir / LATEST_CHECKPOINT_NAME

    def _save(self, *, milestone: bool) -> Path:
        save_checkpoint(
            self.latest_checkpoint_path,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            factory=self.factory,
            step=self.step,
            config=self.config,
        )
        if milestone:
            save_checkpoint(
                self.run_dir / f"ckpt_{self.step:07d}.pt",
                model=self.model,
                ema=self.ema,
                optimizer=self.optimizer,
                factory=self.factory,
                step=self.step,
                config=self.config,
            )
        return self.latest_checkpoint_path

    def _validation_levels(self) -> list[int]:
        num_steps = self.config.schedule.num_steps
        mid = (num_steps + 1) // 2
        return sorted({num_steps, max(1, mid)}, reverse=True)

    def _validate(self, writer: SummaryWriter) -> None:
        if not self.cache.val_sources:
            return
        count = self.config.training.val_images
        self.model.eval()
        with ema_parameters(self.model, self.ema), torch.no_grad():
            for level in self._validation_levels():
                batch: ValidationBatch = self.factory.val_batch(level=level, count=count)
                x_t = batch.x_t.to(self.device)
                prediction = self.model(x_t, batch.t.to(self.device))
                val_loss = F.mse_loss(prediction, batch.eps.to(self.device)).item()
                clean01 = ((batch.clean + 1.0) / 2.0).numpy()
                pred01 = ((prediction.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
                psnr_values = [
                    psnr(np.moveaxis(clean01[i], 0, -1), np.moveaxis(pred01[i], 0, -1))
                    for i in range(pred01.shape[0])
                ]
                writer.add_scalar(f"val/loss_t{level:02d}", val_loss, self.step)
                writer.add_scalar(
                    f"val/psnr_pred_t{level:02d}", float(np.mean(psnr_values)), self.step
                )
                if level == self.config.schedule.num_steps:
                    shown = min(4, pred01.shape[0])
                    input01 = ((batch.x_t.clamp(-1.0, 1.0) + 1.0) / 2.0).numpy()
                    rows = [
                        np.concatenate([input01[i], pred01[i], clean01[i]], axis=-1)
                        for i in range(shown)
                    ]
                    writer.add_image(
                        "val/input_pred_clean", np.concatenate(rows, axis=-2), self.step
                    )
        self.model.train()

    def run(self) -> Path:
        training = self.config.training
        writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        loss_by_t: dict[int, list[float]] = defaultdict(list)
        window_started = time.time()
        stop_reason: str | None = None
        self.model.train()
        try:
            while self.step < training.max_steps:
                if self.stop_file.exists():
                    stop_reason = f"stop file present: {self.stop_file}"
                    break
                batch = self.factory.sample_batch()
                x_t = batch.x_t.to(self.device)
                t = batch.t.to(self.device)
                eps = batch.eps.to(self.device)

                prediction = self.model(x_t, t)
                per_sample = ((prediction - eps) ** 2).mean(dim=(1, 2, 3))
                loss = per_sample.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), training.grad_clip)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.step += 1

                detached = per_sample.detach().cpu()
                for value, level in zip(detached.tolist(), batch.t.tolist()):
                    loss_by_t[int(level)].append(value)

                if self.step == 1 or self.step % training.log_every == 0:
                    elapsed = max(time.time() - window_started, 1e-9)
                    steps_in_window = sum(len(v) for v in loss_by_t.values()) / max(
                        training.batch_size, 1
                    )
                    scalar_loss = float(loss.item())
                    writer.add_scalar("train/loss", scalar_loss, self.step)
                    writer.add_scalar(
                        "train/steps_per_sec", steps_in_window / elapsed, self.step
                    )
                    for level, values in sorted(loss_by_t.items()):
                        writer.add_scalar(
                            f"train/loss_by_t/{level:02d}",
                            float(np.mean(values)),
                            self.step,
                        )
                    logger.info(
                        "step %d | loss %.5f | %.1f steps/s",
                        self.step,
                        scalar_loss,
                        steps_in_window / elapsed,
                    )
                    loss_by_t.clear()
                    window_started = time.time()

                if self.step % training.val_every == 0 or self.step == training.max_steps:
                    self._validate(writer)
                if self.step % training.checkpoint_every == 0 or self.step == training.max_steps:
                    self._save(milestone=True)
        except KeyboardInterrupt:
            stop_reason = "keyboard interrupt"
        finally:
            checkpoint = self._save(milestone=False)
            writer.close()
        if stop_reason is not None:
            logger.info("stopped early at step %d (%s)", self.step, stop_reason)
        else:
            logger.info("finished %d steps", self.step)
        return checkpoint
