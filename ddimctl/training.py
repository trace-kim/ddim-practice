"""Manifest-driven, reproducible single-GPU DDIM training.

This module deliberately does not depend on the command-line layer.  Both a
foreground launch and a scheduler worker call the same ``train_from_manifest``
entry point and therefore receive identical validation, checkpoint, and metric
semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from datasets import data_transform, get_dataset, inverse_data_transform
from functions import get_optimizer
from functions.losses import loss_registry
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import get_beta_schedule

from .checkpoints import (
    build_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from .run_logging import MetricLogger


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return dict(value.dict())
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    raise TypeError("Expected a mapping or model, got {}".format(type(value).__name__))


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return argparse.Namespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def stable_config_sha256(spec: Any) -> str:
    configured = getattr(spec, "config_sha256", None)
    if callable(configured):
        return str(configured())
    if isinstance(configured, str):
        return configured
    payload = _as_mapping(spec)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_legacy_config(spec: Any, dataset_path: os.PathLike[str] | str) -> argparse.Namespace:
    """Adapt the strict flat TrainingSpec to the original model/dataset API."""

    if hasattr(spec, "to_legacy_dict"):
        return _namespace(spec.to_legacy_dict(str(dataset_path)))
    values = _as_mapping(spec)

    def value(name: str, default: Any = None) -> Any:
        return values[name] if name in values else default

    legacy = {
        "data": {
            "dataset": "SEM",
            "data_path": str(dataset_path),
            "image_size": value("image_size"),
            "channels": value("channels", 1),
            "logit_transform": value("logit_transform", False),
            "uniform_dequantization": value("uniform_dequantization", False),
            "gaussian_dequantization": value("gaussian_dequantization", False),
            "random_flip": value("random_flip", False),
            "rescaled": value("rescaled", True),
            "num_workers": value("num_workers", 0),
            "cache_in_memory": value("cache_in_memory", False),
            "recursive": value("recursive", False),
            "validation_split": value("validation_split", 0.1),
            "split_seed": value("split_seed", 2019),
            "extensions": value("extensions", None),
        },
        "model": {
            "type": value("model_type", "simple"),
            "in_channels": value("in_channels", value("channels", 1)),
            "out_ch": value("out_channels", value("channels", 1)),
            "ch": value("model_ch", 64),
            "ch_mult": value("ch_mult", [1, 2, 2, 2]),
            "num_res_blocks": value("num_res_blocks", 2),
            "attn_resolutions": value("attn_resolutions", [16]),
            "dropout": value("dropout", 0.0),
            "var_type": value("var_type", "fixedlarge"),
            "ema_rate": value("ema_rate", 0.999),
            "ema": value("ema", True),
            "resamp_with_conv": value("resamp_with_conv", True),
        },
        "diffusion": {
            "beta_schedule": value("beta_schedule", "linear"),
            "beta_start": value("beta_start", 0.001),
            "beta_end": value("beta_end", 0.2),
            "num_diffusion_timesteps": value("diffusion_steps", 100),
        },
        "training": {
            "batch_size": value("batch_size", 8),
            "max_steps": value("max_steps"),
            "snapshot_freq": value("checkpoint_every", 2500),
            "validation_freq": value("validation_every", 2500),
            "sample_freq": value("sample_every", 2500),
        },
        "sampling": {
            "batch_size": value("sampling_batch_size", 8),
            "last_only": value("sampling_last_only", True),
        },
        "optim": {
            "weight_decay": value("weight_decay", 0.0),
            "optimizer": value("optimizer", "Adam"),
            "lr": value("lr", 0.0002),
            "beta1": value("beta1", 0.9),
            "amsgrad": value("amsgrad", False),
            "eps": value("eps", 1e-8),
            "grad_clip": value("grad_clip", 1.0),
        },
    }
    return _namespace(legacy)


def configure_reproducibility(seed: int, mode: str) -> None:
    mode = str(mode).lower().replace("_", "-")
    aliases = {"seeded": "repeatable", "seeded-repeatable": "repeatable", "deterministic": "strict"}
    mode = aliases.get(mode, mode)
    if mode not in {"repeatable", "strict", "performance"}:
        raise ValueError("reproducibility must be repeatable, strict, or performance")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if mode == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    elif mode == "repeatable":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def select_device(requested: str | torch.device | None = None) -> torch.device:
    if requested is None and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; refusing to start an accidentally CPU-bound training run. "
            "Use an explicit --device cpu only for a deliberate smoke/test invocation."
        )
    device = torch.device(requested) if requested is not None else torch.device("cuda")
    if device.type == "cuda":
        count = torch.cuda.device_count()
        if count != 1:
            raise RuntimeError(
                "The v1 worker requires exactly one visible CUDA GPU; found {}. "
                "Set CUDA_VISIBLE_DEVICES or request one scheduler GPU.".format(count)
            )
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    return device


class EpochSeededDataset(Dataset[Any]):
    """Make random transforms a pure function of seed, epoch, and sample index."""

    def __init__(self, dataset: Dataset[Any], seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def _sample_seed(self, index: int) -> int:
        digest = hashlib.blake2b(
            "{}:{}:{}".format(self.seed, self.epoch, int(index)).encode("ascii"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little") % (2**32)

    def __getitem__(self, index: int) -> Any:
        sample_seed = self._sample_seed(index)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            try:
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                return self.dataset[index]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, size: int, *, seed: int, epoch: int, start_index: int = 0):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = max(0, min(int(start_index), self.size))

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch * 1_000_003) % (2**63 - 1))
        permutation = torch.randperm(self.size, generator=generator).tolist()
        return iter(permutation[self.start_index :])

    def __len__(self) -> int:
        return self.size - self.start_index

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "start_index": self.start_index,
            "dataset_size": self.size,
        }


class StopController:
    def __init__(self, stop_file: os.PathLike[str] | str):
        import threading

        self.stop_file = Path(stop_file)
        self._requested = threading.Event()
        self.reason = "stop requested"

    def request(self, reason: str = "stop requested") -> None:
        self.reason = reason
        self._requested.set()

    def is_requested(self) -> bool:
        return self._requested.is_set() or self.stop_file.exists()


@dataclass(frozen=True)
class TrainingResult:
    status: str
    global_step: int
    epoch: int
    batch_in_epoch: int
    checkpoint: str | None
    best_validation_loss: float | None


@contextmanager
def ema_parameters(model: torch.nn.Module, ema: EMAHelper | None) -> Iterator[None]:
    if ema is None:
        yield
        return
    named = dict(model.named_parameters())
    # Swap storage references instead of cloning another full model on the GPU.
    # EMAHelper owns its shadow tensors and evaluation is read-only.
    backup = {name: parameter.data for name, parameter in named.items() if name in ema.shadow}
    try:
        for name, value in ema.shadow.items():
            named[name].data = value.data
        yield
    finally:
        for name, value in backup.items():
            named[name].data = value


class ModernTrainingRunner:
    def __init__(
        self,
        *,
        spec: Any,
        dataset_path: os.PathLike[str] | str,
        run_dir: os.PathLike[str] | str,
        run_id: str,
        config_sha256: str | None = None,
        device: str | torch.device | None = None,
        stop_controller: StopController | None = None,
        metric_logger: MetricLogger | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        model_factory: Callable[[Any], torch.nn.Module] = Model,
        dataset_factory: Callable[[Any, Any], tuple[Dataset[Any], Dataset[Any]]] = get_dataset,
        optimizer_factory: Callable[[Any, Any], torch.optim.Optimizer] = get_optimizer,
    ):
        self.spec = spec
        self.values = _as_mapping(spec)
        self.run_dir = Path(run_dir).resolve()
        self.run_id = str(run_id)
        self.config_sha256 = config_sha256 or stable_config_sha256(spec)
        self.device = select_device(device)
        self.config = build_legacy_config(spec, dataset_path)
        self.config.device = self.device
        self.stop_controller = stop_controller or StopController(self.run_dir / "stop.request")
        self.metric_logger = metric_logger
        self.progress_callback = progress_callback
        self.model_factory = model_factory
        self.dataset_factory = dataset_factory
        self.optimizer_factory = optimizer_factory
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.sample_dir = self.run_dir / "samples"
        self.train_dataset_size: int | None = None

        self.max_steps = int(self.values["max_steps"])
        self.batch_size = int(self.values["batch_size"])
        self.seed = int(self.values.get("seed", 1234))
        self.reproducibility = str(self.values.get("reproducibility", "repeatable"))
        self.checkpoint_every = int(self.values.get("checkpoint_every", 0) or 0)
        self.validation_every = int(self.values.get("validation_every", 0) or 0)
        self.sample_every = int(self.values.get("sample_every", 0) or 0)
        self.checkpoint_seconds = float(self.values.get("checkpoint_minutes", 30.0)) * 60.0
        self.num_workers = int(self.values.get("num_workers", 0))
        self.sampling_batch_size = int(self.values.get("sampling_batch_size", 8))

    def _make_train_loader(
        self, dataset: EpochSeededDataset, epoch: int, batch_in_epoch: int
    ) -> tuple[DataLoader[Any], ResumableEpochSampler]:
        dataset.set_epoch(epoch)
        start_index = batch_in_epoch * self.batch_size
        sampler = ResumableEpochSampler(
            len(dataset), seed=self.seed, epoch=epoch, start_index=start_index
        )
        loader_generator = torch.Generator().manual_seed(
            (self.seed + epoch * 2_000_003 + 17) % (2**63 - 1)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            generator=loader_generator,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
        )
        return loader, sampler

    def _make_validation_loader(self, dataset: Dataset[Any]) -> DataLoader[Any] | None:
        if len(dataset) == 0:
            return None
        generator = torch.Generator().manual_seed(self.seed + 29)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=generator,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
        )

    def _validation_loss(
        self,
        model: torch.nn.Module,
        ema: EMAHelper | None,
        loader: DataLoader[Any] | None,
        betas: torch.Tensor,
    ) -> float | None:
        if loader is None:
            return None
        was_training = model.training
        total = 0.0
        count = 0
        generator = torch.Generator().manual_seed(self.seed + 0x5EED)
        training_rng = capture_rng_state()
        model.eval()
        try:
            with ema_parameters(model, ema), torch.no_grad():
                for images, _ in loader:
                    images = data_transform(self.config, images.to(self.device, non_blocking=True))
                    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype).to(self.device)
                    timesteps = torch.randint(
                        0, len(betas), (images.size(0),), generator=generator, dtype=torch.long
                    ).to(self.device)
                    loss = loss_registry[self.config.model.type](
                        model, images, timesteps, noise, betas
                    )
                    total += float(loss.item()) * images.size(0)
                    count += images.size(0)
        finally:
            model.train(was_training)
            restore_rng_state(training_rng)
        return total / count if count else None

    @staticmethod
    def _ddim_sample(
        model: torch.nn.Module, noise: torch.Tensor, betas: torch.Tensor, sample_steps: int
    ) -> torch.Tensor:
        count = max(1, min(int(sample_steps), len(betas)))
        sequence = torch.linspace(0, len(betas) - 1, count).round().long().unique().tolist()
        alpha_bar = (1.0 - betas).cumprod(dim=0)
        current = noise
        with torch.no_grad():
            for position in reversed(range(len(sequence))):
                timestep = sequence[position]
                next_timestep = sequence[position - 1] if position > 0 else -1
                t = torch.full(
                    (current.size(0),), timestep, device=current.device, dtype=torch.float32
                )
                alpha = alpha_bar[timestep]
                predicted_noise = model(current, t)
                predicted_clean = (current - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
                if next_timestep < 0:
                    current = predicted_clean
                else:
                    next_alpha = alpha_bar[next_timestep]
                    current = next_alpha.sqrt() * predicted_clean + (1.0 - next_alpha).sqrt() * predicted_noise
        return current

    def _fixed_noise(self) -> torch.Tensor:
        path = self.sample_dir / "fixed_noise.pt"
        expected_shape = (
            self.sampling_batch_size,
            int(self.values.get("channels", 1)),
            int(self.values["image_size"]),
            int(self.values["image_size"]),
        )
        if path.exists():
            try:
                noise = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                noise = torch.load(path, map_location="cpu")
            if tuple(noise.shape) != expected_shape:
                raise RuntimeError(
                    "Persisted fixed noise has shape {}, expected {}".format(
                        tuple(noise.shape), expected_shape
                    )
                )
            return noise
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(self.seed + 0xD01FF)
        noise = torch.randn(expected_shape, generator=generator)
        temporary = path.with_name(".fixed_noise.{}.tmp".format(os.getpid()))
        try:
            with temporary.open("wb") as handle:
                torch.save(noise, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return noise

    def _generate_samples(
        self, model: torch.nn.Module, ema: EMAHelper | None, betas: torch.Tensor, step: int
    ) -> Path:
        import torchvision.utils as tvu

        self.sample_dir.mkdir(parents=True, exist_ok=True)
        fixed_noise = self._fixed_noise().to(self.device)
        was_training = model.training
        model.eval()
        with ema_parameters(model, ema):
            generated = self._ddim_sample(model, fixed_noise, betas, min(50, len(betas)))
        model.train(was_training)
        generated = inverse_data_transform(self.config, generated.detach().cpu())
        destination = self.sample_dir / "step_{:09d}.png".format(step)
        tvu.save_image(generated, destination, nrow=max(1, int(math.sqrt(len(generated)))))
        if self.metric_logger is not None:
            grid = tvu.make_grid(generated, nrow=max(1, int(math.sqrt(len(generated)))))
            self.metric_logger.add_image("validation/fixed_noise_ema", grid, step)
        return destination

    def _save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: EMAHelper | None,
        global_step: int,
        epoch: int,
        batch_in_epoch: int,
        best_validation_loss: float | None,
        milestone: bool,
        is_best: bool = False,
        reason: str,
    ) -> Path:
        payload = build_checkpoint(
            model=model,
            optimizer=optimizer,
            ema_state=ema.state_dict() if ema is not None else None,
            global_step=global_step,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            sampler_state={
                "seed": self.seed,
                "epoch": epoch,
                "start_index": batch_in_epoch * self.batch_size,
                "dataset_size": self.train_dataset_size,
            },
            config_sha256=self.config_sha256,
            run_id=self.run_id,
            extra={"best_validation_loss": best_validation_loss, "reason": reason},
        )
        return save_checkpoint(
            self.checkpoint_dir,
            payload,
            milestone=milestone,
            is_best=is_best,
        )

    def run(self, resume: os.PathLike[str] | str | None = None) -> TrainingResult:
        configure_reproducibility(self.seed, self.reproducibility)
        args = argparse.Namespace(exp=str(self.run_dir))
        train_base, validation_dataset = self.dataset_factory(args, self.config)
        if len(train_base) == 0:
            raise RuntimeError("Training dataset is empty")
        self.train_dataset_size = len(train_base)
        train_dataset = EpochSeededDataset(train_base, self.seed)
        validation_loader = self._make_validation_loader(validation_dataset)

        model = self.model_factory(self.config).to(self.device)
        optimizer = self.optimizer_factory(self.config, model.parameters())
        ema: EMAHelper | None = None
        if bool(self.values.get("ema", True)):
            ema = EMAHelper(mu=float(self.values.get("ema_rate", 0.999)))
            ema.register(model)

        betas = torch.from_numpy(
            get_beta_schedule(
                self.config.diffusion.beta_schedule,
                beta_start=self.config.diffusion.beta_start,
                beta_end=self.config.diffusion.beta_end,
                num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps,
            )
        ).float().to(self.device)

        epoch = 0
        batch_in_epoch = 0
        global_step = 0
        best_validation_loss: float | None = None
        last_checkpoint: Path | None = None
        if resume is not None:
            checkpoint, loaded_path = load_checkpoint(
                resume,
                map_location=self.device,
                expected_run_id=self.run_id,
                expected_config_sha256=self.config_sha256,
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            epoch = int(checkpoint["epoch"])
            batch_in_epoch = int(checkpoint["batch_in_epoch"])
            global_step = int(checkpoint["global_step"])
            if global_step > self.max_steps:
                raise RuntimeError("Checkpoint step exceeds configured max_steps")
            if ema is not None:
                if checkpoint.get("ema_state") is None:
                    raise RuntimeError("EMA is enabled but the checkpoint has no EMA state")
                ema.load_state_dict(checkpoint["ema_state"])
            best_validation_loss = checkpoint.get("extra", {}).get("best_validation_loss")
            restore_rng_state(checkpoint["rng_state"])
            last_checkpoint = loaded_path

        if self.metric_logger is not None:
            self.metric_logger.log(
                "lifecycle",
                global_step,
                {"status": "started", "epoch": epoch, "batch_in_epoch": batch_in_epoch},
            )

        smoothed_loss: float | None = None
        last_step_time = time.perf_counter()
        last_checkpoint_time = time.monotonic()
        while global_step < self.max_steps:
            if self.stop_controller.is_requested():
                last_checkpoint = self._save(
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    best_validation_loss=best_validation_loss,
                    milestone=False,
                    reason="stop",
                )
                return TrainingResult(
                    "interrupted", global_step, epoch, batch_in_epoch, str(last_checkpoint), best_validation_loss
                )

            train_loader, sampler = self._make_train_loader(train_dataset, epoch, batch_in_epoch)
            if len(sampler) == 0:
                epoch += 1
                batch_in_epoch = 0
                continue
            data_started = time.perf_counter()
            for images, _ in train_loader:
                data_seconds = time.perf_counter() - data_started
                model.train()
                images = data_transform(self.config, images.to(self.device, non_blocking=True))
                noise = torch.randn_like(images)
                count = images.size(0)
                half = torch.randint(0, len(betas), (count // 2 + 1,), device=self.device)
                timesteps = torch.cat([half, len(betas) - half - 1], dim=0)[:count]
                loss = loss_registry[self.config.model.type](
                    model, images, timesteps, noise, betas
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss at step {}: {}".format(global_step + 1, loss))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(self.values.get("grad_clip", 0.0) or 0.0)
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    norms = [parameter.grad.detach().norm(2) for parameter in model.parameters() if parameter.grad is not None]
                    grad_norm = torch.stack(norms).norm(2) if norms else torch.tensor(0.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

                global_step += 1
                batch_in_epoch += 1
                now = time.perf_counter()
                elapsed = max(now - last_step_time, 1e-9)
                last_step_time = now
                scalar_loss = float(loss.item())
                smoothed_loss = scalar_loss if smoothed_loss is None else 0.98 * smoothed_loss + 0.02 * scalar_loss
                values = {
                    "loss": scalar_loss,
                    "smoothed_loss": smoothed_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(grad_norm),
                    "samples_per_second": count / elapsed,
                    "data_seconds": data_seconds,
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "gpu_memory_allocated_bytes": (
                        torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
                    ),
                    "gpu_memory_reserved_bytes": (
                        torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
                    ),
                }
                if self.metric_logger is not None:
                    self.metric_logger.log("train", global_step, values)
                if self.progress_callback is not None:
                    self.progress_callback({"global_step": global_step, "epoch": epoch, "batch_in_epoch": batch_in_epoch})

                validation_loss: float | None = None
                new_best = False
                if self.validation_every and global_step % self.validation_every == 0:
                    validation_loss = self._validation_loss(model, ema, validation_loader, betas)
                    if validation_loss is not None:
                        new_best = best_validation_loss is None or validation_loss < best_validation_loss
                        if new_best:
                            best_validation_loss = validation_loss
                        if self.metric_logger is not None:
                            self.metric_logger.log(
                                "validation", global_step, {"loss": validation_loss, "best": new_best}
                            )

                if self.sample_every and global_step % self.sample_every == 0:
                    sample_path = self._generate_samples(model, ema, betas, global_step)
                    if self.metric_logger is not None:
                        self.metric_logger.log("sample", global_step, {"path": str(sample_path.relative_to(self.run_dir))})

                step_checkpoint = bool(
                    self.checkpoint_every and global_step % self.checkpoint_every == 0
                )
                time_checkpoint = bool(
                    self.checkpoint_seconds > 0
                    and time.monotonic() - last_checkpoint_time >= self.checkpoint_seconds
                )
                if step_checkpoint or time_checkpoint or new_best:
                    last_checkpoint = self._save(
                        model=model,
                        optimizer=optimizer,
                        ema=ema,
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                        best_validation_loss=best_validation_loss,
                        milestone=step_checkpoint,
                        is_best=new_best,
                        reason="step" if step_checkpoint else ("best" if new_best else "elapsed-time"),
                    )
                    last_checkpoint_time = time.monotonic()

                if self.stop_controller.is_requested():
                    last_checkpoint = self._save(
                        model=model,
                        optimizer=optimizer,
                        ema=ema,
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                        best_validation_loss=best_validation_loss,
                        milestone=False,
                        reason="stop",
                    )
                    if self.metric_logger is not None:
                        self.metric_logger.log("lifecycle", global_step, {"status": "interrupted"})
                    return TrainingResult(
                        "interrupted", global_step, epoch, batch_in_epoch, str(last_checkpoint), best_validation_loss
                    )

                if global_step >= self.max_steps:
                    break
                data_started = time.perf_counter()

            if global_step < self.max_steps and batch_in_epoch * self.batch_size >= len(train_dataset):
                epoch += 1
                batch_in_epoch = 0

        if last_checkpoint is None or int(last_checkpoint.stem.rsplit("_", 1)[-1]) != global_step:
            last_checkpoint = self._save(
                model=model,
                optimizer=optimizer,
                ema=ema,
                global_step=global_step,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                best_validation_loss=best_validation_loss,
                milestone=True,
                reason="completed",
            )
        if self.metric_logger is not None:
            self.metric_logger.log("lifecycle", global_step, {"status": "completed"})
        return TrainingResult(
            "completed", global_step, epoch, batch_in_epoch, str(last_checkpoint), best_validation_loss
        )


def _manifest_field(manifest: Any, name: str, default: Any = None) -> Any:
    if isinstance(manifest, Mapping):
        return manifest.get(name, default)
    return getattr(manifest, name, default)


def find_dataset_path(manifest: Any) -> Path:
    """Resolve the already-snapshotted dataset path without consulting profiles."""

    direct = _manifest_field(manifest, "dataset_path")
    if direct:
        return Path(direct)
    dataset = _manifest_field(manifest, "dataset", {})
    dataset_values = _as_mapping(dataset) if dataset else {}
    for key in ("path", "dataset_path", "resolved_path", "root"):
        if dataset_values.get(key):
            return Path(dataset_values[key])
    machine = _manifest_field(manifest, "machine", {})
    machine_values = _as_mapping(machine) if machine else {}
    aliases = machine_values.get("dataset_aliases") or machine_values.get("datasets") or {}
    training = _manifest_field(manifest, "training")
    training_values = _as_mapping(training)
    alias = training_values.get("dataset_alias")
    if alias and isinstance(aliases, Mapping) and aliases.get(alias):
        return Path(aliases[alias])
    raise ValueError("Run manifest does not contain a resolved dataset path")


def train_from_manifest(
    manifest: Any,
    run_dir: os.PathLike[str] | str,
    *,
    resume: os.PathLike[str] | str | None = None,
    stop_controller: StopController | None = None,
    metric_logger: MetricLogger | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    device: str | torch.device | None = None,
) -> TrainingResult:
    training = _manifest_field(manifest, "training")
    if training is None:
        raise ValueError("Run manifest has no training specification")
    runner = ModernTrainingRunner(
        spec=training,
        dataset_path=find_dataset_path(manifest),
        run_dir=run_dir,
        run_id=str(_manifest_field(manifest, "run_id")),
        config_sha256=str(
            _manifest_field(manifest, "config_sha256", None) or stable_config_sha256(training)
        ),
        device=device,
        stop_controller=stop_controller,
        metric_logger=metric_logger,
        progress_callback=progress_callback,
    )
    machine = _manifest_field(manifest, "machine", {})
    machine_values = _as_mapping(machine) if machine else {}
    expected_gpu = machine_values.get("expected_gpu")
    if expected_gpu:
        if runner.device.type != "cuda":
            raise RuntimeError(f"expected GPU {expected_gpu!r}, but the selected device is CPU")
        actual_gpu = torch.cuda.get_device_name(runner.device)
        if str(expected_gpu).casefold() not in actual_gpu.casefold():
            raise RuntimeError(
                f"allocated GPU {actual_gpu!r} does not match expected profile value {expected_gpu!r}"
            )
    return runner.run(resume=resume)
