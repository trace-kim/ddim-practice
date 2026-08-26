"""Create deterministic paired microscopy datasets without DDIM dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
from PIL import Image
import requests


BBBC038_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
BBBC038_SHA256 = "dcb6edc2690f137406638b2309581a71522c4dff19157d118453b448dcddcb68"
BBBC038_IMAGE_COUNT = 670

_BBBC038_ARCHIVE_NAME = ".BBBC038v1-stage1_train.zip"
_SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_NOISE_DEFAULTS: dict[str, dict[str, float]] = {
    "gaussian": {"mean": 0.0, "std": 0.01},
    "poisson": {"peak": 10000.0},
    "salt_pepper": {"amount": 0.001, "salt_ratio": 0.5},
}


class _ProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        source_count: int,
        replica_count: int,
        steps: int,
        noise_types: Sequence[str],
        step_mode: str,
        interval: float,
    ) -> None:
        self.enabled = enabled
        self.source_count = source_count
        self.replica_count = replica_count
        self.total = source_count * replica_count
        self.interval = interval
        self.completed = 0
        self.started_at = time.perf_counter()
        self.last_reported_at = self.started_at
        if enabled:
            noises = ",".join(noise_types)
            self._emit(
                f"Starting {source_count} clean images x {replica_count} replicas "
                f"= {self.total} noisy outputs; steps={steps}, "
                f"mode={step_mode}, noise={noises}"
            )

    def advance(self, source_index: int, replica_index: int) -> None:
        self.completed += 1
        if not self.enabled:
            return
        now = time.perf_counter()
        if self.completed != 1 and now - self.last_reported_at < self.interval:
            return
        elapsed = max(now - self.started_at, 1e-9)
        rate = self.completed / elapsed
        remaining = (self.total - self.completed) / rate
        percentage = 100.0 * self.completed / self.total
        self._emit(
            f"{self.completed}/{self.total} noisy images ({percentage:.1f}%); "
            f"source {source_index + 1}/{self.source_count}, "
            f"replica {replica_index + 1}/{self.replica_count}; "
            f"{rate:.2f} images/s; elapsed {_format_duration(elapsed)}; "
            f"ETA {_format_duration(remaining)}"
        )
        self.last_reported_at = now

    def failed(self) -> None:
        if self.enabled:
            elapsed = time.perf_counter() - self.started_at
            self._emit(
                f"Stopped after {self.completed}/{self.total} noisy images; "
                f"elapsed {_format_duration(elapsed)}"
            )

    def finished(self) -> None:
        if self.enabled:
            elapsed = max(time.perf_counter() - self.started_at, 1e-9)
            rate = self.completed / elapsed
            self._emit(
                f"Completed {self.completed}/{self.total} noisy images in "
                f"{_format_duration(elapsed)} ({rate:.2f} images/s)"
            )

    @staticmethod
    def _emit(message: str) -> None:
        print(f"[noising_pipeline] {message}", file=sys.stderr, flush=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def create_noisy_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    n: int,
    steps: int,
    noise_type: str | Sequence[str],
    *,
    download: Literal["bbbc038"] | None = None,
    seed: int = 0,
    noise_params: Mapping[str, Mapping[str, float]] | None = None,
    source_license: str | None = None,
    overwrite: bool | None = None,
    step_mode: Literal["fused", "iterative"] = "fused",
    progress: bool = True,
    progress_interval: float = 5.0,
) -> Path:
    """Create clean/noisy PNG pairs and return the resulting JSONL manifest.

    Each source image is decoded once and saved losslessly as a clean PNG. Every
    one of its ``n`` noisy replicas starts from that same decoded image. By
    default, repeated step strength is fused analytically into one vectorized
    application of each selected distribution. ``step_mode="iterative"`` keeps
    the older, slower application loop for compatibility.

    If ``output_dir`` exists, ``overwrite=None`` prompts before replacing it.
    Pass ``True`` to replace without prompting or ``False`` to reject it without
    prompting. The output must not contain, or be contained by, ``source_dir``.
    When ``download`` is ``"bbbc038"``, a verified copy of BBBC038v1 is
    downloaded to (or reused from) ``source_dir``.
    """

    replica_count = _positive_integer(n, "n")
    step_count = _positive_integer(steps, "steps")
    base_seed = _nonnegative_integer(seed, "seed")
    noise_types = _normalize_noise_types(noise_type)
    resolved_params = _resolve_noise_params(noise_types, noise_params)
    if step_mode not in {"fused", "iterative"}:
        raise ValueError("step_mode must be 'fused' or 'iterative'")
    effective_params = _fuse_noise_params(resolved_params, step_count)
    if overwrite is not None and not isinstance(overwrite, bool):
        raise TypeError("overwrite must be True, False, or None")
    if not isinstance(progress, bool):
        raise TypeError("progress must be True or False")
    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, Real)
        or not math.isfinite(float(progress_interval))
        or progress_interval <= 0
    ):
        raise ValueError("progress_interval must be a positive finite number")
    source_path = _coerce_path(source_dir, "source_dir").resolve()
    output_path = _coerce_path(output_dir, "output_dir").absolute()
    resolved_output_path = output_path.resolve()

    if download not in (None, "bbbc038"):
        raise ValueError("download must be None or 'bbbc038'")
    if source_license is not None and not isinstance(source_license, str):
        raise TypeError("source_license must be a string or None")

    _reject_overlapping_paths(source_path, resolved_output_path)
    replace_existing = False
    if _path_exists(output_path):
        _validate_overwrite_target(source_path, output_path)
        replace_existing = _confirm_overwrite(output_path, overwrite)

    if download == "bbbc038":
        source_files = _ensure_bbbc038(source_path)
        dataset = "BBBC038v1-stage1-train"
        license_name: str | None = "CC0"
    else:
        if not source_path.exists():
            raise FileNotFoundError(f"source_dir does not exist: {source_path}")
        if not source_path.is_dir():
            raise NotADirectoryError(f"source_dir is not a directory: {source_path}")
        source_files = _find_source_images(source_path)
        dataset = "local"
        license_name = source_license

    if not source_files:
        raise ValueError(f"source_dir contains no supported images: {source_path}")

    if replace_existing:
        _remove_existing_path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    _generate_dataset(
        source_path=source_path,
        output_path=output_path,
        source_files=source_files,
        replica_count=replica_count,
        step_count=step_count,
        noise_types=noise_types,
        resolved_params=resolved_params,
        effective_params=effective_params,
        base_seed=base_seed,
        dataset=dataset,
        license_name=license_name,
        step_mode=step_mode,
        progress=progress,
        progress_interval=float(progress_interval),
    )
    return output_path / "manifest.jsonl"


def _generate_dataset(
    *,
    source_path: Path,
    output_path: Path,
    source_files: Sequence[Path],
    replica_count: int,
    step_count: int,
    noise_types: Sequence[str],
    resolved_params: Mapping[str, Mapping[str, float]],
    effective_params: Mapping[str, Mapping[str, float]],
    base_seed: int,
    dataset: str,
    license_name: str | None,
    step_mode: str,
    progress: bool,
    progress_interval: float,
) -> None:
    clean_dir = output_path / "clean"
    noisy_dir = output_path / "noisy"
    clean_dir.mkdir()
    noisy_dir.mkdir()
    manifest_path = output_path / "manifest.jsonl"
    reporter = _ProgressReporter(
        enabled=progress,
        source_count=len(source_files),
        replica_count=replica_count,
        steps=step_count,
        noise_types=noise_types,
        step_mode=step_mode,
        interval=progress_interval,
    )

    try:
        with manifest_path.open(
            "x", encoding="utf-8", newline="\n", buffering=1
        ) as manifest:
            for source_index, source_file in enumerate(source_files):
                clean_array, bit_depth = _load_image(source_file)
                clean_name = f"{source_index:05d}.png"
                clean_relative = Path("clean") / clean_name
                _save_png(clean_array, bit_depth, clean_dir / clean_name)
                normalized = _normalize_image(clean_array, bit_depth)

                try:
                    source_relative = source_file.relative_to(source_path).as_posix()
                except ValueError:
                    source_relative = source_file.name

                for replica_index in range(replica_count):
                    seed_sequence = np.random.SeedSequence(
                        [base_seed, source_index, replica_index]
                    )
                    sample_seed = int(
                        seed_sequence.generate_state(1, dtype=np.uint64)[0]
                    )
                    rng = np.random.default_rng(seed_sequence)
                    noisy = normalized.copy()

                    if step_mode == "iterative":
                        for _ in range(step_count):
                            for selected_noise in noise_types:
                                noisy = _apply_noise(
                                    noisy,
                                    selected_noise,
                                    resolved_params[selected_noise],
                                    rng,
                                )
                                noisy = np.clip(noisy, 0.0, 1.0).astype(
                                    np.float32, copy=False
                                )
                    else:
                        for selected_noise in noise_types:
                            noisy = _apply_noise(
                                noisy,
                                selected_noise,
                                effective_params[selected_noise],
                                rng,
                            )
                            noisy = np.clip(noisy, 0.0, 1.0).astype(
                                np.float32, copy=False
                            )

                    noisy_name = f"{source_index:05d}_{replica_index:05d}.png"
                    noisy_relative = Path("noisy") / noisy_name
                    _save_png(
                        _denormalize_image(noisy, bit_depth),
                        bit_depth,
                        noisy_dir / noisy_name,
                    )
                    row = {
                        "bit_depth": bit_depth,
                        "clean_path": clean_relative.as_posix(),
                        "dataset": dataset,
                        "effective_noise_params": effective_params,
                        "license": license_name,
                        "noise_params": resolved_params,
                        "noise_types": list(noise_types),
                        "noisy_path": noisy_relative.as_posix(),
                        "replica_index": replica_index,
                        "sample_seed": sample_seed,
                        "shape": list(clean_array.shape),
                        "source_index": source_index,
                        "source_path": source_relative,
                        "step_mode": step_mode,
                        "steps": step_count,
                    }
                    manifest.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    reporter.advance(source_index, replica_index)
    except BaseException:
        reporter.failed()
        raise
    reporter.finished()


def _coerce_path(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a string or path-like object")
    return Path(value).expanduser()


def _path_exists(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return (
        path.exists()
        or path.is_symlink()
        or (is_junction is not None and is_junction())
    )


def _confirm_overwrite(path: Path, overwrite: bool | None) -> bool:
    if overwrite is True:
        return True

    message = f"output_dir already exists: {path}"
    if overwrite is False:
        raise FileExistsError(message)

    try:
        response = input(
            f"{message}\nOverwrite it and delete all existing contents? [y/N]: "
        )
    except EOFError as exc:
        raise FileExistsError(
            f"{message}; unable to prompt, so pass overwrite=True to replace it"
        ) from exc

    if response.strip().casefold() in {"y", "yes"}:
        return True
    raise FileExistsError(f"{message}; overwrite declined")


def _validate_overwrite_target(source_dir: Path, output_dir: Path) -> None:
    resolved = output_dir.resolve()
    _reject_overlapping_paths(source_dir, resolved)
    protected_paths = {Path.cwd().resolve(), Path.home().resolve()}
    if resolved.parent == resolved or any(
        resolved == protected or resolved in protected.parents
        for protected in protected_paths
    ):
        raise ValueError(f"refusing to overwrite protected directory: {resolved}")
    if resolved.is_mount():
        raise ValueError(f"refusing to overwrite mounted directory: {resolved}")


def _remove_existing_path(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_symlink():
        path.unlink()
        return
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        path.rmdir()
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _normalize_noise_types(noise_type: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(noise_type, str):
        supplied = [noise_type]
    elif isinstance(noise_type, Sequence) and not isinstance(
        noise_type, (bytes, bytearray)
    ):
        supplied = list(noise_type)
    else:
        raise TypeError("noise_type must be a string or a sequence of strings")

    if not supplied:
        raise ValueError("noise_type must contain at least one noise distribution")

    normalized: list[str] = []
    for name in supplied:
        if not isinstance(name, str):
            raise TypeError("every noise_type entry must be a string")
        canonical = name.strip().casefold()
        if canonical not in _NOISE_DEFAULTS:
            supported = ", ".join(_NOISE_DEFAULTS)
            raise ValueError(f"unknown noise type {name!r}; expected one of: {supported}")
        normalized.append(canonical)
    return tuple(normalized)


def _resolve_noise_params(
    noise_types: Sequence[str],
    overrides: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    selected = tuple(dict.fromkeys(noise_types))
    resolved = {name: dict(_NOISE_DEFAULTS[name]) for name in selected}
    if overrides is None:
        return resolved
    if not isinstance(overrides, Mapping):
        raise TypeError("noise_params must be a mapping or None")

    normalized_overrides: dict[str, Mapping[str, float]] = {}
    for supplied_name, supplied_params in overrides.items():
        if not isinstance(supplied_name, str):
            raise TypeError("noise_params keys must be strings")
        canonical = supplied_name.strip().casefold()
        if canonical not in _NOISE_DEFAULTS:
            raise ValueError(f"noise_params contains unknown noise type {supplied_name!r}")
        if canonical not in resolved:
            raise ValueError(
                f"noise_params was provided for unselected noise type {supplied_name!r}"
            )
        if canonical in normalized_overrides:
            raise ValueError(f"duplicate noise_params entry for {canonical!r}")
        if not isinstance(supplied_params, Mapping):
            raise TypeError(f"noise_params[{supplied_name!r}] must be a mapping")
        normalized_overrides[canonical] = supplied_params

    for noise_name, supplied_params in normalized_overrides.items():
        unknown = set(supplied_params) - set(_NOISE_DEFAULTS[noise_name])
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"unknown {noise_name} parameter(s): {names}")
        for parameter_name, value in supplied_params.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"{noise_name}.{parameter_name} must be a real number"
                )
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(
                    f"{noise_name}.{parameter_name} must be finite"
                )
            resolved[noise_name][parameter_name] = converted

    _validate_resolved_params(resolved)
    return resolved


def _fuse_noise_params(
    params: Mapping[str, Mapping[str, float]], steps: int
) -> dict[str, dict[str, float]]:
    fused: dict[str, dict[str, float]] = {}
    for noise_name, values in params.items():
        if noise_name == "gaussian":
            mean = values["mean"] * steps
            std = values["std"] * math.sqrt(steps)
            if not math.isfinite(mean) or not math.isfinite(std):
                raise ValueError("steps make the effective Gaussian parameters overflow")
            fused[noise_name] = {"mean": mean, "std": std}
        elif noise_name == "poisson":
            peak = values["peak"] / steps
            if peak <= 0.0 or not math.isfinite(peak):
                raise ValueError("steps make the effective Poisson peak invalid")
            fused[noise_name] = {"peak": peak}
        elif noise_name == "salt_pepper":
            amount = values["amount"]
            if amount == 0.0:
                effective_amount = 0.0
            elif amount == 1.0:
                effective_amount = 1.0
            else:
                effective_amount = -math.expm1(steps * math.log1p(-amount))
            fused[noise_name] = {
                "amount": effective_amount,
                "salt_ratio": values["salt_ratio"],
            }
        else:
            raise AssertionError(f"unhandled noise type: {noise_name}")
    return fused


def _validate_resolved_params(params: Mapping[str, Mapping[str, float]]) -> None:
    if "gaussian" in params and params["gaussian"]["std"] < 0.0:
        raise ValueError("gaussian.std must be greater than or equal to zero")
    if "poisson" in params and params["poisson"]["peak"] <= 0.0:
        raise ValueError("poisson.peak must be greater than zero")
    if "salt_pepper" in params:
        amount = params["salt_pepper"]["amount"]
        salt_ratio = params["salt_pepper"]["salt_ratio"]
        if not 0.0 <= amount <= 1.0:
            raise ValueError("salt_pepper.amount must be between zero and one")
        if not 0.0 <= salt_ratio <= 1.0:
            raise ValueError("salt_pepper.salt_ratio must be between zero and one")


def _reject_overlapping_paths(source_dir: Path, output_dir: Path) -> None:
    if (
        source_dir == output_dir
        or source_dir in output_dir.parents
        or output_dir in source_dir.parents
    ):
        raise ValueError("source_dir and output_dir must not overlap")


def _find_source_images(source_dir: Path) -> list[Path]:
    images = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in _SUPPORTED_SUFFIXES
    ]
    return sorted(
        images,
        key=lambda path: (
            path.relative_to(source_dir).as_posix().casefold(),
            path.relative_to(source_dir).as_posix(),
        ),
    )


def _ensure_bbbc038(source_dir: Path) -> list[Path]:
    if source_dir.exists() and not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir is not a directory: {source_dir}")
    source_dir.mkdir(parents=True, exist_ok=True)

    cached_images = _bbbc038_images(source_dir)
    if len(cached_images) == BBBC038_IMAGE_COUNT:
        return cached_images

    archive_path = source_dir / _BBBC038_ARCHIVE_NAME
    if archive_path.exists():
        _verify_bbbc038_checksum(archive_path)
    else:
        _download_bbbc038(archive_path)

    _extract_bbbc038(archive_path, source_dir)
    extracted_images = _bbbc038_images(source_dir)
    if len(extracted_images) != BBBC038_IMAGE_COUNT:
        raise ValueError(
            "BBBC038 extraction did not produce "
            f"{BBBC038_IMAGE_COUNT} source PNG images"
        )
    return extracted_images


def _bbbc038_images(source_dir: Path) -> list[Path]:
    images = [
        path
        for path in source_dir.glob("*/images/*.png")
        if path.is_file()
        and len(path.relative_to(source_dir).parts) == 3
        and path.parent.name == "images"
    ]
    return sorted(
        images,
        key=lambda path: (
            path.relative_to(source_dir).as_posix().casefold(),
            path.relative_to(source_dir).as_posix(),
        ),
    )


def _download_bbbc038(archive_path: Path) -> None:
    temporary_path = archive_path.with_name(archive_path.name + ".part")
    response = None
    try:
        response = requests.get(BBBC038_URL, stream=True, timeout=(10, 120))
        response.raise_for_status()
        with temporary_path.open("wb") as destination:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    destination.write(chunk)
        _verify_bbbc038_checksum(temporary_path)
        os.replace(temporary_path, archive_path)
    finally:
        if response is not None:
            close = getattr(response, "close", None)
            if close is not None:
                close()
        if temporary_path.exists():
            temporary_path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bbbc038_checksum(path: Path) -> None:
    actual = _sha256_file(path)
    if actual != BBBC038_SHA256:
        raise ValueError(
            f"BBBC038 checksum mismatch for {path}: "
            f"expected {BBBC038_SHA256}, got {actual}"
        )


def _extract_bbbc038(archive_path: Path, source_dir: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            selected: list[tuple[zipfile.ZipInfo, tuple[str, str, str]]] = []
            destinations: set[str] = set()

            for member in archive.infolist():
                parts = _safe_zip_parts(member)
                if not parts or member.is_dir() or parts[0] == "__MACOSX":
                    continue
                if (
                    len(parts) == 3
                    and parts[1] == "images"
                    and PurePosixPath(parts[2]).suffix.casefold() == ".png"
                ):
                    destination_parts = (parts[0], parts[1], parts[2])
                    destination_key = "/".join(destination_parts).casefold()
                    if destination_key in destinations:
                        raise ValueError(
                            "BBBC038 archive has a duplicate source image path: "
                            f"{'/'.join(destination_parts)}"
                        )
                    destinations.add(destination_key)
                    selected.append((member, destination_parts))

            if len(selected) != BBBC038_IMAGE_COUNT:
                raise ValueError(
                    "BBBC038 archive must contain exactly "
                    f"{BBBC038_IMAGE_COUNT} <ImageId>/images/*.png files; "
                    f"found {len(selected)}"
                )

            for member, destination_parts in selected:
                target = source_dir.joinpath(*destination_parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary_name: str | None = None
                try:
                    with archive.open(member, "r") as source, tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=target.parent,
                        prefix=f".{target.name}.",
                        suffix=".part",
                        delete=False,
                    ) as temporary:
                        temporary_name = temporary.name
                        shutil.copyfileobj(source, temporary)
                    os.replace(temporary_name, target)
                    temporary_name = None
                finally:
                    if temporary_name is not None:
                        Path(temporary_name).unlink(missing_ok=True)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"BBBC038 archive is not a valid ZIP file: {archive_path}") from exc


def _safe_zip_parts(member: zipfile.ZipInfo) -> tuple[str, ...]:
    name = member.filename
    if not name or "\x00" in name:
        raise ValueError("BBBC038 archive contains an invalid empty or NUL path")
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or normalized.startswith("//")
        or re.match(r"^[A-Za-z]:", normalized)
        or ".." in path.parts
    ):
        raise ValueError(f"BBBC038 archive contains an unsafe path: {name!r}")

    unix_mode = member.external_attr >> 16
    if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
        raise ValueError(f"BBBC038 archive contains a symbolic link: {name!r}")
    return path.parts


def _load_image(path: Path) -> tuple[np.ndarray, int]:
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"multi-frame images are not supported: {path}")

            if image.mode == "L":
                array = np.asarray(image, dtype=np.uint8).copy()
                bit_depth = 8
            elif image.mode == "RGB":
                array = np.asarray(image, dtype=np.uint8).copy()
                bit_depth = 8
            elif image.mode == "RGBA":
                rgba = np.asarray(image, dtype=np.uint8)
                if not np.all(rgba[..., 3] == 255):
                    raise ValueError(
                        f"images with non-opaque alpha are not supported: {path}"
                    )
                array = rgba[..., :3].copy()
                bit_depth = 8
            elif image.mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
                array = np.asarray(image).astype(np.uint16, copy=True)
                bit_depth = 16
            else:
                raise ValueError(
                    f"unsupported image mode {image.mode!r} for {path}; "
                    "expected 8-bit L/RGB/opaque RGBA or 16-bit grayscale"
                )
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"could not decode image: {path}") from exc

    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] != 3):
        raise ValueError(f"unsupported image shape {array.shape!r} for {path}")
    return array, bit_depth


def _normalize_image(array: np.ndarray, bit_depth: int) -> np.ndarray:
    maximum = np.float32(65535.0 if bit_depth == 16 else 255.0)
    return array.astype(np.float32) / maximum


def _denormalize_image(array: np.ndarray, bit_depth: int) -> np.ndarray:
    maximum = np.float32(65535.0 if bit_depth == 16 else 255.0)
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    return np.rint(np.clip(array, 0.0, 1.0) * maximum).astype(dtype)


def _save_png(array: np.ndarray, bit_depth: int, path: Path) -> None:
    if bit_depth == 16:
        image_array = np.ascontiguousarray(array, dtype=np.uint16)
    else:
        image_array = np.ascontiguousarray(array, dtype=np.uint8)
    Image.fromarray(image_array).save(path, format="PNG")


def _apply_noise(
    image: np.ndarray,
    noise_type: str,
    params: Mapping[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_type == "gaussian":
        perturbation = rng.normal(
            params["mean"], params["std"], size=image.shape
        ).astype(np.float32)
        return image + perturbation

    if noise_type == "poisson":
        peak = np.float32(params["peak"])
        return (rng.poisson(image * peak) / peak).astype(np.float32)

    if noise_type == "salt_pepper":
        result = image.copy()
        amount = params["amount"]
        if amount == 0.0:
            return result
        choices = rng.random(image.shape[:2])
        pepper_limit = amount * (1.0 - params["salt_ratio"])
        pepper = choices < pepper_limit
        salt = (choices >= pepper_limit) & (choices < amount)
        result[pepper] = 0.0
        result[salt] = 1.0
        return result

    raise AssertionError(f"unhandled noise type: {noise_type}")
