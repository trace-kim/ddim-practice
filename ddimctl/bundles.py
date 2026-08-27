"""Configuration loading and immutable, portable run-bundle creation."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel

from .schemas import (
    AttemptState,
    DatasetFingerprint,
    MachineProfile,
    ReproducibilityMode,
    RunManifest,
    RunStatus,
    SourceSnapshot,
    TrainingSpec,
)


class ConfigurationError(ValueError):
    """Raised when a configuration is ambiguous, unsupported, or invalid."""


class DuplicateConfigurationKey(ConfigurationError):
    pass


class DuplicateOptionError(ConfigurationError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConfigurationError("configuration mapping keys must be scalar") from exc
        if duplicate:
            mark = key_node.start_mark
            raise DuplicateConfigurationKey(
                f"duplicate YAML key {key!r} at line {mark.line + 1}, column {mark.column + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_CONFIG_KEY_MAP: dict[str, str | None] = {
    "schema_version": "schema_version",
    "data.dataset": None,
    "data.dataset_alias": "dataset_alias",
    "data.image_size": "image_size",
    "data.channels": "channels",
    "data.logit_transform": "logit_transform",
    "data.uniform_dequantization": "uniform_dequantization",
    "data.gaussian_dequantization": "gaussian_dequantization",
    "data.random_flip": "random_flip",
    "data.rescaled": "rescaled",
    "data.num_workers": "num_workers",
    "data.cache_in_memory": "cache_in_memory",
    "data.recursive": "recursive",
    "data.validation_split": "validation_split",
    "data.split_seed": "split_seed",
    "data.extensions": "extensions",
    "model.type": "model_type",
    "model.in_channels": "in_channels",
    "model.out_ch": "out_channels",
    "model.ch": "model_ch",
    "model.ch_mult": "ch_mult",
    "model.num_res_blocks": "num_res_blocks",
    "model.attn_resolutions": "attn_resolutions",
    "model.dropout": "dropout",
    "model.var_type": "var_type",
    "model.ema_rate": "ema_rate",
    "model.ema": "ema",
    "model.resamp_with_conv": "resamp_with_conv",
    "diffusion.beta_schedule": "beta_schedule",
    "diffusion.beta_start": "beta_start",
    "diffusion.beta_end": "beta_end",
    "diffusion.num_diffusion_timesteps": "diffusion_steps",
    "training.batch_size": "batch_size",
    "training.max_steps": "max_steps",
    "training.snapshot_freq": "checkpoint_every",
    "training.checkpoint_every": "checkpoint_every",
    "training.validation_freq": "validation_every",
    "training.validation_every": "validation_every",
    "training.sample_freq": "sample_every",
    "training.sample_every": "sample_every",
    "training.checkpoint_minutes": "checkpoint_minutes",
    "sampling.batch_size": "sampling_batch_size",
    "sampling.last_only": "sampling_last_only",
    "optim.weight_decay": "weight_decay",
    "optim.optimizer": "optimizer",
    "optim.lr": "lr",
    "optim.beta1": "beta1",
    "optim.amsgrad": "amsgrad",
    "optim.eps": "eps",
    "optim.grad_clip": "grad_clip",
    "experiment.label": "label",
    "experiment.dataset_alias": "dataset_alias",
    "experiment.seed": "seed",
    "experiment.reproducibility": "reproducibility",
}

_TUPLE_FIELDS = {"extensions", "ch_mult", "attn_resolutions"}


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigurationError("configuration keys must be strings")
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, dotted))
        else:
            flattened[dotted] = item
    return flattened


def _parse_base_config(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"base configuration not found: {path}") from exc
    loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    if loaded is None:
        raise ConfigurationError(f"base configuration is empty: {path}")
    if not isinstance(loaded, Mapping):
        raise ConfigurationError("base configuration must be a YAML mapping")

    flat = _flatten_mapping(loaded)
    unknown = sorted(set(flat) - set(_CONFIG_KEY_MAP))
    if unknown:
        raise ConfigurationError(
            "unknown base configuration key(s): " + ", ".join(unknown)
        )
    if flat.get("data.dataset", "SEM") != "SEM":
        raise ConfigurationError("the active training configuration supports only data.dataset: SEM")

    parsed: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for source_key, value in flat.items():
        target_key = _CONFIG_KEY_MAP[source_key]
        if target_key is None:
            continue
        if target_key in parsed:
            raise ConfigurationError(
                f"{source_key!r} and {origins[target_key]!r} both configure {target_key!r}"
            )
        if target_key in _TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        if target_key == "reproducibility" and isinstance(value, str):
            try:
                value = ReproducibilityMode(value)
            except ValueError as exc:
                raise ConfigurationError(f"invalid reproducibility mode: {value!r}") from exc
        parsed[target_key] = value
        origins[target_key] = source_key
    return parsed


def load_training_spec(
    config_path: Path | str,
    overrides: Mapping[str, Any] | None = None,
) -> TrainingSpec:
    """Resolve one strict SEM base config plus explicit field-name overrides."""

    resolved = _parse_base_config(config_path)
    if overrides:
        unknown = sorted(set(overrides) - set(TrainingSpec.model_fields))
        if unknown:
            raise ConfigurationError("unknown training override(s): " + ", ".join(unknown))
        for key, value in overrides.items():
            if value is None:
                continue
            if key in _TUPLE_FIELDS and isinstance(value, list):
                value = tuple(value)
            if key == "reproducibility" and isinstance(value, str):
                try:
                    value = ReproducibilityMode(value)
                except ValueError as exc:
                    raise ConfigurationError(f"invalid reproducibility mode: {value!r}") from exc
            resolved[key] = value
    return TrainingSpec.model_validate(resolved)


def duplicate_scalar_options(
    argv: Sequence[str],
    scalar_options: Collection[str],
) -> tuple[str, ...]:
    """Find repeated long scalar flags, including the --flag=value spelling."""

    scalar = set(scalar_options)
    counts = {option: 0 for option in scalar}
    options_enabled = True
    for argument in argv:
        if argument == "--":
            options_enabled = False
            continue
        if not options_enabled or not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        if option in counts:
            counts[option] += 1
    return tuple(sorted(option for option, count in counts.items() if count > 1))


def reject_duplicate_scalar_options(
    argv: Sequence[str],
    scalar_options: Collection[str],
) -> None:
    duplicates = duplicate_scalar_options(argv, scalar_options)
    if duplicates:
        raise DuplicateOptionError("duplicate scalar option(s): " + ", ".join(duplicates))


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Write UTF-8 text via fsync and same-directory atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        if os.name != "nt":
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path | str, value: Any) -> Path:
    payload = json.dumps(
        _jsonable(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    return atomic_write_text(path, payload + "\n")


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    values = tuple(argv)
    if not values:
        raise ValueError("argv cannot be empty")
    for value in values:
        if not isinstance(value, str):
            raise TypeError("argv must contain only strings")
        if any(character in value for character in "\r\n\0"):
            raise ValueError("argv entries cannot contain newline or NUL characters")
    return values


def render_posix_command(argv: Sequence[str]) -> str:
    return shlex.join(_validate_argv(argv))


def _quote_powershell(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_powershell_command(argv: Sequence[str]) -> str:
    values = _validate_argv(argv)
    return "& " + " ".join(_quote_powershell(value) for value in values)


def fingerprint_dataset(
    root: Path | str,
    extensions: Collection[str],
    *,
    recursive: bool = False,
    hash_contents: bool = True,
) -> DatasetFingerprint:
    """Fingerprint sorted SEM inputs without following directories or symlinks."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_root}")
    normalized = {extension.lower() for extension in extensions}
    iterator = dataset_root.rglob("*") if recursive else dataset_root.glob("*")
    files = sorted(
        (
            path
            for path in iterator
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in normalized
        ),
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )
    if not files:
        raise ConfigurationError(
            f"dataset contains no matching image files at {dataset_root} "
            f"(extensions: {', '.join(sorted(normalized))})"
        )

    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(dataset_root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        if hash_contents:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
    return DatasetFingerprint(
        root=dataset_root,
        file_count=len(files),
        total_bytes=total_bytes,
        sha256=digest.hexdigest(),
        method="sha256-content-v1" if hash_contents else "sha256-metadata-v1",
    )


_SNAPSHOT_EXCLUDED_ROOTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "data",
    "runs",
    "experiments",
    "output",
    "tmp",
    "logs",
    "tensorboard",
    "wandb",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def _git_output(repo_root: Path, *arguments: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _is_git_repository_root(repo_root: Path) -> bool:
    value = _git_output(repo_root, "rev-parse", "--show-toplevel")
    if not value:
        return False
    try:
        return Path(os.fsdecode(value).strip()).resolve() == repo_root.resolve()
    except (OSError, ValueError):
        return False


def _source_files(repo_root: Path) -> list[Path]:
    git_files = (
        _git_output(
            repo_root,
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        )
        if _is_git_repository_root(repo_root)
        else None
    )
    candidates: list[Path]
    if git_files is not None:
        candidates = [
            Path(os.fsdecode(item))
            for item in git_files.split(b"\0")
            if item
        ]
    else:
        # A non-editable wheel lives inside site-packages. Never snapshot that
        # entire environment: capture only this distribution's executable code.
        package_roots = ("ddimctl", "datasets", "functions", "models", "runners")
        if (repo_root / "ddimctl").is_dir():
            candidates = []
            for package_root in package_roots:
                root = repo_root / package_root
                if root.is_dir():
                    candidates.extend(path.relative_to(repo_root) for path in root.rglob("*"))
            for name in ("main.py", "pyproject.toml", "README.md", "LICENSE"):
                if (repo_root / name).is_file():
                    candidates.append(Path(name))
        else:
            candidates = [path.relative_to(repo_root) for path in repo_root.rglob("*")]

    selected: list[Path] = []
    for relative in candidates:
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigurationError(f"unsafe source path: {relative}")
        if not relative.parts or relative.parts[0] in _SNAPSHOT_EXCLUDED_ROOTS:
            continue
        absolute = repo_root / relative
        if absolute.is_file() or absolute.is_symlink():
            selected.append(relative)
    return sorted(set(selected), key=lambda path: PurePosixPath(*path.parts).as_posix())


def create_source_snapshot(
    repo_root: Path | str,
    archive_path: Path | str,
) -> SourceSnapshot:
    """Create a deterministic gzip-compressed tar of tracked/nonignored source."""

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root not found: {root}")
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = _source_files(root)
    total_bytes = sum((root / relative).stat().st_size for relative in files if not (root / relative).is_symlink())

    tar_temp = archive.parent / f".{archive.name}.{uuid.uuid4().hex}.tar.tmp"
    gzip_temp = archive.parent / f".{archive.name}.{uuid.uuid4().hex}.gz.tmp"
    try:
        with tarfile.open(tar_temp, mode="w", dereference=False) as tar:
            for relative in files:
                absolute = root / relative
                archive_name = PurePosixPath(*relative.parts).as_posix()
                info = tar.gettarinfo(str(absolute), arcname=archive_name)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if info.isfile():
                    with absolute.open("rb") as handle:
                        tar.addfile(info, handle)
                else:
                    tar.addfile(info)
        with tar_temp.open("rb") as source, gzip_temp.open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(gzip_temp, archive)

        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        is_git_root = _is_git_repository_root(root)
        commit_raw = _git_output(root, "rev-parse", "HEAD") if is_git_root else None
        commit = commit_raw.decode("ascii").strip() if commit_raw else None
        if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
            commit = None
        status = (
            _git_output(root, "status", "--porcelain", "--untracked-files=normal")
            if is_git_root
            else None
        )
        return SourceSnapshot(
            archive=archive.name,
            sha256=digest.hexdigest(),
            file_count=len(files),
            total_bytes=total_bytes,
            git_commit=commit,
            git_dirty=bool(status),
        )
    finally:
        tar_temp.unlink(missing_ok=True)
        gzip_temp.unlink(missing_ok=True)


def materialized_source_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "source"


def worker_bootstrap_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "worker-bootstrap.py"


def _worker_bootstrap_source() -> str:
    return '''"""Run the worker from this bundle's verified source snapshot."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import json
import os
import runpy
import sys


def option_value(name):
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def selected_attempt(run_dir):
    configured = option_value("--attempt")
    if configured is not None:
        number = int(configured)
        if number < 1:
            raise ValueError("--attempt must be positive")
        return number
    attempts = run_dir / "attempts"
    existing = [int(path.name) for path in attempts.iterdir()
                if path.is_dir() and path.name.isdigit()] if attempts.is_dir() else []
    return max(existing, default=0) + 1


def isolate_cuda_gpu(manifest_path):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    machine = payload.get("machine")
    if not isinstance(machine, dict):
        raise ValueError("run manifest has no machine profile")
    gpu_index = int(machine.get("gpu_index", 0))
    if gpu_index < 0:
        raise ValueError("manifest gpu_index must be nonnegative")

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        selected = str(gpu_index)
    else:
        visible = [item.strip() for item in inherited.split(",") if item.strip()]
        if not visible:
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not expose any GPU")
        if gpu_index >= len(visible):
            raise RuntimeError(
                "manifest gpu_index %d is out of range for CUDA_VISIBLE_DEVICES=%r"
                % (gpu_index, inherited)
            )
        selected = visible[gpu_index]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    print(
        "worker GPU isolation: visible index %d -> CUDA_VISIBLE_DEVICES=%r"
        % (gpu_index, selected),
        flush=True,
    )


sys.dont_write_bytecode = True
bundle_root = Path(__file__).resolve().parent
sys.path.insert(0, str(bundle_root / "source"))
run_value = option_value("--run")
if run_value is None:
    manifest_value = option_value("--manifest")
    if manifest_value is not None:
        isolate_cuda_gpu(Path(manifest_value).expanduser().resolve())
    runpy.run_module("ddimctl.worker", run_name="__main__")
else:
    run_root = Path(run_value).expanduser().resolve()
    attempt_dir = run_root / "attempts" / ("%03d" % selected_attempt(run_root))
    attempt_dir.mkdir(parents=True, exist_ok=True)
    with (attempt_dir / "stdout.log").open("a", encoding="utf-8", buffering=1) as stdout_log, \\
         (attempt_dir / "stderr.log").open("a", encoding="utf-8", buffering=1) as stderr_log, \\
         redirect_stdout(stdout_log), redirect_stderr(stderr_log):
        isolate_cuda_gpu(run_root / "manifest.json")
        runpy.run_module("ddimctl.worker", run_name="__main__")
'''


def _safe_archive_name(name: str) -> str:
    if not name or "\\" in name or "\0" in name:
        raise ConfigurationError(f"unsafe source archive member: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ConfigurationError(f"unsafe source archive member: {name!r}")
    if path.parts and re.fullmatch(r"[A-Za-z]:", path.parts[0]):
        raise ConfigurationError(f"unsafe source archive member: {name!r}")
    return path.as_posix()


def _safe_link_target(member_name: str, link_name: str, *, hard_link: bool) -> str:
    if not link_name or "\\" in link_name or "\0" in link_name:
        raise ConfigurationError(
            f"unsafe link target {link_name!r} in source archive member {member_name!r}"
        )
    target = PurePosixPath(link_name)
    if target.is_absolute() or (target.parts and re.fullmatch(r"[A-Za-z]:", target.parts[0])):
        raise ConfigurationError(
            f"absolute link target {link_name!r} in source archive member {member_name!r}"
        )
    base = "" if hard_link else posixpath.dirname(member_name)
    normalized = posixpath.normpath(posixpath.join(base, link_name))
    if normalized == ".." or normalized.startswith("../"):
        raise ConfigurationError(
            f"link target escapes source root: {member_name!r} -> {link_name!r}"
        )
    return _safe_archive_name(normalized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_source_snapshot(run_dir: Path | str) -> Path:
    """Verify and safely extract a run's captured source for scheduler workers.

    Extraction is performed into a private staging directory.  Regular files
    and directories are created before links, preventing a link member from
    redirecting a later file write outside the materialized source root.
    """

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    archive = root / manifest.source.archive
    if not archive.is_file():
        raise FileNotFoundError(f"source archive not found: {archive}")
    actual_sha256 = _sha256_file(archive)
    if actual_sha256 != manifest.source.sha256:
        raise ConfigurationError(
            "source archive checksum mismatch: expected {}, found {}".format(
                manifest.source.sha256,
                actual_sha256,
            )
        )

    destination = materialized_source_path(root)
    if destination.exists():
        raise FileExistsError(f"materialized source already exists: {destination}")
    temporary = root / f".source.materializing-{uuid.uuid4().hex}"
    temporary.mkdir(exist_ok=False)
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) != manifest.source.file_count:
                raise ConfigurationError(
                    "source archive member count does not match its manifest"
                )
            by_name: dict[str, tarfile.TarInfo] = {}
            platform_names: set[str] = set()
            safe_links: dict[str, str] = {}
            for member in members:
                name = _safe_archive_name(member.name)
                platform_name = os.path.normcase(name)
                if name in by_name or platform_name in platform_names:
                    raise ConfigurationError(f"duplicate source archive member: {name!r}")
                if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ConfigurationError(
                        f"unsupported source archive member type: {name!r}"
                    )
                by_name[name] = member
                platform_names.add(platform_name)
                if member.issym() or member.islnk():
                    safe_links[name] = _safe_link_target(
                        name,
                        member.linkname,
                        hard_link=member.islnk(),
                    )

            archived_bytes = sum(member.size for member in members if member.isfile())
            if archived_bytes != manifest.source.total_bytes:
                raise ConfigurationError(
                    "source archive byte count does not match its manifest"
                )

            directories = sorted(
                ((name, member) for name, member in by_name.items() if member.isdir()),
                key=lambda item: (len(PurePosixPath(item[0]).parts), item[0]),
            )
            for name, member in directories:
                target = temporary.joinpath(*PurePosixPath(name).parts)
                target.mkdir(parents=True, exist_ok=True)

            for name, member in by_name.items():
                target = temporary.joinpath(*PurePosixPath(name).parts)
                if member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise ConfigurationError(f"cannot read source archive member: {name!r}")
                    with extracted, target.open("xb") as output:
                        shutil.copyfileobj(extracted, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    target.chmod((member.mode & 0o777) & ~0o222)

            for name, member in by_name.items():
                if not (member.issym() or member.islnk()):
                    continue
                target = temporary.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                safe_target = safe_links[name]
                resolved_target = temporary.joinpath(*PurePosixPath(safe_target).parts)
                if member.islnk():
                    if not resolved_target.is_file() or resolved_target.is_symlink():
                        raise ConfigurationError(
                            f"hard-link target is not a regular extracted file: {name!r}"
                        )
                    os.link(resolved_target, target)
                else:
                    os.symlink(member.linkname, target)

        temporary.rename(destination)
        atomic_write_text(worker_bootstrap_path(root), _worker_bootstrap_source())
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        worker_bootstrap_path(root).unlink(missing_ok=True)
        raise


def _environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    snapshot: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].casefold())),
        "environment": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")
            if key in os.environ
        },
    }
    try:
        import torch

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory": properties.total_memory,
                        "compute_capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
        snapshot["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "devices": devices,
        }
    except Exception as error:
        snapshot["torch"] = {"error": f"{type(error).__name__}: {error}"}
    return snapshot


def _run_id(created_at: datetime, spec: TrainingSpec, profile: MachineProfile, argv: tuple[str, ...]) -> str:
    identity = json.dumps(
        {
            "created_at": created_at.isoformat(),
            "config_sha256": spec.config_sha256,
            "machine_id": profile.machine_id,
            "argv": argv,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:12]
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%z")
    return f"{timestamp}__{spec.label}__{suffix}"


def create_run_bundle(
    repo_root: Path | str,
    profile: MachineProfile,
    spec: TrainingSpec,
    canonical_argv: Sequence[str],
    parent_run_id: str | None = None,
    *,
    now: datetime | None = None,
    hash_dataset_contents: bool = True,
) -> tuple[Path, RunManifest]:
    """Prepare a complete run directory and atomically publish it.

    The target directory is never overwritten.  A hidden staging directory is
    removed on preparation failure and renamed only after all immutable metadata
    and initial state have been flushed.
    """

    argv = _validate_argv(canonical_argv)
    timezone = ZoneInfo(profile.timezone)
    created_at = datetime.now(timezone) if now is None else now.astimezone(timezone)
    run_id = _run_id(created_at, spec, profile, argv)
    day_root = profile.runs_root.expanduser().resolve() / created_at.strftime("%Y-%m-%d")
    final_root = day_root / run_id
    day_root.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise FileExistsError(f"run directory already exists: {final_root}")
    staging = day_root / f".{run_id}.preparing-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)

    try:
        for directory in ("tensorboard", "samples", "checkpoints", "attempts/001"):
            (staging / directory).mkdir(parents=True, exist_ok=False)

        dataset = fingerprint_dataset(
            profile.dataset_path(spec.dataset_alias),
            spec.extensions,
            recursive=spec.recursive,
            hash_contents=hash_dataset_contents,
        )
        source = create_source_snapshot(repo_root, staging / "source.tar.gz")
        manifest = RunManifest(
            run_id=run_id,
            created_at=created_at,
            canonical_argv=argv,
            training=spec,
            machine=profile,
            dataset=dataset,
            source=source,
            config_sha256=spec.config_sha256,
            parent_run_id=parent_run_id,
        )
        state = AttemptState(
            attempt=1,
            status=RunStatus.PREPARED,
            updated_at=created_at,
        )

        atomic_write_json(staging / "manifest.json", manifest)
        atomic_write_json(staging / "argv.json", list(argv))
        atomic_write_json(staging / "dataset.json", dataset)
        atomic_write_json(staging / "environment.json", _environment_snapshot())
        atomic_write_json(staging / "state.json", state)
        atomic_write_json(
            staging / "attempts" / "001" / "backend.json",
            {"executor": profile.executor.value, "status": RunStatus.PREPARED.value},
        )
        atomic_write_text(
            staging / "resolved_config.yml",
            yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=True),
        )
        atomic_write_text(
            staging / "command.sh",
            "#!/usr/bin/env sh\nset -eu\nexec " + render_posix_command(argv) + "\n",
        )
        atomic_write_text(
            staging / "command.ps1",
            "$ErrorActionPreference = 'Stop'\n" + render_powershell_command(argv) + "\n",
        )
        atomic_write_text(staging / "attempts" / "001" / "stdout.log", "")
        atomic_write_text(staging / "attempts" / "001" / "stderr.log", "")
        atomic_write_text(staging / "metrics.jsonl", "")
        materialize_source_snapshot(staging)

        try:
            staging.rename(final_root)
        except FileExistsError:
            raise FileExistsError(f"run directory already exists: {final_root}") from None
        return final_root, manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_manifest(run_root: Path | str) -> RunManifest:
    path = Path(run_root) / "manifest.json"
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_attempt_state(run_root: Path | str) -> AttemptState:
    path = Path(run_root) / "state.json"
    return AttemptState.model_validate_json(path.read_text(encoding="utf-8"))


def save_attempt_state(run_root: Path | str, state: AttemptState) -> Path:
    return atomic_write_json(Path(run_root) / "state.json", state)
