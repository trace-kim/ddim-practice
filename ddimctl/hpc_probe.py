"""Offline runtime probe for Linux/HPC training targets.

This module performs local process and filesystem checks only.  It makes no
network requests and can be copied into a restricted environment with the run
bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .backends import CommandResult, run_command


PROBE_SCHEMA_VERSION = 1


def _version_command(
    executable: str | None,
    arguments: Sequence[str],
    *,
    runner: Callable[..., CommandResult],
) -> dict[str, Any]:
    if not executable:
        return {"available": False, "path": None}
    try:
        result = runner([executable, *arguments], timeout=10)
    except Exception as exc:
        return {"available": False, "path": executable, "error": str(exc)}
    output = (result.stdout or result.stderr).strip()
    return {
        "available": result.returncode == 0,
        "path": executable,
        "returncode": result.returncode,
        "version": output.splitlines()[0] if output else None,
        **({"error": result.stderr.strip()} if result.returncode != 0 else {}),
    }


def _scheduler_report(
    *,
    which: Callable[[str], str | None],
    runner: Callable[..., CommandResult],
) -> dict[str, Any]:
    commands = {
        name: which(name)
        for name in (
            "sbatch",
            "squeue",
            "sacct",
            "scancel",
            "sinfo",
            "qsub",
            "qstat",
            "bsub",
            "bjobs",
        )
    }
    if commands["sbatch"] and commands["squeue"]:
        scheduler = "slurm"
        version = _version_command(commands["sbatch"], ["--version"], runner=runner)
    elif commands["bsub"] and commands["bjobs"]:
        scheduler = "lsf"
        version = _version_command(commands["bsub"], ["-V"], runner=runner)
    elif commands["qsub"] and commands["qstat"]:
        scheduler = "pbs-or-sge"
        version = _version_command(commands["qstat"], ["--version"], runner=runner)
    else:
        scheduler = "unknown"
        version = {"available": False, "path": None}
    return {
        "detected": scheduler,
        "native_slurm_ready": all(
            commands[name] for name in ("sbatch", "squeue", "sacct", "scancel")
        ),
        "commands": commands,
        "version": version,
        "job_environment": {
            key: os.environ[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_GPUS",
                "PBS_JOBID",
                "LSB_JOBID",
            )
            if key in os.environ
        },
    }


def _torch_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    report: dict[str, Any] = {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": 0,
        "devices": [],
    }
    try:
        count = int(torch.cuda.device_count())
        report["device_count"] = count
        devices: list[dict[str, Any]] = []
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
        report["devices"] = devices
        if report["cuda_available"] and count:
            device = torch.device("cuda", 0)
            model = torch.nn.Linear(16, 8).to(device)
            inputs = torch.randn(4, 16, device=device)
            loss = model(inputs).square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            with tempfile.TemporaryDirectory(prefix="ddim-cuda-probe-") as directory:
                checkpoint = Path(directory) / "roundtrip.pt"
                torch.save({"model": model.state_dict(), "loss": float(loss.item())}, checkpoint)
                try:
                    restored = torch.load(checkpoint, map_location=device, weights_only=True)
                except TypeError:
                    restored = torch.load(checkpoint, map_location=device)
                if set(restored["model"]) != set(model.state_dict()):
                    raise RuntimeError("checkpoint roundtrip changed model keys")
            report["acceptance"] = {
                "success": True,
                "device": str(device),
                "forward_backward_loss": float(loss.item()),
                "checkpoint_roundtrip": True,
            }
    except Exception as exc:
        report["device_error"] = f"{type(exc).__name__}: {exc}"
        report["acceptance"] = {"success": False, "error": report["device_error"]}
    return report


def _nvidia_report(
    *,
    which: Callable[[str], str | None],
    runner: Callable[..., CommandResult],
) -> dict[str, Any]:
    executable = which("nvidia-smi")
    if not executable:
        return {"available": False, "path": None, "gpus": []}
    result = runner(
        [
            executable,
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "path": executable,
            "gpus": [],
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            memory_mib: int | None = int(parts[2])
        except ValueError:
            memory_mib = None
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_mib": memory_mib,
                "driver_version": parts[3],
                "compute_capability": parts[4],
            }
        )
    return {"available": True, "path": executable, "gpus": gpus}


def _path_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    requested = Path(path).expanduser()
    try:
        resolved = requested.resolve(strict=False)
    except OSError:
        resolved = requested.absolute()
    ancestor = resolved
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    result: dict[str, Any] = {
        "requested": os.fspath(path),
        "resolved": str(resolved),
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
        "is_directory": resolved.is_dir(),
        "readable": os.access(resolved if resolved.exists() else ancestor, os.R_OK),
        "writable": os.access(resolved if resolved.exists() else ancestor, os.W_OK),
        "existing_ancestor": str(ancestor),
    }
    try:
        usage = shutil.disk_usage(ancestor)
        result["disk"] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except OSError as exc:
        result["disk_error"] = str(exc)
    return result


def _normalize_paths(
    paths: Mapping[str, str | os.PathLike[str]] | Sequence[str | os.PathLike[str]] | None,
) -> dict[str, str | os.PathLike[str]]:
    if paths is None:
        return {"working_directory": Path.cwd()}
    if isinstance(paths, Mapping):
        return {str(key): value for key, value in paths.items()}
    return {f"path_{index + 1}": value for index, value in enumerate(paths)}


def collect_hpc_probe(
    *,
    paths: Mapping[str, str | os.PathLike[str]] | Sequence[str | os.PathLike[str]] | None = None,
    include_torch: bool = True,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., CommandResult] = run_command,
) -> dict[str, Any]:
    """Collect a JSON-serializable, network-free machine capability report."""

    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}
    apptainer_path = which("apptainer")
    singularity_path = which("singularity")
    container_runtime = (
        _version_command(apptainer_path, ["--version"], runner=runner)
        if apptainer_path
        else _version_command(singularity_path, ["--version"], runner=runner)
    )
    normalized_paths = _normalize_paths(paths)
    report: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "offline": True,
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": sys.platform,
            "os_name": os.name,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "os_release": os_release,
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "version_detail": sys.version,
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "virtual_environment": sys.prefix != sys.base_prefix,
            "architecture": platform.architecture()[0],
        },
        "scheduler": _scheduler_report(which=which, runner=runner),
        "container_runtime": container_runtime,
        "nvidia": _nvidia_report(which=which, runner=runner),
        "paths": {name: _path_report(value) for name, value in normalized_paths.items()},
    }
    report["torch"] = _torch_report() if include_torch else {"skipped": True}
    return report


def write_hpc_probe(
    report: Mapping[str, Any] | str | os.PathLike[str],
    output_path: str | os.PathLike[str] | Mapping[str, Any],
) -> Path:
    """Atomically write a probe and include a sidecar SHA-256 checksum.

    ``(report, output_path)`` is canonical.  ``(output_path, report)`` is also
    accepted for compatibility with early launcher builds.
    """

    if not isinstance(report, Mapping) and isinstance(output_path, Mapping):
        report, output_path = output_path, report
    if not isinstance(report, Mapping) or isinstance(output_path, Mapping):
        raise TypeError("write_hpc_probe requires a report mapping and an output path")

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    checksum_path = target.with_name(target.name + ".sha256")
    checksum_path.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return target


def _parse_path_arguments(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"path must use NAME=PATH syntax: {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"path must use nonempty NAME=PATH syntax: {value!r}")
        parsed[name] = path
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect an offline DDIM HPC capability probe")
    parser.add_argument("--output", type=Path, help="Write JSON plus a .sha256 sidecar")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Check a required workspace, runs, dataset, or environment path",
    )
    parser.add_argument("--skip-torch", action="store_true", help="Do not import PyTorch")
    arguments = parser.parse_args(argv)
    try:
        paths = _parse_path_arguments(arguments.path)
    except ValueError as exc:
        parser.error(str(exc))
    report = collect_hpc_probe(paths=paths or None, include_torch=not arguments.skip_torch)
    if arguments.output:
        target = write_hpc_probe(report, arguments.output)
        print(target)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROBE_SCHEMA_VERSION", "collect_hpc_probe", "main", "write_hpc_probe"]
