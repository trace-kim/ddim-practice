"""Execution backends for durable, reproducible training jobs.

The adapters in this module deliberately depend only on the standard library.
They accept mappings as well as profile objects so that the execution layer is
not coupled to the configuration model used by the CLI.
"""

from __future__ import annotations

import base64
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


TERMINAL_STATES = frozenset(
    {"completed", "failed", "cancelled", "timed_out", "interrupted", "lost"}
)


class BackendError(RuntimeError):
    """Base class for execution backend errors."""


class BackendUnavailableError(BackendError):
    """Raised when a requested executor is not installed or usable."""


class SubmissionError(BackendError):
    """Raised when a scheduler rejects a job."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult: ...


def run_command(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = 30,
) -> CommandResult:
    """Run an argument vector without involving a command interpreter."""

    normalized = tuple(str(part) for part in argv)
    if not normalized:
        raise ValueError("argv must not be empty")
    completed = subprocess.run(
        normalized,
        cwd=None if cwd is None else os.fspath(cwd),
        env=None if env is None else dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        argv=normalized,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _value(source: object | None, *names: str, default: Any = None) -> Any:
    if source is None:
        return default
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_no_newline(value: object, label: str) -> str:
    result = str(value)
    if "\n" in result or "\r" in result:
        raise ValueError(f"{label} must not contain a newline")
    return result


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    located = shutil.which(value)
    if located is None:
        raise BackendUnavailableError(f"executable not found: {value}")
    return str(Path(located).resolve())


@dataclass(frozen=True)
class LaunchRequest:
    """Scheduler-neutral request for one worker process."""

    argv: Sequence[str]
    cwd: Path | str
    run_dir: Path | str
    name: str = "ddim-training"
    stdout_path: Path | str | None = None
    stderr_path: Path | str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    expected_gpu: str | None = None

    def __post_init__(self) -> None:
        argv = tuple(str(part) for part in self.argv)
        if not argv or any("\x00" in part for part in argv):
            raise ValueError("argv must be a nonempty argument vector without NUL bytes")
        cwd = Path(self.cwd).expanduser().resolve()
        run_dir = Path(self.run_dir).expanduser().resolve()
        stdout_path = (
            Path(self.stdout_path).expanduser().resolve()
            if self.stdout_path is not None
            else run_dir / "stdout.log"
        )
        stderr_path = (
            Path(self.stderr_path).expanduser().resolve()
            if self.stderr_path is not None
            else run_dir / "stderr.log"
        )
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.name).strip("-.")
        if not name:
            raise ValueError("name must contain at least one safe character")
        normalized_env: dict[str, str] = {}
        for key, value in self.env.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
                raise ValueError(f"invalid environment variable name: {key!r}")
            normalized_env[str(key)] = _check_no_newline(value, f"environment variable {key}")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "run_dir", run_dir)
        object.__setattr__(self, "stdout_path", stdout_path)
        object.__setattr__(self, "stderr_path", stderr_path)
        object.__setattr__(self, "name", name[:120])
        object.__setattr__(self, "env", normalized_env)
        object.__setattr__(self, "resources", dict(self.resources))

    @classmethod
    def from_profile(
        cls,
        profile: object,
        *,
        run_dir: Path | str,
        argv: Sequence[str],
        cwd: Path | str | None = None,
        name: str | None = None,
        stdout_path: Path | str | None = None,
        stderr_path: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        resources: Mapping[str, Any] | None = None,
        expected_gpu: str | None = None,
    ) -> "LaunchRequest":
        profile_resources = _value(profile, "scheduler_resources", "resources", default={}) or {}
        merged_resources = dict(profile_resources)
        if resources:
            merged_resources.update(resources)
        return cls(
            argv=argv,
            cwd=cwd or _value(profile, "workspace", "working_directory", default=Path.cwd()),
            run_dir=run_dir,
            name=name or _value(profile, "id", "name", default="ddim-training"),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=env or {},
            resources=merged_resources,
            expected_gpu=expected_gpu or _value(profile, "expected_gpu", "gpu_type"),
        )


@dataclass(frozen=True)
class JobSubmission:
    backend: str
    job_id: str
    state: str
    submitted_at: str = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "job_id": self.job_id,
            "state": self.state,
            "submitted_at": self.submitted_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class JobStatus:
    backend: str
    job_id: str
    state: str
    exit_code: int | None = None
    detail: str | None = None
    checked_at: str = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "job_id": self.job_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "checked_at": self.checked_at,
            "metadata": dict(self.metadata),
        }


class ForegroundBackend:
    """Run the worker synchronously in the current terminal."""

    name = "foreground"

    def __init__(self, *, popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen):
        self._popen_factory = popen_factory
        self._statuses: dict[str, JobStatus] = {}

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def probe(profile: object | None = None) -> dict[str, Any]:
        executable = _value(profile, "python_executable", default=sys.executable)
        try:
            resolved = _resolve_executable(str(executable))
        except BackendError as exc:
            return {"available": False, "error": str(exc)}
        return {"available": True, "python_executable": resolved}

    def submit(self, request: LaunchRequest) -> JobSubmission:
        request.run_dir.mkdir(parents=True, exist_ok=True)
        request.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        request.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(request.env)
        with request.stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_handle, request.stderr_path.open(
            "a", encoding="utf-8", buffering=1
        ) as stderr_handle:
            process = self._popen_factory(
                tuple(request.argv),
                cwd=str(request.cwd),
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            job_id = f"local-{process.pid}"
            returncode = process.wait()
        state = "completed" if returncode == 0 else "failed"
        status = JobStatus(
            backend=self.name,
            job_id=job_id,
            state=state,
            exit_code=returncode,
        )
        self._statuses[job_id] = status
        return JobSubmission(
            backend=self.name,
            job_id=job_id,
            state=state,
            metadata={"returncode": returncode, "worker_argv": list(request.argv)},
        )

    def status(self, job: JobSubmission | str) -> JobStatus:
        job_id = job.job_id if isinstance(job, JobSubmission) else str(job)
        return self._statuses.get(
            job_id,
            JobStatus(
                backend=self.name,
                job_id=job_id,
                state="lost",
                detail="foreground process status is only retained by the launching process",
            ),
        )

    def stop(self, job: JobSubmission | str) -> JobStatus:
        job_id = job.job_id if isinstance(job, JobSubmission) else str(job)
        return JobStatus(
            backend=self.name,
            job_id=job_id,
            state="lost",
            detail="foreground jobs must be interrupted in their owning terminal",
        )


_TASK_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
ET.register_namespace("", _TASK_NS)


def _task_element(parent: ET.Element, name: str, text: str | None = None) -> ET.Element:
    element = ET.SubElement(parent, f"{{{_TASK_NS}}}{name}")
    if text is not None:
        element.text = text
    return element


class WindowsTaskBackend:
    """Durable Windows execution through an on-demand Scheduled Task."""

    name = "windows-task"

    def __init__(
        self,
        *,
        runner: CommandRunner = run_command,
        schtasks_executable: str | None = None,
        powershell_executable: str | None = None,
    ) -> None:
        self._runner = runner
        self.schtasks_executable = schtasks_executable or shutil.which("schtasks.exe") or "schtasks.exe"
        self.powershell_executable = (
            powershell_executable
            or shutil.which("powershell.exe")
            or shutil.which("pwsh.exe")
            or "powershell.exe"
        )

    @staticmethod
    def available() -> bool:
        return os.name == "nt" and shutil.which("schtasks.exe") is not None

    def probe(self, profile: object | None = None, *, exercise: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "available": self.available(),
            "schtasks": shutil.which("schtasks.exe"),
            "powershell": shutil.which("powershell.exe") or shutil.which("pwsh.exe"),
        }
        if not result["available"] or not exercise:
            return result
        result["exercise"] = self.exercise_probe(
            python_executable=str(_value(profile, "python_executable", default=sys.executable))
        )
        result["available"] = bool(result["exercise"].get("success"))
        return result

    @staticmethod
    def task_name(request: LaunchRequest) -> str:
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.run_dir.name)[:64]
        return f"DDIM-{request.name}-{suffix}"[:200]

    @staticmethod
    def _task_tree(request: LaunchRequest, *, user_id: str | None = None) -> ET.ElementTree:
        executable = _resolve_executable(request.argv[0])
        if request.env:
            raise ValueError(
                "Windows scheduled tasks do not accept per-process environment overrides; "
                "put operational environment configuration in the worker or machine profile"
            )
        task = ET.Element(f"{{{_TASK_NS}}}Task", {"version": "1.4"})
        registration = _task_element(task, "RegistrationInfo")
        _task_element(registration, "Description", "DDIM durable training worker")
        _task_element(task, "Triggers")
        principals = _task_element(task, "Principals")
        principal = _task_element(principals, "Principal")
        principal.set("id", "Author")
        if user_id is None:
            domain = os.environ.get("USERDOMAIN")
            user = getpass.getuser()
            user_id = f"{domain}\\{user}" if domain else user
        _task_element(principal, "UserId", user_id)
        _task_element(principal, "LogonType", "InteractiveToken")
        _task_element(principal, "RunLevel", "LeastPrivilege")
        settings = _task_element(task, "Settings")
        _task_element(settings, "MultipleInstancesPolicy", "IgnoreNew")
        _task_element(settings, "DisallowStartIfOnBatteries", "false")
        _task_element(settings, "StopIfGoingOnBatteries", "false")
        _task_element(settings, "AllowHardTerminate", "true")
        _task_element(settings, "StartWhenAvailable", "true")
        _task_element(settings, "RunOnlyIfNetworkAvailable", "false")
        _task_element(settings, "AllowStartOnDemand", "true")
        _task_element(settings, "Enabled", "true")
        _task_element(settings, "Hidden", "false")
        _task_element(settings, "ExecutionTimeLimit", "PT0S")
        _task_element(settings, "Priority", "7")
        actions = _task_element(task, "Actions")
        actions.set("Context", "Author")
        action = _task_element(actions, "Exec")
        _task_element(action, "Command", executable)
        if len(request.argv) > 1:
            _task_element(action, "Arguments", subprocess.list2cmdline(list(request.argv[1:])))
        _task_element(action, "WorkingDirectory", str(request.cwd))
        return ET.ElementTree(task)

    def render_task_xml(self, request: LaunchRequest, *, user_id: str | None = None) -> str:
        buffer = BytesIO()
        self._task_tree(request, user_id=user_id).write(
            buffer, encoding="utf-16", xml_declaration=True
        )
        return buffer.getvalue().decode("utf-16")

    def submit(self, request: LaunchRequest) -> JobSubmission:
        if not self.available() and self.schtasks_executable == "schtasks.exe":
            raise BackendUnavailableError("Windows Task Scheduler is not available")
        request.run_dir.mkdir(parents=True, exist_ok=True)
        # Keep the exact task definition with its numbered attempt.
        xml_path = request.stdout_path.parent / "windows-task.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        self._task_tree(request).write(xml_path, encoding="utf-16", xml_declaration=True)
        task_name = self.task_name(request)
        create_argv = [
            self.schtasks_executable,
            "/Create",
            "/TN",
            task_name,
            "/XML",
            str(xml_path),
            "/F",
        ]
        created = self._runner(create_argv)
        if created.returncode != 0:
            raise SubmissionError(created.stderr.strip() or created.stdout.strip())
        start_argv = [self.schtasks_executable, "/Run", "/TN", task_name]
        started = self._runner(start_argv)
        if started.returncode != 0:
            raise SubmissionError(started.stderr.strip() or started.stdout.strip())
        return JobSubmission(
            backend=self.name,
            job_id=task_name,
            state="submitted",
            metadata={
                "task_xml": str(xml_path),
                "create_argv": create_argv,
                "start_argv": start_argv,
            },
        )

    def _powershell(self, source: str) -> CommandResult:
        encoded = base64.b64encode(source.encode("utf-16le")).decode("ascii")
        return self._runner(
            [
                self.powershell_executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ]
        )

    def status(self, job: JobSubmission | str) -> JobStatus:
        task_name = job.job_id if isinstance(job, JobSubmission) else str(job)
        safe_name = task_name.replace("'", "''")
        result = self._powershell(
            "$ErrorActionPreference='Stop';"
            f"$task=Get-ScheduledTask -TaskName '{safe_name}';"
            f"$info=Get-ScheduledTaskInfo -TaskName '{safe_name}';"
            "[pscustomobject]@{State=[string]$task.State;"
            "LastTaskResult=[int64]$info.LastTaskResult;"
            "LastRunTime=[string]$info.LastRunTime}|ConvertTo-Json -Compress"
        )
        if result.returncode != 0:
            return JobStatus(
                backend=self.name,
                job_id=task_name,
                state="lost",
                detail=result.stderr.strip() or "scheduled task was not found",
            )
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            return JobStatus(
                backend=self.name,
                job_id=task_name,
                state="lost",
                detail=f"could not parse Task Scheduler status: {exc}",
            )
        raw_state = str(payload.get("State", "Unknown"))
        raw_result = int(payload.get("LastTaskResult", 0))
        if raw_state.casefold() == "running":
            state, exit_code = "running", None
        elif raw_state.casefold() == "queued":
            state, exit_code = "queued", None
        elif raw_result in {267008, 267009, 267045}:
            state, exit_code = "submitted", None
        elif raw_result == 0:
            state, exit_code = "completed", 0
        elif raw_result in {0xC000013A, -1073741510}:
            state, exit_code = "interrupted", raw_result
        else:
            state, exit_code = "failed", raw_result
        return JobStatus(
            backend=self.name,
            job_id=task_name,
            state=state,
            exit_code=exit_code,
            detail=raw_state,
            metadata=payload,
        )

    def stop(self, job: JobSubmission | str) -> JobStatus:
        task_name = job.job_id if isinstance(job, JobSubmission) else str(job)
        safe_name = task_name.replace("'", "''")
        result = self._powershell(
            "$ErrorActionPreference='Stop';"
            f"Stop-ScheduledTask -TaskName '{safe_name}'"
        )
        if result.returncode != 0:
            raise BackendError(result.stderr.strip() or f"could not stop task {task_name}")
        return JobStatus(backend=self.name, job_id=task_name, state="interrupted")

    def remove(self, job: JobSubmission | str) -> None:
        task_name = job.job_id if isinstance(job, JobSubmission) else str(job)
        result = self._runner(
            [self.schtasks_executable, "/Delete", "/TN", task_name, "/F"]
        )
        if result.returncode != 0:
            raise BackendError(result.stderr.strip() or f"could not remove task {task_name}")

    def exercise_probe(self, *, python_executable: str = sys.executable) -> dict[str, Any]:
        """Create, run, verify, and remove a harmless user task."""

        with tempfile.TemporaryDirectory(prefix="ddim-task-probe-") as temp_dir:
            root = Path(temp_dir).resolve()
            marker = root / "ok.txt"
            request = LaunchRequest(
                argv=(
                    _resolve_executable(python_executable),
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
                    str(marker),
                ),
                cwd=root,
                run_dir=root,
                name=f"probe-{os.getpid()}",
            )
            submission: JobSubmission | None = None
            try:
                submission = self.submit(request)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline and not marker.exists():
                    time.sleep(0.2)
                return {"success": marker.exists(), "task_name": submission.job_id}
            except Exception as exc:  # The doctor should report, rather than hide, policy failures.
                return {"success": False, "error": str(exc)}
            finally:
                if submission is not None:
                    try:
                        self.remove(submission)
                    except BackendError:
                        pass


TaskSchedulerBackend = WindowsTaskBackend


@dataclass(frozen=True)
class SlurmResources:
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodes: int = 1
    ntasks: int = 1
    gpus: int = 1
    cpus_per_task: int = 4
    memory: str = "32G"
    time_limit: str = "1-00:00:00"
    signal_seconds: int = 300
    constraint: str | None = None
    gpu_type: str | None = None

    def __post_init__(self) -> None:
        for name in ("nodes", "ntasks", "gpus", "cpus_per_task", "signal_seconds"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("partition", "account", "qos", "memory", "time_limit", "constraint", "gpu_type"):
            value = getattr(self, name)
            if value is not None:
                _check_no_newline(value, name)

    @classmethod
    def from_sources(
        cls, profile: object | None = None, overrides: Mapping[str, Any] | None = None
    ) -> "SlurmResources":
        nested = _value(profile, "slurm", "scheduler_resources", "resources", default={}) or {}

        def choose(name: str, *aliases: str, default: Any = None) -> Any:
            if overrides and name in overrides:
                return overrides[name]
            found = _value(nested, name, *aliases, default=None)
            if found is not None:
                return found
            return _value(profile, name, *aliases, default=default)

        memory = choose("memory", "mem")
        if memory is None:
            memory_gb = choose("memory_gb")
            memory = f"{int(memory_gb)}G" if memory_gb is not None else "32G"
        return cls(
            partition=choose("partition", "slurm_partition"),
            account=choose("account", "slurm_account"),
            qos=choose("qos", "slurm_qos"),
            nodes=int(choose("nodes", default=1)),
            ntasks=int(choose("ntasks", default=1)),
            gpus=int(choose("gpus", "gpu_count", default=1)),
            cpus_per_task=int(choose("cpus_per_task", "cpus", default=4)),
            memory=str(memory),
            time_limit=str(choose("time_limit", "walltime", default="1-00:00:00")),
            signal_seconds=int(choose("signal_seconds", default=300)),
            constraint=choose("constraint"),
            gpu_type=choose("gpu_type"),
        )


def _sbatch_value(value: object) -> str:
    checked = _check_no_newline(value, "SBATCH value")
    if re.fullmatch(r"[A-Za-z0-9_./:+@%=-]+", checked):
        return checked
    return '"' + checked.replace("\\", "\\\\").replace('"', '\\"') + '"'


_SLURM_STATES = {
    "PENDING": "queued",
    "CONFIGURING": "queued",
    "RESV_DEL_HOLD": "queued",
    "REQUEUE_FED": "queued",
    "REQUEUED": "queued",
    "RUNNING": "running",
    "COMPLETING": "running",
    "STAGE_OUT": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "PREEMPTED": "interrupted",
    "REVOKED": "interrupted",
    "CANCELLED": "cancelled",
    "TIMEOUT": "timed_out",
    "SUSPENDED": "interrupted",
}


class SlurmBackend:
    name = "slurm"

    def __init__(
        self,
        *,
        runner: CommandRunner = run_command,
        sbatch_executable: str | None = None,
        squeue_executable: str | None = None,
        sacct_executable: str | None = None,
        scancel_executable: str | None = None,
    ) -> None:
        self._runner = runner
        self.sbatch_executable = sbatch_executable or shutil.which("sbatch") or "sbatch"
        self.squeue_executable = squeue_executable or shutil.which("squeue") or "squeue"
        self.sacct_executable = sacct_executable or shutil.which("sacct") or "sacct"
        self.scancel_executable = scancel_executable or shutil.which("scancel") or "scancel"

    @staticmethod
    def available() -> bool:
        return all(shutil.which(command) for command in ("sbatch", "squeue", "sacct", "scancel"))

    def probe(self, profile: object | None = None, *, exercise: bool = False) -> dict[str, Any]:
        binaries = {
            command: shutil.which(command)
            for command in ("sbatch", "squeue", "sacct", "scancel", "sinfo")
        }
        result: dict[str, Any] = {
            "available": all(binaries[name] for name in ("sbatch", "squeue", "sacct", "scancel")),
            "binaries": binaries,
            "resources": SlurmResources.from_sources(profile).__dict__,
        }
        if binaries.get("sinfo"):
            response = self._runner([str(binaries["sinfo"]), "--noheader", "--format=%P|%a|%l|%G"])
            result["partitions"] = response.stdout.strip().splitlines() if response.returncode == 0 else []
            if response.returncode != 0:
                result["error"] = response.stderr.strip()
        if exercise and result["available"]:
            resources = SlurmResources.from_sources(profile)
            with tempfile.TemporaryDirectory(prefix="ddim-slurm-probe-") as temp_dir:
                probe_root = Path(temp_dir).resolve()
                request = LaunchRequest(
                    argv=(sys.executable, "-c", "raise SystemExit(0)"),
                    cwd=Path.cwd(),
                    run_dir=probe_root,
                    stdout_path=probe_root / "stdout.log",
                    stderr_path=probe_root / "stderr.log",
                    name="ddim-doctor",
                    resources=resources.__dict__,
                )
                script = probe_root / "job.sbatch"
                script.write_text(self.render_script(request, resources), encoding="utf-8", newline="\n")
                checked = self._runner([self.sbatch_executable, "--test-only", str(script)])
                result["test_only"] = {
                    "success": checked.returncode == 0,
                    "stdout": checked.stdout.strip(),
                    "stderr": checked.stderr.strip(),
                }
                result["available"] = bool(result["test_only"]["success"])
        return result

    @staticmethod
    def render_script(
        request: LaunchRequest,
        resources: SlurmResources | None = None,
    ) -> str:
        resources = resources or SlurmResources.from_sources(overrides=request.resources)
        gpu_resource = (
            f"gpu:{resources.gpu_type}:{resources.gpus}"
            if resources.gpu_type
            else f"gpu:{resources.gpus}"
        )
        directives = [
            ("job-name", request.name),
            ("chdir", request.cwd),
            ("output", request.stdout_path),
            ("error", request.stderr_path),
            ("open-mode", "append"),
            ("nodes", resources.nodes),
            ("ntasks", resources.ntasks),
            ("gres", gpu_resource),
            ("cpus-per-task", resources.cpus_per_task),
            ("mem", resources.memory),
            ("time", resources.time_limit),
            ("signal", f"B:USR1@{resources.signal_seconds}"),
        ]
        if resources.partition:
            directives.append(("partition", resources.partition))
        if resources.account:
            directives.append(("account", resources.account))
        if resources.qos:
            directives.append(("qos", resources.qos))
        if resources.constraint:
            directives.append(("constraint", resources.constraint))
        lines = ["#!/usr/bin/env bash"]
        lines.extend(f"#SBATCH --{key}={_sbatch_value(value)}" for key, value in directives)
        lines.extend(["#SBATCH --no-requeue", "", "set -euo pipefail"])
        environment = {"PYTHONUNBUFFERED": "1", **request.env}
        for key, value in sorted(environment.items()):
            lines.append(f"export {key}={shlex.quote(value)}")
        if request.expected_gpu:
            expected = shlex.quote(request.expected_gpu)
            lines.extend(
                [
                    "if ! command -v nvidia-smi >/dev/null 2>&1; then",
                    "  echo 'nvidia-smi is required on the allocated compute node' >&2",
                    "  exit 70",
                    "fi",
                    "if ! nvidia-smi --query-gpu=name --format=csv,noheader "
                    f"| grep -Fi -- {expected} >/dev/null; then",
                    f"  echo {shlex.quote('expected GPU not found: ' + request.expected_gpu)} >&2",
                    "  exit 70",
                    "fi",
                ]
            )
        lines.append(f"exec srun --unbuffered {shlex.join(list(request.argv))}")
        return "\n".join(lines) + "\n"

    def submit(
        self,
        request: LaunchRequest,
        *,
        profile: object | None = None,
        script_path: Path | str | None = None,
    ) -> JobSubmission:
        resources = SlurmResources.from_sources(profile, request.resources)
        request.run_dir.mkdir(parents=True, exist_ok=True)
        target = (
            Path(script_path).resolve()
            if script_path
            else request.stdout_path.parent / "job.sbatch"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render_script(request, resources), encoding="utf-8", newline="\n")
        try:
            target.chmod(target.stat().st_mode | 0o100)
        except OSError:
            pass
        submit_argv = [self.sbatch_executable, "--parsable", str(target)]
        response = self._runner(submit_argv)
        if response.returncode != 0:
            raise SubmissionError(response.stderr.strip() or response.stdout.strip())
        raw_id = response.stdout.strip().splitlines()[0] if response.stdout.strip() else ""
        job_id = raw_id.split(";", 1)[0]
        if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
            raise SubmissionError(f"sbatch returned an invalid parsable job id: {raw_id!r}")
        return JobSubmission(
            backend=self.name,
            job_id=job_id,
            state="submitted",
            metadata={
                "script": str(target),
                "submit_argv": submit_argv,
                "sbatch_response": raw_id,
            },
        )

    @staticmethod
    def _state(raw_state: str) -> str:
        normalized = raw_state.strip().split()[0].rstrip("+").upper() if raw_state.strip() else ""
        return _SLURM_STATES.get(normalized, "lost")

    def status(self, job: JobSubmission | str) -> JobStatus:
        job_id = job.job_id if isinstance(job, JobSubmission) else str(job)
        if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
            raise ValueError(f"invalid Slurm job id: {job_id!r}")
        queued = self._runner(
            [self.squeue_executable, "--noheader", "--jobs", job_id, "--format=%T"]
        )
        if queued.returncode == 0 and queued.stdout.strip():
            raw = queued.stdout.strip().splitlines()[0]
            return JobStatus(
                backend=self.name,
                job_id=job_id,
                state=self._state(raw),
                detail=raw,
            )
        accounting = self._runner(
            [
                self.sacct_executable,
                "-X",
                "--noheader",
                "--parsable2",
                "--jobs",
                job_id,
                "--format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS",
            ]
        )
        if accounting.returncode != 0:
            return JobStatus(
                backend=self.name,
                job_id=job_id,
                state="lost",
                detail=accounting.stderr.strip() or queued.stderr.strip(),
            )
        rows = [line.split("|") for line in accounting.stdout.splitlines() if line.strip()]
        row = next((parts for parts in rows if parts and parts[0] == job_id), None)
        if row is None or len(row) < 3:
            return JobStatus(
                backend=self.name,
                job_id=job_id,
                state="lost",
                detail="job was absent from both squeue and sacct",
            )
        raw_state, raw_exit = row[1], row[2]
        try:
            exit_code = int(raw_exit.split(":", 1)[0])
        except ValueError:
            exit_code = None
        return JobStatus(
            backend=self.name,
            job_id=job_id,
            state=self._state(raw_state),
            exit_code=exit_code,
            detail=raw_state,
            metadata={
                "slurm_exit_code": raw_exit,
                "elapsed": row[3] if len(row) > 3 else None,
                "max_rss": row[4] if len(row) > 4 else None,
            },
        )

    def stop(self, job: JobSubmission | str) -> JobStatus:
        job_id = job.job_id if isinstance(job, JobSubmission) else str(job)
        if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
            raise ValueError(f"invalid Slurm job id: {job_id!r}")
        result = self._runner([self.scancel_executable, "--signal=TERM", job_id])
        if result.returncode != 0:
            raise BackendError(result.stderr.strip() or f"could not cancel Slurm job {job_id}")
        return JobStatus(backend=self.name, job_id=job_id, state="interrupted")


class ExternalHPCBackend:
    """Prepare a scheduler-neutral bundle for a corporate portal or scheduler."""

    name = "external-hpc"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def render_worker_script(request: LaunchRequest) -> str:
        expected_gpu = request.expected_gpu or str(request.resources.get("gpu_type", "")).strip()
        minimum_gpus = int(request.resources.get("gpus", request.resources.get("gpu_count", 1)))
        if minimum_gpus <= 0:
            raise ValueError("external HPC worker requires at least one GPU")
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "if ! command -v nvidia-smi >/dev/null 2>&1; then",
            "  echo 'nvidia-smi is required; refusing to train on a login or CPU-only node' >&2",
            "  exit 70",
            "fi",
            "mapfile -t DDIM_GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)",
            f"if (( ${{#DDIM_GPU_NAMES[@]}} < {minimum_gpus} )); then",
            f"  echo 'at least {minimum_gpus} compute GPU(s) required' >&2",
            "  exit 70",
            "fi",
        ]
        if expected_gpu:
            expected = shlex.quote(expected_gpu.casefold())
            lines.extend(
                [
                    "DDIM_GPU_LIST=$(printf '%s\\n' \"${DDIM_GPU_NAMES[@]}\" | tr '[:upper:]' '[:lower:]')",
                    f"if ! grep -F -- {expected} <<<\"$DDIM_GPU_LIST\" >/dev/null; then",
                    f"  echo {shlex.quote('expected GPU not found: ' + expected_gpu)} >&2",
                    "  exit 70",
                    "fi",
                ]
            )
        lines.append(f"cd {shlex.quote(str(request.cwd))}")
        for key, value in sorted(request.env.items()):
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.append(f"exec {shlex.join(list(request.argv))}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def render_probe_script() -> str:
        return """#!/usr/bin/env bash
set -euo pipefail
DDIM_PROBE_PYTHON=${PYTHON:-python3}
exec "$DDIM_PROBE_PYTHON" -m ddimctl.hpc_probe "$@"
"""

    def submit(
        self,
        request: LaunchRequest,
        *,
        script_path: Path | str | None = None,
    ) -> JobSubmission:
        request.run_dir.mkdir(parents=True, exist_ok=True)
        worker_path = Path(script_path).resolve() if script_path else request.run_dir / "worker.sh"
        probe_path = request.run_dir / "probe.sh"
        worker_path.write_text(self.render_worker_script(request), encoding="utf-8", newline="\n")
        probe_path.write_text(self.render_probe_script(), encoding="utf-8", newline="\n")
        for path in (worker_path, probe_path):
            try:
                path.chmod(path.stat().st_mode | 0o100)
            except OSError:
                pass
        return JobSubmission(
            backend=self.name,
            job_id=request.run_dir.name,
            state="prepared",
            metadata={
                "worker_script": str(worker_path),
                "probe_script": str(probe_path),
                "run_dir": str(request.run_dir),
                "instruction": f"submit {worker_path} through the approved corporate scheduler",
            },
        )

    @staticmethod
    def probe(profile: object | None = None) -> dict[str, Any]:
        from .hpc_probe import collect_hpc_probe

        paths: dict[str, str] = {}
        for name in ("workspace", "runs_root", "dataset_root"):
            value = _value(profile, name)
            if value:
                paths[name] = str(value)
        return collect_hpc_probe(paths=paths)

    def status(self, run: JobSubmission | Path | str) -> JobStatus:
        if isinstance(run, JobSubmission):
            run_dir = Path(str(run.metadata.get("run_dir", run.job_id)))
            job_id = run.job_id
        else:
            run_dir = Path(run)
            job_id = run_dir.name
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            return JobStatus(
                backend=self.name,
                job_id=job_id,
                state="prepared",
                detail="awaiting external scheduler invocation",
            )
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            state = str(payload.get("state", payload.get("status", "lost")))
            exit_code = payload.get("exit_code")
            return JobStatus(
                backend=self.name,
                job_id=job_id,
                state=state,
                exit_code=int(exit_code) if exit_code is not None else None,
                metadata=payload,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return JobStatus(
                backend=self.name,
                job_id=job_id,
                state="lost",
                detail=f"invalid state file: {exc}",
            )

    def stop(self, run_dir: Path | str) -> JobStatus:
        root = Path(run_dir).resolve()
        request_path = root / "stop.request"
        temporary = request_path.with_name(f".{request_path.name}.{os.getpid()}.tmp")
        temporary.write_text(f"{_utc_now()} user_requested\n", encoding="utf-8")
        os.replace(temporary, request_path)
        return JobStatus(
            backend=self.name,
            job_id=root.name,
            state="interrupted",
            detail="graceful stop requested; external scheduler cancellation may still be required",
        )


def get_backend(name: str, **kwargs: Any) -> Any:
    normalized = name.strip().casefold().replace("_", "-")
    if normalized in {"foreground", "local"}:
        return ForegroundBackend(**kwargs)
    if normalized in {"windows-task", "task-scheduler", "windows-task-scheduler"}:
        return WindowsTaskBackend(**kwargs)
    if normalized == "slurm":
        return SlurmBackend(**kwargs)
    if normalized in {"external-hpc", "external", "portal"}:
        return ExternalHPCBackend()
    raise ValueError(f"unknown execution backend: {name}")


def detect_backend(profile: object | None = None) -> str:
    """Resolve an explicit profile executor or choose the safest native backend."""

    configured = _value(profile, "executor", "backend", "execution_backend")
    if configured and str(configured).casefold() not in {"auto", "detect"}:
        normalized = str(configured).strip().casefold().replace("_", "-")
        aliases = {
            "local": "foreground",
            "task-scheduler": "windows-task",
            "windows-task-scheduler": "windows-task",
            "external": "external-hpc",
            "portal": "external-hpc",
        }
        return aliases.get(normalized, normalized)
    if SlurmBackend.available():
        return "slurm"
    if WindowsTaskBackend.available():
        return "windows-task"
    return "foreground"


__all__ = [
    "BackendError",
    "BackendUnavailableError",
    "CommandResult",
    "ExternalHPCBackend",
    "ForegroundBackend",
    "JobStatus",
    "JobSubmission",
    "LaunchRequest",
    "SlurmBackend",
    "SlurmResources",
    "SubmissionError",
    "TaskSchedulerBackend",
    "WindowsTaskBackend",
    "detect_backend",
    "get_backend",
    "run_command",
]
