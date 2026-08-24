"""Platform-local machine profile persistence."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Mapping

from .bundles import atomic_write_json
from .schemas import MachineProfile


_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def user_config_dir(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return the OS-standard per-user configuration directory for ddimctl."""

    env = os.environ if environ is None else environ
    platform_name = sys.platform if platform is None else platform
    home_dir = Path.home() if home is None else home

    if platform_name == "win32":
        base = env.get("APPDATA") or env.get("LOCALAPPDATA")
        return (Path(base) if base else home_dir / "AppData" / "Roaming") / "ddimctl"
    if platform_name == "darwin":
        return home_dir / "Library" / "Application Support" / "ddimctl"
    base = env.get("XDG_CONFIG_HOME")
    return (Path(base) if base else home_dir / ".config") / "ddimctl"


def _validate_machine_id(machine_id: str) -> None:
    if not _PROFILE_ID.fullmatch(machine_id):
        raise ValueError(
            "machine_id must start with a letter or digit and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )


def profile_path(machine_id: str, config_dir: Path | str | None = None) -> Path:
    _validate_machine_id(machine_id)
    root = user_config_dir() if config_dir is None else Path(config_dir)
    return root / "profiles" / f"{machine_id}.json"


def save_profile(
    profile: MachineProfile,
    config_dir: Path | str | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically save a machine profile outside the repository."""

    destination = profile_path(profile.machine_id, config_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"machine profile already exists: {destination}")
    atomic_write_json(destination, profile.model_dump(mode="json", exclude_none=True))
    return destination


def load_profile(
    machine_id: str,
    config_dir: Path | str | None = None,
) -> MachineProfile:
    path = profile_path(machine_id, config_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"machine profile not found: {path}") from exc
    profile = MachineProfile.model_validate_json(raw)
    if profile.machine_id != machine_id:
        raise ValueError(
            f"profile identity mismatch: requested {machine_id!r}, "
            f"file contains {profile.machine_id!r}"
        )
    return profile


def list_profiles(config_dir: Path | str | None = None) -> tuple[str, ...]:
    root = user_config_dir() if config_dir is None else Path(config_dir)
    profiles_dir = root / "profiles"
    if not profiles_dir.is_dir():
        return ()
    return tuple(
        sorted(
            path.stem
            for path in profiles_dir.glob("*.json")
            if _PROFILE_ID.fullmatch(path.stem)
        )
    )
