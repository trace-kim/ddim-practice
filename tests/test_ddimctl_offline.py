from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ddimctl.offline import OfflineBundleError, build_wheelhouse, verify_wheelhouse


def test_wheelhouse_build_is_atomic_portable_and_verifiable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    output = tmp_path / "offline-wheels"

    def fake_pip(argv, **kwargs):
        wheel_dir = Path(argv[argv.index("--wheel-dir") + 1])
        (wheel_dir / "ddim_training_workflow-0.1.0-py3-none-any.whl").write_bytes(b"project")
        (wheel_dir / "numpy-2.0.0-cp313-cp313-win_amd64.whl").write_bytes(b"numpy")
        return subprocess.CompletedProcess(argv, 0, "built", "")

    built = build_wheelhouse(repo, output, runner=fake_pip)
    report = verify_wheelhouse(built)

    assert report["verified"] is True
    assert report["files"] == 2
    assert "exec python -m pip install --no-index" in (built / "install.sh").read_text(
        encoding="utf-8"
    )
    assert not list(tmp_path.glob(".offline-wheels.building-*"))
    standalone = subprocess.run(
        [sys.executable, "verify.py"], cwd=built, text=True, capture_output=True
    )
    assert standalone.returncode == 0
    assert "verified 2 wheels" in standalone.stdout
    with pytest.raises(FileExistsError):
        build_wheelhouse(repo, output, runner=fake_pip)


def test_wheelhouse_verifier_detects_tampering(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    def fake_pip(argv, **kwargs):
        wheel_dir = Path(argv[argv.index("--wheel-dir") + 1])
        (wheel_dir / "ddim_training_workflow-0.1.0-py3-none-any.whl").write_bytes(b"clean")
        return subprocess.CompletedProcess(argv, 0, "", "")

    bundle = build_wheelhouse(repo, tmp_path / "bundle", runner=fake_pip)
    wheel = next((bundle / "wheels").glob("*.whl"))
    wheel.write_bytes(b"tampered")

    with pytest.raises(OfflineBundleError, match="mismatch"):
        verify_wheelhouse(bundle)
