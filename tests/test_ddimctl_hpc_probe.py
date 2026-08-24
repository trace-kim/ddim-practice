from __future__ import annotations

import hashlib
import json

from ddimctl.hpc_probe import collect_hpc_probe, write_hpc_probe


def test_offline_probe_reports_runtime_scheduler_and_paths(tmp_path):
    report = collect_hpc_probe(
        paths={"workspace": tmp_path, "future_runs": tmp_path / "runs"},
        include_torch=False,
        which=lambda _name: None,
    )

    assert report["offline"] is True
    assert report["scheduler"]["detected"] == "unknown"
    assert report["scheduler"]["native_slurm_ready"] is False
    assert report["torch"] == {"skipped": True}
    assert report["paths"]["workspace"]["exists"] is True
    assert report["paths"]["future_runs"]["existing_ancestor"] == str(tmp_path.resolve())


def test_probe_write_is_json_with_matching_checksum(tmp_path):
    report = {"schema_version": 1, "offline": True}
    target = write_hpc_probe(report, tmp_path / "probe.json")
    serialized = target.read_bytes()
    checksum = (tmp_path / "probe.json.sha256").read_text(encoding="ascii").split()[0]

    assert json.loads(serialized) == report
    assert checksum == hashlib.sha256(serialized).hexdigest()
