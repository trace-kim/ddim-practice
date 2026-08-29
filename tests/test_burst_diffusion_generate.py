from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from burst_diffusion.generate import generate_burst_dataset, select_sources
from burst_diffusion.preview import make_preview_grid


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _make_sources(root: Path, count: int = 3) -> Path:
    source_dir = root / "sources"
    for index in range(count):
        gradient = np.linspace(0, 255, 12 * 9, dtype=np.float64).reshape(12, 9)
        array = np.rint((gradient + index * 7) % 256).astype(np.uint8)
        _save_image(source_dir / f"img_{index}.png", array)
    return source_dir


def _generate(tmp_path: Path, **overrides) -> Path:
    kwargs = dict(
        source_dir=_make_sources(tmp_path),
        output_dir=tmp_path / "out",
        num_sources=3,
        replicas=3,
        noise_type="poisson",
        noise_params={"poisson": {"peak": 30.0}},
        margin=0.15,
        max_side=None,
        overwrite=False,
        progress=False,
    )
    kwargs.update(overrides)
    return generate_burst_dataset(**kwargs)


def test_generates_burst_layout_with_stats_and_provenance(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):  # tiny 12x9 images make PSNR bands noisy; warnings expected
        stats_path = _generate(tmp_path)
    out = tmp_path / "out"
    assert stats_path == out / "stats.json"
    assert sorted(p.name for p in (out / "_sources").iterdir()) == [
        "00000.png", "00001.png", "00002.png",
    ]
    assert (out / "burst" / "manifest.jsonl").is_file()
    assert len(list((out / "burst" / "noisy").glob("*.png"))) == 9

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    aggregate = stats["aggregate"]
    for key in (
        "single_frame_psnr_median",
        "single_frame_psnr_p10",
        "avg_of_n_psnr_median",
        "bias_abs_median",
    ):
        assert np.isfinite(aggregate[key])
    assert len(stats["per_source"]) == 3

    provenance = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    assert [row["original"] for row in provenance["sources"]] == [
        "img_0.png", "img_1.png", "img_2.png",
    ]


def test_margin_prescale_keeps_staged_values_inside_the_headroom_band(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    extremes = np.zeros((10, 10), dtype=np.uint8)
    extremes[:, 5:] = 255
    _save_image(source_dir / "extremes.png", extremes)
    with pytest.warns(UserWarning):
        generate_burst_dataset(
            source_dir=source_dir, output_dir=tmp_path / "out", num_sources=1, replicas=2,
            noise_type="poisson", noise_params={"poisson": {"peak": 30.0}},
            margin=0.15, max_side=None, progress=False,
        )
    staged = np.asarray(Image.open(tmp_path / "out" / "_sources" / "00000.png"))
    assert staged.min() == round(0.15 * 255)
    assert staged.max() == round(0.85 * 255)


def test_grayscale_and_max_side_apply_to_staged_clean_only(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    rgb = np.zeros((40, 20, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    _save_image(source_dir / "wide.png", rgb)
    with pytest.warns(UserWarning):
        generate_burst_dataset(
            source_dir=source_dir, output_dir=tmp_path / "out", num_sources=1, replicas=2,
            noise_type="poisson", noise_params={"poisson": {"peak": 30.0}},
            margin=0.1, max_side=20, progress=False,
        )
    staged = Image.open(tmp_path / "out" / "_sources" / "00000.png")
    assert staged.mode == "L"
    assert staged.size == (10, 20)  # (width, height): 40x20 -> 20x10 spatial


def test_source_selection_is_seed_deterministic(tmp_path: Path) -> None:
    source_dir = _make_sources(tmp_path, count=8)
    first = select_sources(source_dir, 4, seed=7)
    second = select_sources(source_dir, 4, seed=7)
    other = select_sources(source_dir, 4, seed=8)
    assert first == second
    assert first != other
    assert first == sorted(first)


def test_requesting_more_sources_than_available_warns_and_uses_all(tmp_path: Path) -> None:
    source_dir = _make_sources(tmp_path, count=2)
    with pytest.warns(UserWarning, match="using all of them"):
        selected = select_sources(source_dir, 5, seed=0)
    assert len(selected) == 2


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        _generate(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _generate(tmp_path)
    with pytest.warns(UserWarning):
        _generate(tmp_path, overwrite=True)


def test_output_inside_source_dir_is_rejected(tmp_path: Path) -> None:
    source_dir = _make_sources(tmp_path)
    with pytest.raises(ValueError, match="must not live inside"):
        generate_burst_dataset(
            source_dir=source_dir, output_dir=source_dir / "out", num_sources=1,
            replicas=2, noise_type="poisson", progress=False,
        )


def test_too_clean_and_salt_pepper_warnings_fire(tmp_path: Path) -> None:
    source_dir = _make_sources(tmp_path)
    with pytest.warns(UserWarning) as captured:
        generate_burst_dataset(
            source_dir=source_dir, output_dir=tmp_path / "out", num_sources=3, replicas=2,
            noise_type=["gaussian", "salt_pepper"],
            noise_params={"gaussian": {"std": 1e-4}, "salt_pepper": {"amount": 1e-4}},
            margin=0.15, max_side=None, progress=False,
        )
    messages = [str(warning.message) for warning in captured]
    assert any("too clean" in message for message in messages)
    assert any("salt_pepper" in message for message in messages)
    stats = json.loads((tmp_path / "out" / "stats.json").read_text(encoding="utf-8"))
    assert len(stats["warnings"]) >= 2


def test_preview_grid_is_written_with_expected_geometry(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        _generate(tmp_path)
    out_path = tmp_path / "grids" / "preview.png"
    result = make_preview_grid(
        tmp_path / "out", source_index=0, out_path=out_path, avg_counts=(1, 2, 16), max_tile=64,
    )
    assert result == out_path
    grid = Image.open(out_path)
    # 3 replicas clamp avg_counts to (1, 2, 3); columns = 1 + 3, tiles 12x9 (h x w).
    pad, caption = 4, 16
    assert grid.size == (4 * (9 + pad) + pad, 2 * (12 + caption + pad) + pad)


def test_preview_rejects_missing_source_index(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        _generate(tmp_path)
    with pytest.raises(FileNotFoundError, match="no clean image for source 9"):
        make_preview_grid(tmp_path / "out", source_index=9, out_path=tmp_path / "p.png")
