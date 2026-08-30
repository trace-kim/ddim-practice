from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from burst_diffusion.data import BatchFactory, BurstCache, content_key, resolve_burst_dir


def _write_burst(
    root: Path,
    sources: dict[int, tuple[np.ndarray, list[np.ndarray]]],
    *,
    manifest_tail: str | None = None,
) -> Path:
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True, exist_ok=True)
    (burst / "noisy").mkdir(parents=True, exist_ok=True)
    rows = []
    for source_index, (clean, frames) in sources.items():
        clean_name = f"clean/{source_index:05d}.png"
        Image.fromarray(clean).save(burst / clean_name)
        for replica_index, frame in enumerate(frames):
            noisy_name = f"noisy/{source_index:05d}_{replica_index:05d}.png"
            Image.fromarray(frame).save(burst / noisy_name)
            rows.append(
                json.dumps(
                    {
                        "source_index": source_index,
                        "replica_index": replica_index,
                        "clean_path": clean_name,
                        "noisy_path": noisy_name,
                    },
                    sort_keys=True,
                )
            )
    text = "\n".join(rows) + "\n"
    if manifest_tail is not None:
        text += manifest_tail
    (burst / "manifest.jsonl").write_text(text, encoding="utf-8")
    return root


def _gradient(height: int = 20, width: int = 24, offset: int = 0) -> np.ndarray:
    """Distinct-per-``offset`` gradient: sources must not share clean content,
    or the content-group split correctly collapses them into one group."""
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    return np.rint((rows * 7 + cols * 3 + offset * 11) % 256).astype(np.uint8)


def _standard_dataset(tmp_path: Path, *, num_sources: int = 5, replicas: int = 5) -> Path:
    rng = np.random.default_rng(0)
    sources = {}
    for index in range(num_sources):
        clean = _gradient(offset=index)
        frames = [
            np.clip(
                clean.astype(np.int16) + rng.integers(-40, 41, clean.shape), 0, 255
            ).astype(np.uint8)
            for _ in range(replicas)
        ]
        sources[index] = (clean, frames)
    return _write_burst(tmp_path, sources)


def _cache(root: Path, **overrides) -> BurstCache:
    kwargs = dict(channels=1, min_replicas=4, min_size=16, val_fraction=0.2, split_seed=7)
    kwargs.update(overrides)
    return BurstCache(root, **kwargs)


def _factory(cache: BurstCache, **overrides) -> BatchFactory:
    kwargs = dict(num_steps=3, image_size=16, batch_size=6, antithetic=True, seed=11)
    kwargs.update(overrides)
    return BatchFactory(cache, **kwargs)


def test_resolve_burst_dir_accepts_root_or_burst_dir(tmp_path: Path) -> None:
    root = _standard_dataset(tmp_path)
    assert resolve_burst_dir(root) == root / "burst"
    assert resolve_burst_dir(root / "burst") == root / "burst"
    with pytest.raises(FileNotFoundError, match="no manifest.jsonl"):
        resolve_burst_dir(tmp_path / "elsewhere")


def test_cache_groups_frames_by_source_in_replica_order(tmp_path: Path) -> None:
    root = _standard_dataset(tmp_path, num_sources=3)
    cache = _cache(root, val_fraction=0.0)
    assert len(cache.train_sources) == 3
    assert not cache.val_sources
    for source in cache.train_sources:
        assert len(source.frames) == 5
        assert source.clean.shape == (20, 24)
        assert source.clean.dtype == np.uint8
    summary = cache.summary()
    assert summary["min_frames"] == 5
    assert summary["ram_bytes"] == 3 * (20 * 24) * 6


def test_under_replicated_and_undersized_sources_are_dropped_with_warnings(tmp_path: Path) -> None:
    clean = _gradient()
    small = _gradient(8, 8)
    sources = {
        0: (clean, [clean] * 5),
        1: (clean, [clean] * 2),  # too few replicas
        2: (small, [small] * 5),  # too small
    }
    root = _write_burst(tmp_path, sources)
    with pytest.warns(UserWarning) as captured:
        cache = _cache(root, val_fraction=0.0)
    messages = [str(w.message) for w in captured]
    assert any("fewer than 4 replicas" in m for m in messages)
    assert any("smaller than 16 px" in m for m in messages)
    assert [s.source_index for s in cache.train_sources] == [0]
    assert cache.dropped_replicas == [1]
    assert cache.dropped_size == [2]


def test_all_sources_dropped_raises_with_counts(tmp_path: Path) -> None:
    clean = _gradient()
    root = _write_burst(tmp_path, {0: (clean, [clean] * 2)})
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="no usable sources"):
        _cache(root)


def test_truncated_final_manifest_line_is_skipped_but_corrupt_middle_raises(tmp_path: Path) -> None:
    root = _standard_dataset(tmp_path, num_sources=2)
    manifest = root / "burst" / "manifest.jsonl"
    original = manifest.read_text(encoding="utf-8")
    manifest.write_text(original + '{"source_index": 9, "replica', encoding="utf-8")
    with pytest.warns(UserWarning, match="truncated final manifest line"):
        cache = _cache(root, val_fraction=0.0)
    assert len(cache.train_sources) == 2

    lines = original.splitlines()
    lines[1] = "not json at all"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed manifest line 2"):
        _cache(root, val_fraction=0.0)


def test_split_is_deterministic_disjoint_and_by_source(tmp_path: Path) -> None:
    root = _standard_dataset(tmp_path, num_sources=10)
    first = _cache(root)
    second = _cache(root)
    val_indices = [s.source_index for s in first.val_sources]
    assert val_indices == [s.source_index for s in second.val_sources]
    assert len(first.val_sources) == 2  # round(0.2 * 10)
    train_indices = {s.source_index for s in first.train_sources}
    assert train_indices.isdisjoint(val_indices)
    assert train_indices | set(val_indices) == set(range(10))
    other_seed = _cache(root, split_seed=8)
    assert [s.source_index for s in other_seed.val_sources] != val_indices


def test_duplicate_clean_content_never_crosses_the_split(tmp_path: Path) -> None:
    """The leakage regression: image corpora ship the same picture under several
    filenames, and an index-level split then puts byte-identical content on both
    sides (fresh noise, SEEN scene), silently invalidating every held-out number.
    Duplicated sources must move together."""
    rng = np.random.default_rng(0)
    sources = {}
    for index in range(10):
        # Sources 0..4 are distinct; 5..9 duplicate the content of 0..4.
        clean = _gradient(offset=index % 5)
        frames = [
            np.clip(clean.astype(np.int16) + rng.integers(-40, 41, clean.shape), 0, 255)
            .astype(np.uint8)
            for _ in range(5)
        ]
        sources[index] = (clean, frames)
    root = _write_burst(tmp_path, sources)
    with pytest.warns(UserWarning, match="duplicated clean image"):
        cache = _cache(root, val_fraction=0.3)

    train_keys = {content_key(s.clean) for s in cache.train_sources}
    val_keys = {content_key(s.clean) for s in cache.val_sources}
    assert train_keys and val_keys
    assert not (train_keys & val_keys), "validation content also appears in training"
    # Each duplicate pair (i, i+5) landed wholly on one side.
    val_indices = {s.source_index for s in cache.val_sources}
    for index in range(5):
        assert (index in val_indices) == (index + 5 in val_indices)
    assert len(cache.duplicate_groups) == 5


def test_locked_test_split_is_disjoint_and_survives_a_val_fraction_change(
    tmp_path: Path,
) -> None:
    """A test split must be an anchor: changing val_fraction during development
    must not leak development scenes into it, or move scenes out of it."""
    root = _standard_dataset(tmp_path, num_sources=10)
    cache = _cache(root, val_fraction=0.2, test_fraction=0.2)
    test_indices = [s.source_index for s in cache.test_sources]
    val_indices = [s.source_index for s in cache.val_sources]
    train_indices = [s.source_index for s in cache.train_sources]
    assert len(test_indices) == 2 and len(val_indices) == 2
    assert set(test_indices).isdisjoint(val_indices)
    assert set(test_indices).isdisjoint(train_indices)
    assert set(train_indices) | set(val_indices) | set(test_indices) == set(range(10))

    widened = _cache(root, val_fraction=0.4, test_fraction=0.2)
    assert [s.source_index for s in widened.test_sources] == test_indices
    assert len(widened.val_sources) == 4


def test_split_without_a_test_fraction_is_unchanged(tmp_path: Path) -> None:
    """Backward compatibility: test_fraction=0 must reproduce the historical
    val split exactly, so results predating the test split stay reproducible."""
    root = _standard_dataset(tmp_path, num_sources=10)
    cache = _cache(root, val_fraction=0.2, test_fraction=0.0)
    assert not cache.test_sources
    assert [s.source_index for s in cache.val_sources] == [
        s.source_index for s in _cache(root, val_fraction=0.2).val_sources
    ]


def test_sources_for_split_rejects_an_unknown_name(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path))
    assert cache.sources_for_split("train") is cache.train_sources
    assert cache.sources_for_split("test") is cache.test_sources
    with pytest.raises(ValueError, match="split must be"):
        cache.sources_for_split("holdout")


def test_single_source_keeps_training_split_and_warns(tmp_path: Path) -> None:
    root = _standard_dataset(tmp_path, num_sources=1)
    with pytest.warns(UserWarning, match="only one usable content group"):
        cache = _cache(root)
    assert len(cache.train_sources) == 1
    assert not cache.val_sources


def test_rgb_frames_loaded_as_grayscale_warn_about_channel_averaging(tmp_path: Path) -> None:
    first = np.stack([_gradient(offset=0)] * 3, axis=-1)
    second = np.stack([_gradient(offset=1)] * 3, axis=-1)
    root = _write_burst(tmp_path, {0: (first, [first] * 5), 1: (second, [second] * 5)})
    with pytest.warns(UserWarning, match="partially denoises"):
        _cache(root, val_fraction=0.0)


def test_batches_have_the_documented_shapes_dtypes_and_ranges(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path))
    factory = _factory(cache)
    batch = factory.sample_batch()
    assert batch.x_t.shape == (6, 1, 16, 16)
    assert batch.eps.shape == (6, 1, 16, 16)
    assert batch.t.shape == (6,)
    assert batch.x_t.dtype == torch.float32
    assert batch.t.dtype == torch.float32
    assert batch.x_t.min() >= -1.0 and batch.x_t.max() <= 1.0
    assert batch.eps.min() >= -1.0 and batch.eps.max() <= 1.0
    assert all(1 <= int(t) <= 3 for t in batch.t)


def test_fresh_targets_are_excluded_from_the_averaged_subset(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path))
    factory = _factory(cache)
    for _ in range(40):
        _, info = factory.sample_batch(return_info=True)
        for sample in info:
            assert sample.target_replica not in sample.subset_replicas
            assert len(sample.subset_replicas) == 3 + 1 - sample.t  # m(t) = T+1-t


def test_included_mode_warns_and_puts_the_target_inside_the_subset(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path))
    with pytest.warns(UserWarning, match="documented-degenerate"):
        factory = _factory(cache, target_mode="included")
    _, info = factory.sample_batch(return_info=True)
    for sample in info:
        assert sample.target_replica in sample.subset_replicas


def test_antithetic_pairing_mirrors_the_noise_levels(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path))
    factory = _factory(cache, batch_size=8)
    batch = factory.sample_batch()
    t = batch.t.tolist()
    for index in range(4):
        assert t[index + 4] == 3 + 1 - t[index]


def test_subset_averaging_math_is_exact_for_constant_frames(tmp_path: Path) -> None:
    values_by_source = {0: [40, 80, 120, 200, 240], 1: [30, 70, 110, 190, 230]}
    sources = {
        index: (
            np.full((20, 24), 128 + index, dtype=np.uint8),
            [np.full((20, 24), value, dtype=np.uint8) for value in values],
        )
        for index, values in values_by_source.items()
    }
    root = _write_burst(tmp_path, sources)
    cache = _cache(root, val_fraction=0.0)
    factory = _factory(cache)
    batch, info = factory.sample_batch(return_info=True)
    for row in range(len(info)):
        values = values_by_source[info[row].source_index]
        subset_mean = np.mean([values[r] for r in info[row].subset_replicas])
        expected_x = subset_mean / 255.0 * 2.0 - 1.0
        expected_eps = values[info[row].target_replica] / 255.0 * 2.0 - 1.0
        assert batch.x_t[row].unique().numel() == 1
        assert batch.x_t[row, 0, 0, 0].item() == pytest.approx(expected_x, abs=1e-6)
        assert batch.eps[row, 0, 0, 0].item() == pytest.approx(expected_eps, abs=1e-6)


def test_crops_are_aligned_across_frames_and_match_reported_coordinates(tmp_path: Path) -> None:
    gradients = {index: _gradient(offset=index) for index in (0, 1)}
    root = _write_burst(
        tmp_path,
        {index: (image, [image] * 5) for index, image in gradients.items()},
    )
    cache = _cache(root, val_fraction=0.0)
    factory = _factory(cache)
    batch, info = factory.sample_batch(return_info=True)
    for row, sample in enumerate(info):
        top, left = sample.crop_yx
        gradient = gradients[sample.source_index]
        expected = gradient[top : top + 16, left : left + 16] / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(
            batch.x_t[row, 0].numpy(), expected.astype(np.float32), atol=1e-6
        )
        np.testing.assert_allclose(
            batch.eps[row, 0].numpy(), expected.astype(np.float32), atol=1e-6
        )


def test_sampling_is_seed_deterministic_and_state_roundtrips(tmp_path: Path) -> None:
    root = _standard_dataset(tmp_path)
    cache = _cache(root)
    first = _factory(cache)
    second = _factory(cache)
    a = first.sample_batch()
    b = second.sample_batch()
    assert torch.equal(a.x_t, b.x_t) and torch.equal(a.t, b.t) and torch.equal(a.eps, b.eps)

    state = first.state_dict()
    next_direct = first.sample_batch()
    third = _factory(cache, seed=999)
    third.load_state_dict(state)
    next_restored = third.sample_batch()
    assert torch.equal(next_direct.x_t, next_restored.x_t)
    assert torch.equal(next_direct.t, next_restored.t)
    assert torch.equal(next_direct.eps, next_restored.eps)


def test_val_batches_are_deterministic_and_carry_clean_targets(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path, num_sources=10))
    factory = _factory(cache)
    first = factory.val_batch(level=3, count=4)
    second = factory.val_batch(level=3, count=4)
    assert torch.equal(first.x_t, second.x_t)
    assert torch.equal(first.clean, second.clean)
    assert first.clean.shape == (4, 1, 16, 16)
    assert first.t.tolist() == [3.0] * 4
    with pytest.raises(ValueError, match=r"level must be in \[1, 3\]"):
        factory.val_batch(level=4, count=1)


def test_val_batch_without_val_sources_raises(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path), val_fraction=0.0)
    factory = _factory(cache)
    with pytest.raises(ValueError, match="no validation sources"):
        factory.val_batch(level=1, count=1)


def test_factory_validates_replica_count_and_crop_size(tmp_path: Path) -> None:
    cache = _cache(_standard_dataset(tmp_path, replicas=4))
    with pytest.raises(ValueError, match="needs at least 5"):
        _factory(cache, num_steps=4)
    with pytest.raises(ValueError, match="crops need at least 24x24"):
        _factory(cache, image_size=24)
