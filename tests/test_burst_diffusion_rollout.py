"""Tests for the opt-in self-rollout finetuning stage (method doc S5, Q&A Q6).

The two invariants that make self-rollout finetuning valid are pinned here:

1. model outputs enter ONLY as network inputs -- every training target is a
   bit-exact crop of a real cached frame (proved with a poisoned model whose
   forward returns a sentinel constant);
2. ``rollout.fraction: 0`` (or omitting the block) is bit-identical to
   baseline training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from burst_diffusion.config import Config
from burst_diffusion.data import BatchFactory, BurstCache, RolloutPairInfo, SampleInfo
from burst_diffusion.ema import ema_parameters
from burst_diffusion.rollout import harvest_rollout_states
from burst_diffusion.schedule import frames_at, sample_step
from burst_diffusion.train import Trainer, load_checkpoint

NUM_STEPS = 4  # stop levels {1..3}; min_replicas = 5


def _write_burst(root: Path, *, num_sources: int = 3, replicas: int = 6) -> Path:
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(num_sources):
        clean = rng.integers(40, 216, size=(20, 24), dtype=np.uint8)
        Image.fromarray(clean).save(burst / "clean" / f"{source_index:05d}.png")
        for replica_index in range(replicas):
            noisy = np.clip(
                clean.astype(np.int16) + rng.integers(-30, 31, clean.shape), 0, 255
            ).astype(np.uint8)
            name = f"noisy/{source_index:05d}_{replica_index:05d}.png"
            Image.fromarray(noisy).save(burst / name)
            rows.append(
                json.dumps(
                    {
                        "source_index": source_index,
                        "replica_index": replica_index,
                        "clean_path": f"clean/{source_index:05d}.png",
                        "noisy_path": name,
                    }
                )
            )
    (burst / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _cache(root: Path, **overrides) -> BurstCache:
    kwargs = dict(channels=1, min_replicas=5, min_size=16, val_fraction=0.34, split_seed=7)
    kwargs.update(overrides)
    return BurstCache(root, **kwargs)


def _factory(cache: BurstCache, **overrides) -> BatchFactory:
    kwargs = dict(num_steps=NUM_STEPS, image_size=16, batch_size=4, antithetic=True, seed=11)
    kwargs.update(overrides)
    return BatchFactory(cache, **kwargs)


def _config(
    dataset_dir: Path, run_dir: Path, *, rollout: dict | None, **training_overrides
) -> Config:
    training: dict = {
        "run_dir": str(run_dir),
        "batch_size": 2,
        "max_steps": 3,
        "log_every": 1,
        "val_every": 2,
        "val_images": 2,
        "checkpoint_every": 2,
        "device": "cpu",
        "seed": 5,
    }
    training.update(training_overrides)
    if rollout is not None:
        training["rollout"] = rollout
    return Config.model_validate(
        {
            "data": {
                "dataset_dir": str(dataset_dir),
                "image_size": 16,
                "channels": 1,
                "val_fraction": 0.34,
            },
            "schedule": {"num_steps": NUM_STEPS},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": training,
            "sampling": {},
        }
    )


def _crop_to_model_range(frame: np.ndarray, top: int, left: int, size: int) -> torch.Tensor:
    crop = frame[top : top + size, left : left + size].astype(np.float32)
    return torch.from_numpy(crop / 255.0 * 2.0 - 1.0)[None]


class _AffineOfInput(torch.nn.Module):
    """Deterministic stand-in network: ``eps_hat = 0.5*x + 0.05*t`` (elementwise,
    so batched and per-sample trajectories agree bit-for-bit)."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * x + 0.05 * t.view(-1, 1, 1, 1)


# --- config validation ------------------------------------------------------


def test_rollout_config_defaults_and_activity_flag(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "run", rollout={})
    assert config.training.rollout is not None
    assert config.training.rollout.fraction == pytest.approx(0.5)
    assert config.training.rollout.use_ema is True
    assert config.rollout_active
    assert not _config(tmp_path, tmp_path / "run", rollout=None).rollout_active
    assert not _config(tmp_path, tmp_path / "run", rollout={"fraction": 0.0}).rollout_active


def test_rollout_config_rejects_unknown_keys_and_bad_fractions(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="surprise"):
        _config(tmp_path, tmp_path / "run", rollout={"surprise": 1})
    with pytest.raises(Exception, match="fraction"):
        _config(tmp_path, tmp_path / "run", rollout={"fraction": 1.5})


def test_rollout_requires_at_least_two_schedule_levels(tmp_path: Path) -> None:
    raw = _config(tmp_path, tmp_path / "run", rollout={"fraction": 0.5}).model_dump(mode="json")
    raw["schedule"]["num_steps"] = 1
    with pytest.raises(Exception, match="num_steps >= 2"):
        Config.model_validate(raw)
    raw["training"]["rollout"]["fraction"] = 0.0  # inert block: allowed
    Config.model_validate(raw)


def test_rollout_use_ema_requires_training_ema(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="requires training.ema"):
        _config(tmp_path, tmp_path / "run", rollout={"use_ema": True}, ema=False)
    _config(tmp_path, tmp_path / "run", rollout={"use_ema": False}, ema=False)
    _config(tmp_path, tmp_path / "run", rollout={"fraction": 0.0}, ema=False)


# --- data layer: rollout pairs ----------------------------------------------


def test_rollout_pairs_have_documented_shapes_dtypes_ranges_and_levels(tmp_path: Path) -> None:
    factory = _factory(_cache(_write_burst(tmp_path / "data")))
    batch = factory.rollout_pair_batch(count=6)
    assert batch.seed.shape == (6, 1, 16, 16)
    assert batch.target.shape == (6, 1, 16, 16)
    assert batch.stop_level.shape == (6,)
    assert batch.seed.dtype == torch.float32
    assert batch.target.dtype == torch.float32
    assert batch.stop_level.dtype == torch.int64
    assert batch.seed.min() >= -1.0 and batch.seed.max() <= 1.0
    assert batch.target.min() >= -1.0 and batch.target.max() <= 1.0
    assert all(1 <= int(level) <= NUM_STEPS - 1 for level in batch.stop_level)


def test_rollout_targets_are_real_frames_from_a_different_replica(tmp_path: Path) -> None:
    cache = _cache(_write_burst(tmp_path / "data"))
    factory = _factory(cache)
    by_index = {source.source_index: source for source in cache.train_sources}
    for _ in range(20):
        batch, info = factory.rollout_pair_batch(count=4, return_info=True)
        for row, sample in enumerate(info):
            assert sample.target_replica != sample.seed_replica
            source = by_index[sample.source_index]
            top, left = sample.crop_yx
            expected_seed = _crop_to_model_range(source.frames[sample.seed_replica], top, left, 16)
            expected_target = _crop_to_model_range(
                source.frames[sample.target_replica], top, left, 16
            )
            assert torch.equal(batch.seed[row], expected_seed)
            assert torch.equal(batch.target[row], expected_target)


def test_rollout_stop_levels_are_antithetic_within_the_reachable_range(tmp_path: Path) -> None:
    factory = _factory(_cache(_write_burst(tmp_path / "data")))
    batch = factory.rollout_pair_batch(count=8)
    levels = batch.stop_level.tolist()
    for index in range(4):
        assert levels[index] + levels[index + 4] == NUM_STEPS


def test_rollout_pair_batch_rejects_t1_schedules_and_zero_counts(tmp_path: Path) -> None:
    cache = _cache(_write_burst(tmp_path / "data"), min_replicas=2)
    single_level = _factory(cache, num_steps=1)
    with pytest.raises(ValueError, match="num_steps >= 2"):
        single_level.rollout_pair_batch(count=2)
    factory = _factory(_cache(_write_burst(tmp_path / "data2")))
    with pytest.raises(ValueError, match="count"):
        factory.rollout_pair_batch(count=0)


def test_rollout_draws_are_seed_deterministic_and_state_roundtrips(tmp_path: Path) -> None:
    root = _write_burst(tmp_path / "data")
    first = _factory(_cache(root))
    second = _factory(_cache(root))
    batch_a = first.rollout_pair_batch(count=4)
    batch_b = second.rollout_pair_batch(count=4)
    assert torch.equal(batch_a.seed, batch_b.seed)
    assert torch.equal(batch_a.target, batch_b.target)
    assert torch.equal(batch_a.stop_level, batch_b.stop_level)

    state = first.state_dict()
    before = first.rollout_pair_batch(count=4)
    first.load_state_dict(state)
    after = first.rollout_pair_batch(count=4)
    assert torch.equal(before.seed, after.seed)
    assert torch.equal(before.target, after.target)
    assert torch.equal(before.stop_level, after.stop_level)


def test_sample_batch_count_override_returns_that_many_samples(tmp_path: Path) -> None:
    factory = _factory(_cache(_write_burst(tmp_path / "data")))
    batch = factory.sample_batch(count=3)
    assert batch.x_t.shape[0] == 3
    assert factory.sample_batch().x_t.shape[0] == 4  # default unchanged
    with pytest.raises(ValueError, match="count"):
        factory.sample_batch(count=0)


# --- rollout state harvesting -----------------------------------------------


def test_harvested_states_match_a_manual_sampler_trajectory() -> None:
    model = _AffineOfInput()
    generator = torch.Generator().manual_seed(3)
    seeds = torch.rand((4, 1, 8, 8), generator=generator) * 2.0 - 1.0
    stop_levels = torch.tensor([1, 3, 2, 1])
    states = harvest_rollout_states(model, seeds, stop_levels, num_steps=NUM_STEPS)
    for row, stop in enumerate(stop_levels.tolist()):
        x = seeds[row : row + 1]
        for level in range(NUM_STEPS, stop, -1):
            eps_hat = model(x, torch.tensor([float(level)]))
            x = sample_step(x, eps_hat, level, level - 1, NUM_STEPS)
        assert torch.equal(states[row], x[0])


def test_harvest_is_gradient_free_and_restores_training_mode() -> None:
    model = _AffineOfInput()
    model.train()
    seeds = torch.zeros((2, 1, 8, 8), requires_grad=True)
    states = harvest_rollout_states(model, seeds, torch.tensor([1, 2]), num_steps=NUM_STEPS)
    assert not states.requires_grad
    assert model.training  # eval during the rollout, restored afterwards


def test_harvest_validates_shapes_levels_and_num_steps() -> None:
    model = _AffineOfInput()
    seeds = torch.zeros((2, 1, 8, 8))
    with pytest.raises(ValueError, match="seeds must be"):
        harvest_rollout_states(model, seeds[0], torch.tensor([1]), num_steps=NUM_STEPS)
    with pytest.raises(ValueError, match="stop_levels must be"):
        harvest_rollout_states(model, seeds, torch.tensor([1]), num_steps=NUM_STEPS)
    with pytest.raises(ValueError, match="stop levels must be in"):
        harvest_rollout_states(model, seeds, torch.tensor([0, 1]), num_steps=NUM_STEPS)
    with pytest.raises(ValueError, match="stop levels must be in"):
        harvest_rollout_states(model, seeds, torch.tensor([NUM_STEPS, 1]), num_steps=NUM_STEPS)
    with pytest.raises(ValueError, match="num_steps must be >= 2"):
        harvest_rollout_states(model, seeds, torch.tensor([1, 1]), num_steps=1)


# --- trainer integration -----------------------------------------------------


def test_poisoned_model_outputs_never_reach_the_target_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE validity invariant: even with a model that emits a sentinel constant,
    every assembled target is a bit-exact crop of a real cached frame, while
    the sentinel demonstrably flows into the rollout INPUT states."""
    dataset = _write_burst(tmp_path / "data")
    config = _config(dataset, tmp_path / "run", rollout={"fraction": 0.5}, batch_size=4)
    trainer = Trainer(config)
    poison = 0.777
    monkeypatch.setattr(
        trainer.model, "forward", lambda x, t: torch.full_like(x, poison)
    )
    by_index = {source.source_index: source for source in trainer.cache.train_sources}
    for _ in range(5):
        batch, rollout_count, info = trainer._assemble_batch(return_info=True)
        assert rollout_count == 2
        assert len(info) == 4
        for row, sample in enumerate(info):
            if row < len(info) - rollout_count:
                assert isinstance(sample, SampleInfo)
            else:
                assert isinstance(sample, RolloutPairInfo)
            top, left = sample.crop_yx
            source = by_index[sample.source_index]
            expected_target = _crop_to_model_range(
                source.frames[sample.target_replica], top, left, 16
            )
            assert torch.equal(batch.eps[row], expected_target)
        for row, sample in zip(range(len(info) - rollout_count, len(info)), info[-rollout_count:]):
            # With a constant prediction the composition proposition gives the
            # state in closed form: (seed + (m(stop)-1)*poison) / m(stop).
            m_stop = frames_at(sample.stop_level, NUM_STEPS)
            source = by_index[sample.source_index]
            top, left = sample.crop_yx
            seed = _crop_to_model_range(source.frames[sample.seed_replica], top, left, 16)
            expected_state = (seed + (m_stop - 1) * poison) / m_stop
            torch.testing.assert_close(batch.x_t[row], expected_state, atol=1e-6, rtol=0.0)
            assert float(batch.t[row]) == float(sample.stop_level)


def test_fraction_zero_is_bit_identical_to_baseline_training(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    baseline = Trainer(_config(dataset, tmp_path / "run_a", rollout=None))
    baseline.run()
    inert = Trainer(_config(dataset, tmp_path / "run_b", rollout={"fraction": 0.0}))
    inert.run()
    assert baseline.step == inert.step == 3
    for name, value in baseline.model.state_dict().items():
        assert torch.equal(value, inert.model.state_dict()[name]), name
    for name, value in baseline.ema.state_dict().items():
        assert torch.equal(value, inert.ema.state_dict()[name]), name
    next_a = baseline.factory.sample_batch()
    next_b = inert.factory.sample_batch()
    assert torch.equal(next_a.x_t, next_b.x_t)
    assert torch.equal(next_a.eps, next_b.eps)


def test_rollout_training_runs_validates_and_checkpoints(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    run_dir = tmp_path / "run"
    trainer = Trainer(_config(dataset, run_dir, rollout={"fraction": 0.5}))
    assert trainer._rollout_count == 1 and trainer._real_count == 1
    checkpoint = trainer.run()
    assert trainer.step == 3
    assert checkpoint.is_file()
    payload = load_checkpoint(checkpoint)
    restored = Config.model_validate(payload["config"])
    assert restored.training.rollout is not None
    assert restored.training.rollout.fraction == pytest.approx(0.5)
    assert list((run_dir / "tb").glob("events.out.tfevents.*"))


def test_resume_with_rollout_reproduces_an_uninterrupted_run_exactly(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    rollout = {"fraction": 0.5}
    full = Trainer(_config(dataset, tmp_path / "run_full", rollout=rollout, max_steps=4))
    full.run()

    partial = Trainer(_config(dataset, tmp_path / "run_split", rollout=rollout, max_steps=2))
    checkpoint = partial.run()
    with pytest.warns(UserWarning, match="different config"):
        resumed = Trainer(
            _config(dataset, tmp_path / "run_split", rollout=rollout, max_steps=4),
            resume_from=checkpoint,
        )
    resumed.run()
    assert resumed.step == 4
    for name, value in full.model.state_dict().items():
        assert torch.allclose(value, resumed.model.state_dict()[name], atol=1e-6), name
    for name, value in full.ema.state_dict().items():
        assert torch.allclose(value, resumed.ema.state_dict()[name], atol=1e-6), name


def test_rollout_states_are_generated_with_the_ema_weights(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    trainer = Trainer(_config(dataset, tmp_path / "run", rollout={"fraction": 0.5}))
    # Make live and EMA weights genuinely different (at init the shadow is a copy).
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(0.05)
    state = trainer.factory.state_dict()
    batch, rollout_count = trainer._assemble_batch()
    assert rollout_count == 1

    trainer.factory.load_state_dict(state)
    trainer.factory.sample_batch(count=trainer._real_count)  # consume the real draw
    pair = trainer.factory.rollout_pair_batch(count=1)
    with ema_parameters(trainer.model, trainer.ema):
        with_ema = harvest_rollout_states(
            trainer.model, pair.seed, pair.stop_level, num_steps=NUM_STEPS
        )
    with_live = harvest_rollout_states(
        trainer.model, pair.seed, pair.stop_level, num_steps=NUM_STEPS
    )
    assert torch.equal(batch.x_t[-1], with_ema[0])
    assert not torch.equal(with_ema, with_live)


def test_fraction_rounding_to_zero_warns_and_trains_as_baseline(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    with pytest.warns(UserWarning, match="zero rollout samples"):
        trainer = Trainer(
            _config(dataset, tmp_path / "run", rollout={"fraction": 0.05}, batch_size=2)
        )
    assert trainer._rollout_count == 0
    trainer.run()
    assert trainer.step == 3


def test_fraction_one_warns_about_dropping_real_inputs(tmp_path: Path) -> None:
    dataset = _write_burst(tmp_path / "data")
    with pytest.warns(UserWarning, match="original competency"):
        trainer = Trainer(
            _config(dataset, tmp_path / "run", rollout={"fraction": 1.0}, batch_size=2)
        )
    assert trainer._rollout_count == 2 and trainer._real_count == 0
    trainer.run()
    assert trainer.step == 3
