from __future__ import annotations

import numpy as np
import pytest

from burst_diffusion.schedule import frames_at, min_replicas, sample_step, sampling_schedule


def test_frames_at_is_a_bijection_onto_frame_counts() -> None:
    num_steps = 15
    counts = [frames_at(t, num_steps) for t in range(1, num_steps + 1)]
    assert sorted(counts) == list(range(1, num_steps + 1))
    assert frames_at(num_steps, num_steps) == 1
    assert frames_at(1, num_steps) == num_steps
    assert frames_at(0, num_steps) == num_steps + 1


def test_frames_at_rejects_out_of_range_and_bad_types() -> None:
    with pytest.raises(ValueError, match=r"t must be in \[0, 5\]"):
        frames_at(-1, 5)
    with pytest.raises(ValueError, match=r"t must be in \[0, 5\]"):
        frames_at(6, 5)
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        frames_at(0, 0)
    with pytest.raises(TypeError, match="t must be an int"):
        frames_at(1.0, 5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_steps must be an int"):
        frames_at(1, 5.0)  # type: ignore[arg-type]


def test_min_replicas_is_one_more_than_the_step_count() -> None:
    assert min_replicas(1) == 2
    assert min_replicas(15) == 16
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        min_replicas(0)


def test_unit_sample_step_matches_the_running_average_identity() -> None:
    rng = np.random.default_rng(0)
    num_steps = 7
    x = rng.normal(size=(3, 5)).astype(np.float64)
    eps = rng.normal(size=(3, 5)).astype(np.float64)
    for t in range(num_steps, 0, -1):
        m = frames_at(t, num_steps)
        expected = x + (eps - x) / (m + 1)
        actual = sample_step(x, eps, t, t - 1, num_steps)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_sample_step_composes_exactly_for_a_shared_prediction() -> None:
    rng = np.random.default_rng(1)
    num_steps = 12
    x = rng.normal(size=(4, 4))
    eps = rng.normal(size=(4, 4))
    paths = [
        (12, 7, 0),
        (12, 11, 5, 0),
        (12, 8, 4, 2, 1, 0),
        (9, 3, 0),
    ]
    for path in paths:
        direct = sample_step(x, eps, path[0], path[-1], num_steps)
        chained = x
        for t, t_next in zip(path, path[1:]):
            chained = sample_step(chained, eps, t, t_next, num_steps)
        np.testing.assert_allclose(chained, direct, rtol=0, atol=1e-12)


def test_full_rollout_with_real_frames_reproduces_the_burst_mean() -> None:
    # Spec-faithfulness: if the prediction at each step is the actual next
    # fresh frame, the T -> 0 rollout must equal the plain mean of all T+1
    # frames -- the cumulative-average recursion the method is built on.
    rng = np.random.default_rng(2)
    num_steps = 9
    frames = rng.normal(size=(num_steps + 1, 6, 6))
    x = frames[0]
    for t in range(num_steps, 0, -1):
        next_frame = frames[frames_at(t, num_steps)]
        x = sample_step(x, next_frame, t, t - 1, num_steps)
    np.testing.assert_allclose(x, frames.mean(axis=0), rtol=0, atol=1e-12)


def test_sample_step_rejects_non_decreasing_or_out_of_range_levels() -> None:
    for t, t_next in [(3, 3), (3, 4), (0, -1), (6, 2), (-1, -2)]:
        with pytest.raises(ValueError, match="expected 0 <= t_next < t <= 5"):
            sample_step(0.0, 0.0, t, t_next, 5)


def test_sampling_schedule_full_and_accelerated_forms() -> None:
    assert sampling_schedule(5, None) == [5, 4, 3, 2, 1]
    assert sampling_schedule(5, 5) == [5, 4, 3, 2, 1]
    assert sampling_schedule(5, 99) == [5, 4, 3, 2, 1]
    assert sampling_schedule(15, 2) == [15, 1]
    assert sampling_schedule(15, 1) == [15]
    for num_sample_steps in (2, 3, 4, 7, 10):
        schedule = sampling_schedule(15, num_sample_steps)
        assert schedule[0] == 15
        assert schedule[-1] == 1
        assert all(a > b for a, b in zip(schedule, schedule[1:]))
        assert all(1 <= level <= 15 for level in schedule)
        assert len(schedule) == num_sample_steps


def test_sampling_schedule_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="num_sample_steps must be >= 1"):
        sampling_schedule(5, 0)
    with pytest.raises(TypeError, match="num_sample_steps must be an int"):
        sampling_schedule(5, 2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        sampling_schedule(0, None)
