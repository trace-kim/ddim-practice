"""Frame-averaging schedule: the pure math behind burst diffusion.

Notation (documented in README.md): ``t`` runs over ``{1..T}`` in the DDPM
direction, ``t = T`` being the noisiest state. ``m(t) = T + 1 - t`` is the
number of burst frames averaged at level ``t``, so ``x_T`` is a single raw
frame and ``x_1`` averages ``T`` frames. ``m(0) = T + 1`` exists only as an
averaging weight for the final sampling step -- the network is never called
with ``t = 0``.

The update ``sample_step`` treats the network prediction as ``m(t_next)-m(t)``
additional fresh frames folded into the running average. It composes exactly:
chaining any decreasing path of steps with a shared prediction equals the
direct jump, so accelerated sampling only approximates the *changing*
prediction, never the update itself.

All functions use plain arithmetic so array-likes (numpy arrays, torch
tensors) pass through unchanged.
"""

from __future__ import annotations

from typing import TypeVar

ArrayT = TypeVar("ArrayT")


def _validate_num_steps(num_steps: int) -> None:
    if not isinstance(num_steps, int) or isinstance(num_steps, bool):
        raise TypeError(f"num_steps must be an int, got {type(num_steps).__name__}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")


def frames_at(t: int, num_steps: int) -> int:
    """Number of burst frames averaged at noise level ``t``: ``m(t) = T + 1 - t``.

    Valid for ``t`` in ``{0..T}``; ``t = 0`` is the post-final-step state whose
    frame count is used only as an averaging weight.
    """
    _validate_num_steps(num_steps)
    if not isinstance(t, int) or isinstance(t, bool):
        raise TypeError(f"t must be an int, got {type(t).__name__}")
    if not 0 <= t <= num_steps:
        raise ValueError(f"t must be in [0, {num_steps}], got {t}")
    return num_steps + 1 - t


def min_replicas(num_steps: int) -> int:
    """Minimum burst frames per source required for training: ``T + 1``.

    The cleanest training level (``t = 1``) averages ``T`` frames and the
    fresh target must come from outside that subset.
    """
    _validate_num_steps(num_steps)
    return num_steps + 1


def sample_step(x_t: ArrayT, eps_hat: ArrayT, t: int, t_next: int, num_steps: int) -> ArrayT:
    """One cumulative-average sampling update from level ``t`` to ``t_next``.

    ``x_{t_next} = (m(t) * x_t + (m(t_next) - m(t)) * eps_hat) / m(t_next)``

    The prediction stands in for the ``m(t_next) - m(t)`` fresh frames that a
    real burst average would have folded in. Requires ``0 <= t_next < t <= T``.
    """
    _validate_num_steps(num_steps)
    if not 0 <= t_next < t <= num_steps:
        raise ValueError(
            f"expected 0 <= t_next < t <= {num_steps}, got t={t}, t_next={t_next}"
        )
    m_now = frames_at(t, num_steps)
    m_next = frames_at(t_next, num_steps)
    return (m_now * x_t + (m_next - m_now) * eps_hat) / m_next


def sampling_schedule(num_steps: int, num_sample_steps: int | None = None) -> list[int]:
    """Strictly decreasing noise levels to visit, starting at ``T``, ending at 1.

    ``None`` (or ``>= T``) yields the full schedule ``[T, T-1, .., 1]``. A
    smaller count picks evenly spaced levels (endpoints always included),
    giving DDIM-style accelerated sampling. ``t = 0`` is never in the
    schedule -- it is reached implicitly as the final ``sample_step`` target.
    """
    _validate_num_steps(num_steps)
    if num_sample_steps is None or num_sample_steps >= num_steps:
        return list(range(num_steps, 0, -1))
    if not isinstance(num_sample_steps, int) or isinstance(num_sample_steps, bool):
        raise TypeError(
            f"num_sample_steps must be an int or None, got {type(num_sample_steps).__name__}"
        )
    if num_sample_steps < 1:
        raise ValueError(f"num_sample_steps must be >= 1, got {num_sample_steps}")
    if num_sample_steps == 1:
        return [num_steps]
    # Evenly spaced floats from T down to 1, rounded; dedupe preserves order.
    span = num_steps - 1
    raw = [
        num_steps - round(index * span / (num_sample_steps - 1))
        for index in range(num_sample_steps)
    ]
    schedule: list[int] = []
    for level in raw:
        if not schedule or level < schedule[-1]:
            schedule.append(level)
    return schedule
