"""Self-rollout state generation for finetuning (method doc S5, Q&A Q6).

Runs the deterministic sampler on real seed frames and harvests its
intermediate pseudo-average states so training can use them as network
INPUTS -- closing the measured train/inference input-distribution gap
(the sampler's own states are statistically unlike the real frame-averages
seen in ordinary training at the same nominal level).

Deliberate asymmetry, load-bearing for validity: this module produces only
*inputs*. Targets never pass through it -- the training loop pairs every
harvested state with a REAL fresh frame drawn by the data layer
(:meth:`burst_diffusion.data.BatchFactory.rollout_pair_batch`), so the loss
is always graded by a real measurement and the conditional-mean argument of
Theorem 1 applies unchanged. Model outputs on the target side would be
self-distillation, free to drift toward the model's own artifacts.

The caller controls which weights generate the states (wrap the call in
:func:`burst_diffusion.ema.ema_parameters` to match inference, which samples
with the EMA weights); this function pins everything else to the inference
configuration: ``torch.no_grad``, eval mode (restored afterwards), the
full unit-step schedule, and no intermediate clamping.
"""

from __future__ import annotations

import torch

from .schedule import sample_step


def harvest_rollout_states(
    model: torch.nn.Module,
    seeds: torch.Tensor,
    stop_levels: torch.Tensor,
    *,
    num_steps: int,
) -> torch.Tensor:
    """Run the sampler from ``t = T`` on each seed; return the state at its stop level.

    ``seeds`` is ``[R, C, S, S]`` (real frames in model range, level ``T``);
    ``stop_levels`` is ``[R]`` with values in ``{1..T-1}`` -- the nominal level
    whose state is harvested, i.e. the state the inference sampler would feed
    the network at its level-``stop_level`` call, after ``T - stop_level``
    unit steps. The whole sub-batch shares one batched trajectory (each sample
    still has its own seed), stopping once the deepest requested level is
    reached. States are returned exactly as the sampler would feed them:
    un-clamped, detached, on the seeds' device.
    """
    if seeds.dim() != 4:
        raise ValueError(f"seeds must be [R, C, S, S], got shape {tuple(seeds.shape)}")
    if stop_levels.shape != (seeds.shape[0],):
        raise ValueError(
            f"stop_levels must be [{seeds.shape[0]}], got shape {tuple(stop_levels.shape)}"
        )
    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2 for rollouts, got {num_steps}")
    levels = stop_levels.tolist()
    if any(not 1 <= level <= num_steps - 1 for level in levels):
        raise ValueError(
            f"stop levels must be in [1, {num_steps - 1}] (T itself is the seed, "
            f"not a rollout state), got {sorted(set(levels))}"
        )

    was_training = model.training
    model.eval()
    try:
        x = seeds.to(dtype=torch.float32)
        harvested = torch.empty_like(x)
        stop_on_device = stop_levels.to(x.device)
        min_stop = min(levels)
        with torch.no_grad():
            for level in range(num_steps, min_stop, -1):
                t_tensor = torch.full((x.shape[0],), float(level), device=x.device)
                eps_hat = model(x, t_tensor)
                x = sample_step(x, eps_hat, level, level - 1, num_steps)
                arrived = stop_on_device == level - 1
                if bool(arrived.any()):
                    harvested[arrived] = x[arrived]
    finally:
        model.train(was_training)
    return harvested
