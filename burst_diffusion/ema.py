"""Exponential moving average of model parameters.

``EMAHelper`` is a standalone copy of ``models/ema.py`` from the legacy DDIM
code, minus the ``DataParallel`` unwrapping and the destructive ``ema_copy``
(burst_diffusion is single-device and evaluates through the non-destructive
``ema_parameters`` swap below, a pattern adopted from ddimctl).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


class EMAHelper:
    def __init__(self, mu: float = 0.999):
        if not 0.0 <= mu < 1.0:
            raise ValueError(f"mu must be in [0, 1), got {mu}")
        self.mu = mu
        self.shadow: dict[str, torch.Tensor] = {}

    def register(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = state_dict


@contextmanager
def ema_parameters(model: torch.nn.Module, ema: EMAHelper | None) -> Iterator[None]:
    """Temporarily evaluate ``model`` with the EMA weights, restoring afterwards.

    Swaps storage references instead of cloning another full model; the
    EMAHelper owns its shadow tensors and evaluation is read-only.
    """
    if ema is None:
        yield
        return
    named = dict(model.named_parameters())
    backup = {name: parameter.data for name, parameter in named.items() if name in ema.shadow}
    try:
        for name, value in ema.shadow.items():
            named[name].data = value.data
        yield
    finally:
        for name, value in backup.items():
            named[name].data = value
