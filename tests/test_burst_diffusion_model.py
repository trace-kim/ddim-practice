from __future__ import annotations

import pytest
import torch

from burst_diffusion.config import Config
from burst_diffusion.ema import EMAHelper, ema_parameters
from burst_diffusion.unet import UNet, build_unet


def _tiny_unet(**overrides) -> UNet:
    kwargs = dict(
        in_channels=1,
        out_ch=1,
        ch=8,
        ch_mult=[1, 2],
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        resamp_with_conv=True,
        resolution=16,
    )
    kwargs.update(overrides)
    return UNet(**kwargs)


def test_forward_preserves_shape_and_produces_out_ch_channels() -> None:
    net = _tiny_unet(out_ch=3)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([1.0, 3.0])
    with torch.no_grad():
        out = net(x, t)
    assert out.shape == (2, 3, 16, 16)
    assert torch.isfinite(out).all()


def test_small_channel_count_works_via_adaptive_num_groups() -> None:
    # The legacy model hardcodes GroupNorm(32) and would crash for ch=8; the
    # adapted copy defaults num_groups to min(32, ch).
    net = _tiny_unet(ch=8)
    with torch.no_grad():
        out = net(torch.randn(1, 1, 16, 16), torch.tensor([2.0]))
    assert out.shape == (1, 1, 16, 16)


def test_non_integer_float_timesteps_are_accepted() -> None:
    net = _tiny_unet()
    with torch.no_grad():
        out = net(torch.randn(1, 1, 16, 16), torch.tensor([2.5]))
    assert torch.isfinite(out).all()


def test_wrong_input_size_raises_value_error() -> None:
    net = _tiny_unet()
    with pytest.raises(ValueError, match=r"expected square input \[B, C, 16, 16\]"):
        net(torch.randn(1, 1, 16, 12), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="expected square input"):
        net(torch.randn(1, 1, 8, 8), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="timesteps must be 1-D"):
        net(torch.randn(1, 1, 16, 16), torch.tensor([[1.0]]))


def test_invalid_constructor_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="num_groups .* must divide ch"):
        _tiny_unet(num_groups=5)
    with pytest.raises(ValueError, match="ch must be >= 4"):
        _tiny_unet(ch=2, num_groups=2)


def test_build_unet_uses_config_defaults() -> None:
    config = Config.model_validate(
        {
            "data": {"dataset_dir": "data/x", "image_size": 16, "channels": 3},
            "schedule": {"num_steps": 3},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": "runs/x"},
            "sampling": {},
        }
    )
    net = build_unet(config)
    with torch.no_grad():
        out = net(torch.randn(1, 3, 16, 16), torch.tensor([1.0]))
    assert out.shape == (1, 3, 16, 16)  # out_ch defaulted to data.channels


def test_ema_update_follows_the_exponential_formula() -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    ema = EMAHelper(mu=0.9)
    ema.register(module)
    with torch.no_grad():
        module.weight.fill_(2.0)
    ema.update(module)
    # shadow = 0.1 * 2.0 + 0.9 * 1.0
    assert ema.shadow["weight"].item() == pytest.approx(1.1)
    with pytest.raises(ValueError, match=r"mu must be in \[0, 1\)"):
        EMAHelper(mu=1.0)


def test_ema_parameters_swaps_in_shadow_and_restores_exactly() -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    ema = EMAHelper(mu=0.5)
    ema.register(module)
    live = module.weight.data.clone()
    with torch.no_grad():
        module.weight.add_(1.0)
    updated = module.weight.data.clone()
    ema.update(module)

    with ema_parameters(module, ema):
        assert torch.equal(module.weight.data, ema.shadow["weight"].data)
        assert not torch.equal(module.weight.data, updated)
    assert torch.equal(module.weight.data, updated)
    assert not torch.equal(module.weight.data, live)


def test_ema_parameters_is_a_no_op_without_a_helper() -> None:
    module = torch.nn.Linear(1, 1)
    before = module.weight.data.clone()
    with ema_parameters(module, None):
        assert torch.equal(module.weight.data, before)
    assert torch.equal(module.weight.data, before)
