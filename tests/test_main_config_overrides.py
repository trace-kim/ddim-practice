import argparse

import pytest

from main import apply_config_overrides


def legacy_config() -> dict:
    return {
        "data": {
            "data_path": "D:/data/original",
            "image_size": 32,
            "num_workers": 0,
            "random_flip": False,
        },
        "model": {"ch": 64, "ch_mult": [1, 2, 2, 2], "dropout": 0.0},
        "diffusion": {"num_diffusion_timesteps": 100},
        "training": {"batch_size": 7, "n_iters": 1000},
        "sampling": {"batch_size": 8},
        "optim": {"lr": 0.0002},
    }


def override_args(**values: object) -> argparse.Namespace:
    defaults = {
        "image_size": None,
        "batch_size": None,
        "learning_rate": None,
        "max_steps": None,
        "data_path": None,
        "num_workers": None,
        "diffusion_steps": None,
        "model_ch": None,
        "config_overrides": [],
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_convenience_flags_override_legacy_config_values() -> None:
    config = legacy_config()
    applied = apply_config_overrides(
        config,
        override_args(
            image_size=64,
            batch_size=4,
            learning_rate=0.0001,
            max_steps=20000,
            data_path="E:/datasets/sem",
            num_workers=2,
            diffusion_steps=200,
            model_ch=96,
        ),
    )

    assert config["data"] == {
        "data_path": "E:/datasets/sem",
        "image_size": 64,
        "num_workers": 2,
        "random_flip": False,
    }
    assert config["training"] == {"batch_size": 4, "n_iters": 20000}
    assert config["optim"]["lr"] == 0.0001
    assert config["diffusion"]["num_diffusion_timesteps"] == 200
    assert config["model"]["ch"] == 96
    assert "data.image_size=64" in applied
    assert "training.n_iters=20000" in applied


def test_repeated_set_overrides_parse_yaml_scalars_and_lists() -> None:
    config = legacy_config()
    apply_config_overrides(
        config,
        override_args(
            config_overrides=[
                "data.random_flip=true",
                "model.dropout=0.1",
                "model.ch_mult=[1, 2, 4]",
                "sampling.batch_size=16",
            ]
        ),
    )

    assert config["data"]["random_flip"] is True
    assert config["model"]["dropout"] == 0.1
    assert config["model"]["ch_mult"] == [1, 2, 4]
    assert config["sampling"]["batch_size"] == 16


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ("data.imag_size=64", "unknown config key"),
        ("data.image_size=large", "expects an integer"),
        ("data=anything", "is a section"),
        ("missing-separator", "SECTION.KEY=VALUE"),
    ),
)
def test_set_rejects_unknown_keys_and_incompatible_values(
    override: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_config_overrides(
            legacy_config(), override_args(config_overrides=[override])
        )
