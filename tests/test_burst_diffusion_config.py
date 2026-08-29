from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from burst_diffusion.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_raw() -> dict:
    return {
        "data": {"dataset_dir": "data/example", "image_size": 32},
        "schedule": {"num_steps": 3},
        "model": {"ch": 32, "ch_mult": [1, 2], "attn_resolutions": []},
        "training": {"run_dir": "runs/example"},
        "sampling": {},
    }


def test_load_config_round_trips_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_dir: data/example",
                "  image_size: 32",
                "schedule:",
                "  num_steps: 3",
                "model:",
                "  ch: 32",
                "  ch_mult: [1, 2]",
                "  attn_resolutions: []",
                "training:",
                "  run_dir: runs/example",
                "sampling: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.data.dataset_dir == Path("data/example")
    assert config.schedule.num_steps == 3
    assert config.schedule.target_mode == "fresh"
    assert config.training.lr == pytest.approx(2.0e-4)


def test_shipped_configs_are_valid() -> None:
    for name in (
        "smoke.yml",
        "bbbc038_p10.yml",
        "miic_p10.yml",
        "bbbc038_p10_rollout.yml",
        "miic_p10_rollout.yml",
    ):
        config = load_config(REPO_ROOT / "configs" / "burst_diffusion" / name)
        assert config.schedule.num_steps >= 1
    for name in ("bbbc038_p10_rollout.yml", "miic_p10_rollout.yml"):
        config = load_config(REPO_ROOT / "configs" / "burst_diffusion" / name)
        assert config.rollout_active
        assert config.training.rollout.fraction == pytest.approx(0.5)


def test_unknown_keys_are_rejected_in_every_section() -> None:
    raw = _base_raw()
    raw["data"]["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        Config.model_validate(raw)
    raw = _base_raw()
    raw["mystery_section"] = {}
    with pytest.raises(ValidationError, match="mystery_section"):
        Config.model_validate(raw)


def test_image_size_must_match_the_level_count() -> None:
    raw = _base_raw()
    raw["data"]["image_size"] = 100
    raw["model"]["ch_mult"] = [1, 2, 2, 2]
    with pytest.raises(ValidationError, match="divisible by 2"):
        Config.model_validate(raw)


def test_num_groups_must_divide_ch() -> None:
    raw = _base_raw()
    raw["model"]["num_groups"] = 5
    with pytest.raises(ValidationError, match="must divide model.ch"):
        Config.model_validate(raw)


def test_effective_num_groups_defaults_to_min_of_32_and_ch() -> None:
    raw = _base_raw()
    raw["model"]["ch"] = 8
    config = Config.model_validate(raw)
    assert config.model.effective_num_groups == 8
    raw["model"]["ch"] = 64
    config = Config.model_validate(raw)
    assert config.model.effective_num_groups == 32


def test_out_ch_defaults_to_data_channels() -> None:
    config = Config.model_validate(_base_raw())
    assert config.effective_out_ch == 1
    raw = _base_raw()
    raw["data"]["channels"] = 3
    assert Config.model_validate(raw).effective_out_ch == 3
    raw["model"]["out_ch"] = 1
    assert Config.model_validate(raw).effective_out_ch == 1


def test_unreachable_attention_resolution_is_rejected() -> None:
    raw = _base_raw()
    raw["model"]["attn_resolutions"] = [17]
    with pytest.raises(ValidationError, match="not one of the reachable"):
        Config.model_validate(raw)


def test_large_attention_resolution_warns() -> None:
    raw = _base_raw()
    raw["data"]["image_size"] = 32
    raw["model"]["attn_resolutions"] = [32]
    with pytest.warns(UserWarning, match="O\\(\\(H\\*W\\)\\^2\\)"):
        Config.model_validate(raw)


def test_invalid_scalars_are_rejected() -> None:
    for section, key, value, match in [
        ("schedule", "num_steps", 0, "num_steps"),
        ("schedule", "target_mode", "sideways", "target_mode"),
        ("data", "val_fraction", 1.0, "val_fraction"),
        ("data", "channels", 2, "channels"),
        ("training", "ema_rate", 1.0, "ema_rate"),
        ("training", "device", "tpu", "device"),
        ("sampling", "num_sample_steps", 0, "num_sample_steps"),
    ]:
        raw = _base_raw()
        raw[section][key] = value
        with pytest.raises(ValidationError, match=match):
            Config.model_validate(raw)


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config root must be a mapping"):
        load_config(config_path)
