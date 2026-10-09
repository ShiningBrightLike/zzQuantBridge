from pathlib import Path

import pytest

from quant.config import ConfigError, load_yaml_config


def test_load_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "strategy.yaml"
    path.write_text("name: baseline\nwindow: 20\n", encoding="utf-8")

    assert load_yaml_config(path) == {"name": "baseline", "window": 20}


def test_config_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_yaml_config(path)
