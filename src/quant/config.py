"""Configuration loading and validation for the command line workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or is invalid."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and fail with an actionable error message.

    PyYAML is deliberately kept at the application boundary. The rest of the
    package receives an ordinary mapping and does not depend on a YAML object.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in bare installs
        raise ConfigError("PyYAML is required to read YAML configuration files") from exc

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level configuration must be a mapping: {config_path}")
    return value
