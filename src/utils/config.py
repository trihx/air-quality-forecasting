"""Configuration loader — đọc YAML configs."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dictionary with config values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config path is outside project directory.
    """
    path = _validate_config_path(config_path)
    logger.debug(f"Loading config from {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        logger.warning(f"Empty config file: {path}")
        return {}

    logger.info(f"Config loaded: {path.name} ({len(config)} top-level keys)")
    return config


def load_model_config(model_name: str) -> dict[str, Any]:
    """Load model-specific config by name.

    Args:
        model_name: Model name (e.g., 'xgboost', 'lstm', 'random_forest').

    Returns:
        Dictionary with model config values.
    """
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "configs" / "model_configs" / f"{model_name}.yaml"
    return load_config(config_path)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override config into base config.

    Args:
        base: Base config dictionary.
        override: Override config dictionary (takes priority).

    Returns:
        Merged config dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_config_path(config_path: str | Path) -> Path:
    """Validate config path is within project and exists."""
    path = Path(config_path).resolve()
    project_root = Path(__file__).parent.parent.parent.resolve()

    if not str(path).startswith(str(project_root)):
        raise ValueError(f"Config path '{config_path}' is outside project directory")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Config must be YAML file, got: {path.suffix}")

    return path
