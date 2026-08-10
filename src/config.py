"""Configuration loading and project path helpers.

The checked-in config is JSON syntax saved as ``config.yaml``. JSON is valid
YAML, and this keeps the core pipeline dependency-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(ValueError):
    """Raised when the project configuration is invalid."""


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("Config not found: %s" % config_path) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "%s must use JSON-compatible YAML syntax: %s" % (config_path, exc)
        ) from exc

    for key in ("benchmark", "datasets"):
        if key not in config:
            raise ConfigError("Missing required config section: %s" % key)
    return config


def project_path(value: str, root: Optional[Path] = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root or PROJECT_ROOT) / path
