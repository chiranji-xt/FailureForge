"""Optional YAML run configuration for the CLI.

Command-line flags always override the file; the file overrides built-in
defaults. Unknown sections or keys are rejected so typos are not silently
ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for unreadable or invalid configuration files."""


#: section -> allowed keys
SCHEMA: dict[str, set[str]] = {
    "dataset": {"data", "split", "allow_missing_labels"},
    "detector": {"weights", "conf", "imgsz", "device"},
    "evaluation": {"iou_threshold"},
    "search": {
        "method", "budget", "seed", "metric", "transforms", "holdout_fraction",
        "partition_seed", "confirm_top", "confirm_seeds", "confirm_seed",
    },
    "output": {"dir"},
}


def load_config(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load and validate a config file (``None`` -> empty config)."""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config {p} must be a mapping of sections")
    for section, values in raw.items():
        if section not in SCHEMA:
            raise ConfigError(
                f"unknown config section '{section}'; allowed: {sorted(SCHEMA)}"
            )
        if values is None:
            raw[section] = values = {}
        if not isinstance(values, dict):
            raise ConfigError(f"config section '{section}' must be a mapping")
        unknown = set(values) - SCHEMA[section]
        if unknown:
            raise ConfigError(
                f"unknown key(s) in '{section}': {sorted(unknown)}; "
                f"allowed: {sorted(SCHEMA[section])}"
            )
    return raw


def pick(cli_value: Any, config: dict, section: str, key: str, default: Any = None) -> Any:
    """CLI value if given, else config file value, else ``default``."""
    if cli_value is not None:
        return cli_value
    value = config.get(section, {}).get(key)
    return default if value is None else value
