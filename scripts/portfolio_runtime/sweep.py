from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import validate_config
from .errors import ConfigError


_VARIANT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def load_sweep_variants(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read sweep matrix {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid sweep matrix YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("sweep matrix root must be a mapping")
    unknown = set(payload) - {"schema_version", "variants"}
    if unknown:
        raise ConfigError(f"unknown sweep matrix key(s): {sorted(unknown)}")
    if payload.get("schema_version") != 1:
        raise ConfigError("sweep matrix schema_version must equal 1")
    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ConfigError("sweep matrix variants must be a non-empty list")

    variants: list[dict[str, Any]] = []
    names: set[str] = set()
    for position, raw in enumerate(raw_variants):
        if not isinstance(raw, dict) or set(raw) != {"name", "overrides"}:
            raise ConfigError(
                f"sweep variant {position} must contain exactly name and overrides"
            )
        name = str(raw["name"])
        if not _VARIANT_NAME.fullmatch(name):
            raise ConfigError(f"invalid sweep variant name: {name}")
        if name in names:
            raise ConfigError(f"duplicate sweep variant name: {name}")
        overrides = raw["overrides"]
        if not isinstance(overrides, dict):
            raise ConfigError(f"sweep variant {name} overrides must be a mapping")
        if not all(isinstance(key, str) and key for key in overrides):
            raise ConfigError(f"sweep variant {name} contains an invalid override path")
        variants.append({"name": name, "overrides": dict(overrides)})
        names.add(name)
    return variants


def apply_dotted_overrides(
    base_config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(base_config)
    for path, value in overrides.items():
        parts = path.split(".")
        if any(not part for part in parts):
            raise ConfigError(f"invalid configuration override path: {path}")
        target: dict[str, Any] = result
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ConfigError(f"unknown configuration override path: {path}")
            target = target[part]
        leaf = parts[-1]
        if leaf not in target:
            raise ConfigError(f"unknown configuration override path: {path}")
        target[leaf] = deepcopy(value)
    return validate_config(result)


def flatten_sweep_metrics(
    name: str, metrics: dict[str, Any], *, status: str = "success"
) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": name, "status": status}
    for portfolio, prefix in (
        ("risk_optimized", "optimized"),
        ("equal_weight_signal", "equal_weight"),
        ("benchmark", "benchmark"),
    ):
        values = metrics.get("portfolios", {}).get(portfolio, {})
        for key in (
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "maximum_drawdown",
            "annualized_excess_return",
            "realized_tracking_error",
            "information_ratio",
            "maximum_active_drawdown",
            "total_one_way_turnover",
            "total_transaction_cost",
        ):
            row[f"{prefix}_{key}"] = values.get(key)
    return row
