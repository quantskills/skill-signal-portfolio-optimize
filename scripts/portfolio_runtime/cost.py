from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .errors import ConfigError


def resolve_linear_cost_bps(
    config: dict[str, Any], override_bps: float | None
) -> tuple[float, dict[str, Any]]:
    configured = float(config["cost_model"]["linear_cost_bps"])
    if override_bps is None:
        return configured, {
            "source": "config",
            "configured_linear_cost_bps": configured,
            "override_linear_cost_bps": None,
            "effective_linear_cost_bps": configured,
        }
    value = float(override_bps)
    if not np.isfinite(value) or value < 0:
        raise ConfigError("transaction cost override must be finite and non-negative")
    return value, {
        "source": "cli_override",
        "configured_linear_cost_bps": configured,
        "override_linear_cost_bps": value,
        "effective_linear_cost_bps": value,
    }


def one_way_turnover(weights: pd.Series, current: pd.Series | None) -> float | None:
    if current is None:
        return None
    aligned = current.reindex(weights.index, fill_value=0.0)
    return float(0.5 * (weights - aligned).abs().sum())


def linear_transaction_cost(turnover: float | None, linear_cost_bps: float) -> float | None:
    if turnover is None:
        return None
    return float(turnover * linear_cost_bps / 10000.0)
