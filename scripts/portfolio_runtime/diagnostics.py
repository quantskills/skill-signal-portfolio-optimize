from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .errors import ConfigError


def expand_limits(
    specification: float | dict[str, float] | None,
    names: list[str],
    label: str,
) -> dict[str, float]:
    if specification is None:
        return {}
    if isinstance(specification, dict):
        missing = set(names) - set(specification)
        extra = set(specification) - set(names)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unknown {sorted(extra)}")
            raise ConfigError(f"{label} mapping does not match inputs: {"; ".join(parts)}")
        return {name: float(specification[name]) for name in names}
    return {name: float(specification) for name in names}


def resolve_industry_ranges(
    constraint_config: dict[str, Any], names: list[str]
) -> dict[str, dict[str, float]]:
    specification = constraint_config["industry_active_range"]
    if specification is not None:
        overrides = specification["overrides"]
        unknown = set(overrides) - set(names)
        if unknown:
            raise ConfigError(f"industry overrides contain unknown industry(s): {sorted(unknown)}")
        default = specification["default"]
        missing = set(names) - set(overrides)
        if missing and default is None:
            raise ConfigError(f"industry ranges missing industry(s): {sorted(missing)}")
        return {
            name: dict(overrides.get(name) or default)
            for name in names
        }
    legacy = expand_limits(
        constraint_config["sector_active_limit"], names, "sector_active_limit"
    )
    return {
        name: {"lower_active": -limit, "upper_active": limit}
        for name, limit in legacy.items()
    }


def resolve_style_ranges(
    constraint_config: dict[str, Any], names: list[str]
) -> dict[str, dict[str, float]]:
    specification = constraint_config["style_active_ranges"]
    if specification is not None:
        enabled = {
            name: values for name, values in specification.items()
            if values.get("enabled", False)
        }
        missing = set(enabled) - set(names)
        if missing:
            raise ConfigError(f"style exposures missing configured factor(s): {sorted(missing)}")
        return {
            name: {
                "lower_active": float(enabled[name]["lower_active"]),
                "upper_active": float(enabled[name]["upper_active"]),
            }
            for name in names if name in enabled
        }
    legacy = expand_limits(
        constraint_config["factor_active_limit"], names, "factor_active_limit"
    )
    return {
        name: {"lower_active": -limit, "upper_active": limit}
        for name, limit in legacy.items()
    }


def _range_detail(
    benchmark_exposure: float,
    portfolio_exposure: float,
    active_exposure: float,
    specification: dict[str, float] | None,
    tolerance: float,
) -> dict[str, float | bool | None]:
    if specification is None:
        return {
            "benchmark_exposure": benchmark_exposure,
            "portfolio_exposure": portfolio_exposure,
            "active_exposure": active_exposure,
            "lower_bound": None,
            "upper_bound": None,
            "lower_slack": None,
            "upper_slack": None,
            "binding": False,
            "passed": True,
        }
    lower = float(specification["lower_active"])
    upper = float(specification["upper_active"])
    lower_slack = active_exposure - lower
    upper_slack = upper - active_exposure
    return {
        "benchmark_exposure": benchmark_exposure,
        "portfolio_exposure": portfolio_exposure,
        "active_exposure": active_exposure,
        "lower_bound": lower,
        "upper_bound": upper,
        "lower_slack": lower_slack,
        "upper_slack": upper_slack,
        "binding": bool(lower_slack <= tolerance or upper_slack <= tolerance),
        "passed": bool(lower_slack >= -tolerance and upper_slack >= -tolerance),
    }


def portfolio_metrics(
    weights: pd.Series,
    expected_return: pd.Series,
    covariance: pd.DataFrame,
    benchmark: pd.Series,
) -> dict[str, float]:
    w = weights.to_numpy(dtype=float)
    mu = expected_return.to_numpy(dtype=float)
    b = benchmark.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    active = w - b
    absolute_variance = max(float(w @ sigma @ w), 0.0)
    active_variance = max(float(active @ sigma @ active), 0.0)
    return {
        "expected_return": float(w @ mu),
        "absolute_volatility": float(np.sqrt(absolute_variance)),
        "active_volatility": float(np.sqrt(active_variance)),
        "absolute_variance": absolute_variance,
        "active_variance": active_variance,
    }


def constraint_report(
    weights: pd.Series,
    benchmark: pd.Series,
    current: pd.Series | None,
    covariance: pd.DataFrame,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    tradable: pd.Series,
    constraint_config: dict[str, Any],
) -> dict[str, Any]:
    tolerance = float(constraint_config["constraint_tolerance"])
    weight_sum_tolerance = float(constraint_config["weight_sum_tolerance"])
    active = weights - benchmark
    bound_scope = tradable if current is not None else pd.Series(True, index=weights.index)
    controllable_weights = weights[bound_scope]
    controllable_active = active[bound_scope]
    maximum_controllable_weight = (
        float(controllable_weights.max()) if len(controllable_weights) else 0.0
    )
    maximum_controllable_active_weight = (
        float(controllable_active.abs().max()) if len(controllable_active) else 0.0
    )
    report: dict[str, Any] = {
        "weight_sum": float(weights.sum()),
        "weight_sum_error": float(abs(weights.sum() - 1.0)),
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "maximum_absolute_active_weight": float(active.abs().max()),
        "maximum_controllable_weight": maximum_controllable_weight,
        "maximum_controllable_absolute_active_weight": (
            maximum_controllable_active_weight
        ),
        "frozen_bound_exceptions": [],
        "one_way_turnover": None,
        "tracking_error": float(
            np.sqrt(max(float(active.to_numpy() @ covariance.to_numpy() @ active.to_numpy()), 0.0))
        ),
        "maximum_frozen_weight_deviation": None,
        "industry_exposures": {},
        "style_exposures": {},
        "sector_active_exposure": {},
        "factor_active_exposure": {},
        "binding_constraints": [],
        "violations": [],
    }

    if current is not None:
        report["one_way_turnover"] = float(0.5 * (weights - current).abs().sum())
        frozen = ~tradable
        if frozen.any():
            report["maximum_frozen_weight_deviation"] = float(
                (weights[frozen] - current[frozen]).abs().max()
            )
            max_weight = float(constraint_config["max_weight"])
            max_active_weight = float(constraint_config["max_active_weight"])
            for ticker in weights.index[frozen]:
                frozen_weight = float(current[ticker])
                benchmark_weight = float(benchmark[ticker])
                active_weight = frozen_weight - benchmark_weight
                if (
                    frozen_weight > max_weight + tolerance
                    or abs(active_weight) > max_active_weight + tolerance
                ):
                    report["frozen_bound_exceptions"].append(
                        {
                            "ticker": str(ticker),
                            "frozen_weight": frozen_weight,
                            "benchmark_weight": benchmark_weight,
                            "active_weight": active_weight,
                            "max_weight": max_weight,
                            "max_active_weight": max_active_weight,
                        }
                    )

    sector_ranges: dict[str, dict[str, float]] = {}
    if sectors is not None:
        names = sorted(sectors.unique().tolist())
        sector_ranges = resolve_industry_ranges(constraint_config, names)
        for name in names:
            coefficients = (sectors == name).to_numpy(dtype=float)
            benchmark_value = float(coefficients @ benchmark.to_numpy(dtype=float))
            portfolio_value = float(coefficients @ weights.to_numpy(dtype=float))
            active_value = portfolio_value - benchmark_value
            specification = sector_ranges.get(name)
            report["sector_active_exposure"][name] = active_value
            report["industry_exposures"][name] = _range_detail(
                benchmark_value, portfolio_value, active_value, specification, tolerance
            )

    factor_ranges: dict[str, dict[str, float]] = {}
    if exposures is not None:
        names = list(exposures.columns)
        factor_ranges = resolve_style_ranges(constraint_config, names)
        for name in names:
            coefficients = exposures[name].to_numpy(dtype=float)
            benchmark_value = float(coefficients @ benchmark.to_numpy(dtype=float))
            portfolio_value = float(coefficients @ weights.to_numpy(dtype=float))
            active_value = portfolio_value - benchmark_value
            specification = factor_ranges.get(name)
            report["factor_active_exposure"][name] = active_value
            report["style_exposures"][name] = _range_detail(
                benchmark_value, portfolio_value, active_value, specification, tolerance
            )

    def violation(name: str, excess: float) -> None:
        if excess > tolerance:
            report["violations"].append({"constraint": name, "excess": float(excess)})
        elif abs(excess) <= tolerance:
            report["binding_constraints"].append(name)

    violation("fully_invested", report["weight_sum_error"] - weight_sum_tolerance)
    violation("long_only", -report["minimum_weight"])
    violation(
        "max_weight",
        report["maximum_controllable_weight"]
        - float(constraint_config["max_weight"]),
    )
    violation(
        "max_active_weight",
        report["maximum_controllable_absolute_active_weight"]
        - float(constraint_config["max_active_weight"]),
    )
    if constraint_config["max_turnover"] is not None and report["one_way_turnover"] is not None:
        violation(
            "max_turnover",
            report["one_way_turnover"] - float(constraint_config["max_turnover"]),
        )
    if constraint_config["max_tracking_error"] is not None:
        violation(
            "max_tracking_error",
            report["tracking_error"] - float(constraint_config["max_tracking_error"]),
        )
    if report["maximum_frozen_weight_deviation"] is not None:
        violation("tradability_freeze", report["maximum_frozen_weight_deviation"])

    for name in sector_ranges:
        detail = report["industry_exposures"][name]
        violation(
            f"industry_active_range:{name}:lower", -float(detail["lower_slack"])
        )
        violation(
            f"industry_active_range:{name}:upper", -float(detail["upper_slack"])
        )
    for name in factor_ranges:
        detail = report["style_exposures"][name]
        violation(
            f"style_active_range:{name}:lower", -float(detail["lower_slack"])
        )
        violation(
            f"style_active_range:{name}:upper", -float(detail["upper_slack"])
        )

    report["passed"] = not report["violations"]
    return report
