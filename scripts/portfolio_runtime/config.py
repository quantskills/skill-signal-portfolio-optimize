from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "signal": {
        "type": "rank_score",
        "higher_is_better": True,
        "winsorize_mad": 5.0,
        "zscore": True,
        "rank_transform": "uniform",
        "rank_power": 1.0,
        "annualized_alpha_scale": 0.05,
        "missing_prediction_policy": "neutral",
    },
    "covariance": {
        "risk_form": "asset_covariance",
        "annualized": True,
        "periods_per_year": 252,
        "eigenvalue_floor": 1.0e-10,
        "symmetry_tolerance": 1.0e-10,
    },
    "optimizer": {
        "objective_mode": "mean_variance",
        "solver_backend": "scipy_slsqp",
        "risk_aversion": 5.0,
        "turnover_penalty": 0.001,
        "smoothing_epsilon": 1.0e-8,
        "max_iterations": 2000,
        "ftol": 1.0e-10,
        "minimum_signal_capture": 0.995,
        "stability_penalty": 1.0e-8,
    },
    "cost_model": {"linear_cost_bps": 0.0},
    "constraints": {
        "max_weight": 0.20,
        "max_active_weight": 0.15,
        "max_turnover": None,
        "max_tracking_error": None,
        "sector_active_limit": None,
        "factor_active_limit": None,
        "industry_active_range": None,
        "style_active_ranges": None,
        "candidate_weight_range": None,
        "weight_sum_tolerance": 1.0e-8,
        "constraint_tolerance": 1.0e-6,
    },
    "baseline": {"top_n": 5},
}


def _merge_strict(default: dict[str, Any], supplied: dict[str, Any], path: str = "") -> dict[str, Any]:
    unknown = set(supplied) - set(default)
    if unknown:
        names = ", ".join(sorted(f"{path}{key}" for key in unknown))
        raise ConfigError(f"unknown configuration key(s): {names}")
    merged = deepcopy(default)
    for key, value in supplied.items():
        if isinstance(default[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}{key} must be a mapping")
            merged[key] = _merge_strict(default[key], value, f"{path}{key}.")
        else:
            merged[key] = value
    return merged


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise ConfigError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    return number


def _optional_limit(value: Any, name: str) -> float | dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            raise ConfigError(f"{name} mapping cannot be empty")
        return {
            str(key): _finite_number(item, f"{name}.{key}", minimum=0.0)
            for key, item in value.items()
        }
    return _finite_number(value, name, minimum=0.0)


def _exposure_range(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    keys = set(value)
    if keys == {"target_active", "tolerance"}:
        target = _finite_number(value["target_active"], f"{name}.target_active")
        tolerance = _finite_number(
            value["tolerance"], f"{name}.tolerance", minimum=0.0
        )
        return {"lower_active": target - tolerance, "upper_active": target + tolerance}
    if keys == {"min_active", "max_active"}:
        lower = _finite_number(value["min_active"], f"{name}.min_active")
        upper = _finite_number(value["max_active"], f"{name}.max_active")
        if lower > upper:
            raise ConfigError(f"{name}.min_active must not exceed max_active")
        return {"lower_active": lower, "upper_active": upper}
    if keys == {"lower_active", "upper_active"}:
        lower = _finite_number(value["lower_active"], f"{name}.lower_active")
        upper = _finite_number(value["upper_active"], f"{name}.upper_active")
        if lower > upper:
            raise ConfigError(f"{name}.lower_active must not exceed upper_active")
        return {"lower_active": lower, "upper_active": upper}
    raise ConfigError(
        f"{name} must contain exactly target_active+tolerance, min_active+max_active, "
        "or lower_active+upper_active"
    )


def _industry_ranges(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    name = "constraints.industry_active_range"
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    unknown = set(value) - {"default", "overrides"}
    if unknown:
        raise ConfigError(
            "unknown configuration key(s): "
            + ", ".join(sorted(f"{name}.{key}" for key in unknown))
        )
    default = value.get("default")
    overrides = value.get("overrides", {})
    if default is None and not overrides:
        raise ConfigError(f"{name} must define default or overrides")
    if not isinstance(overrides, dict):
        raise ConfigError(f"{name}.overrides must be a mapping")
    return {
        "default": None if default is None else _exposure_range(default, f"{name}.default"),
        "overrides": {
            str(industry): _exposure_range(specification, f"{name}.overrides.{industry}")
            for industry, specification in overrides.items()
        },
    }


def _style_ranges(value: Any) -> dict[str, dict[str, Any]] | None:
    if value is None:
        return None
    name = "constraints.style_active_ranges"
    if not isinstance(value, dict) or not value:
        raise ConfigError(f"{name} must be a non-empty mapping")
    result: dict[str, dict[str, Any]] = {}
    for raw_factor, raw_specification in value.items():
        factor = str(raw_factor).strip()
        label = f"{name}.{factor}"
        if not factor:
            raise ConfigError(f"{name} contains an empty factor name")
        if factor.upper().startswith("INDUSTRY:"):
            raise ConfigError(f"{label} is an industry dummy; use industry_active_range")
        if not isinstance(raw_specification, dict):
            raise ConfigError(f"{label} must be a mapping")
        specification = dict(raw_specification)
        enabled = specification.pop("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{label}.enabled must be boolean")
        if not enabled:
            if specification:
                raise ConfigError(f"{label} is disabled and must not define an exposure range")
            result[factor] = {"enabled": False}
            continue
        result[factor] = {"enabled": True, **_exposure_range(specification, label)}
    return result


def _candidate_weight_range(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    name = "constraints.candidate_weight_range"
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    unknown = set(value) - {"min_weight", "max_weight"}
    if unknown:
        raise ConfigError(
            "unknown configuration key(s): "
            + ", ".join(sorted(f"{name}.{key}" for key in unknown))
        )
    if not value or all(value.get(key) is None for key in ("min_weight", "max_weight")):
        raise ConfigError(f"{name} must define min_weight or max_weight")
    lower = (
        0.0
        if value.get("min_weight") is None
        else _finite_number(value["min_weight"], f"{name}.min_weight", minimum=0.0)
    )
    upper = (
        1.0
        if value.get("max_weight") is None
        else _finite_number(value["max_weight"], f"{name}.max_weight", minimum=0.0)
    )
    if lower > 1.0 or upper > 1.0:
        raise ConfigError(f"{name} bounds must not exceed 1")
    if lower > upper:
        raise ConfigError(f"{name}.min_weight must not exceed max_weight")
    return {"min_weight": lower, "max_weight": upper}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["schema_version"] not in {1, 2, 3, 4, 5}:
        raise ConfigError("schema_version must equal 1, 2, 3, 4, or 5")

    signal = config["signal"]
    if signal["type"] not in {"rank_score", "expected_return"}:
        raise ConfigError("signal.type must be rank_score or expected_return")
    if not isinstance(signal["higher_is_better"], bool):
        raise ConfigError("signal.higher_is_better must be boolean")
    signal["winsorize_mad"] = _finite_number(
        signal["winsorize_mad"], "signal.winsorize_mad", minimum=0.0
    )
    if not isinstance(signal["zscore"], bool):
        raise ConfigError("signal.zscore must be boolean")
    if signal["rank_transform"] not in {"uniform", "normal_score", "power"}:
        raise ConfigError(
            "signal.rank_transform must be uniform, normal_score, or power"
        )
    signal["rank_power"] = _finite_number(
        signal["rank_power"], "signal.rank_power", minimum=0.0
    )
    if signal["rank_power"] <= 0:
        raise ConfigError("signal.rank_power must be positive")
    if signal["type"] == "expected_return" and signal["zscore"]:
        raise ConfigError("signal.zscore must be false for expected_return inputs")
    if signal["type"] == "expected_return" and (
        signal["rank_transform"] != "uniform" or signal["rank_power"] != 1.0
    ):
        raise ConfigError(
            "signal rank_transform and rank_power apply only to rank_score inputs"
        )
    signal["annualized_alpha_scale"] = _finite_number(
        signal["annualized_alpha_scale"],
        "signal.annualized_alpha_scale",
        minimum=0.0,
    )
    if signal["missing_prediction_policy"] not in {
        "neutral",
        "error_except_frozen",
    }:
        raise ConfigError(
            "signal.missing_prediction_policy must be neutral or error_except_frozen"
        )
    if (
        config["schema_version"] in {3, 4, 5}
        and signal["missing_prediction_policy"] != "error_except_frozen"
    ):
        raise ConfigError(
            "schema_version 3, 4, or 5 requires signal.missing_prediction_policy "
            "error_except_frozen"
        )

    covariance = config["covariance"]
    if covariance["risk_form"] not in {"asset_covariance", "factor_model"}:
        raise ConfigError(
            "covariance.risk_form must be asset_covariance or factor_model"
        )
    if not isinstance(covariance["annualized"], bool):
        raise ConfigError("covariance.annualized must be boolean")
    periods = covariance["periods_per_year"]
    if isinstance(periods, bool) or not isinstance(periods, int) or periods <= 0:
        raise ConfigError("covariance.periods_per_year must be a positive integer")
    covariance["eigenvalue_floor"] = _finite_number(
        covariance["eigenvalue_floor"],
        "covariance.eigenvalue_floor",
        minimum=0.0,
    )
    covariance["symmetry_tolerance"] = _finite_number(
        covariance["symmetry_tolerance"],
        "covariance.symmetry_tolerance",
        minimum=0.0,
    )

    optimizer = config["optimizer"]
    if optimizer["objective_mode"] not in {
        "mean_variance",
        "score_max_te",
        "lexicographic_signal_cost",
    }:
        raise ConfigError(
            "optimizer.objective_mode must be mean_variance, score_max_te, or "
            "lexicographic_signal_cost"
        )
    if optimizer["solver_backend"] not in {"scipy_slsqp", "scipy_highs", "cvxpy", "auto"}:
        raise ConfigError(
            "optimizer.solver_backend must be scipy_slsqp, scipy_highs, cvxpy, or auto"
        )
    for key in (
        "risk_aversion",
        "turnover_penalty",
        "smoothing_epsilon",
        "ftol",
        "stability_penalty",
    ):
        optimizer[key] = _finite_number(
            optimizer[key], f"optimizer.{key}", minimum=0.0
        )
    if optimizer["risk_aversion"] <= 0:
        raise ConfigError("optimizer.risk_aversion must be positive")
    if optimizer["smoothing_epsilon"] <= 0:
        raise ConfigError("optimizer.smoothing_epsilon must be positive")
    if optimizer["ftol"] <= 0:
        raise ConfigError("optimizer.ftol must be positive")
    if optimizer["stability_penalty"] <= 0:
        raise ConfigError("optimizer.stability_penalty must be positive")
    optimizer["minimum_signal_capture"] = _finite_number(
        optimizer["minimum_signal_capture"],
        "optimizer.minimum_signal_capture",
        minimum=0.0,
    )
    if optimizer["minimum_signal_capture"] > 1.0:
        raise ConfigError("optimizer.minimum_signal_capture must not exceed 1")
    iterations = optimizer["max_iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ConfigError("optimizer.max_iterations must be a positive integer")

    constraints = config["constraints"]
    for key in ("max_weight", "max_active_weight", "weight_sum_tolerance", "constraint_tolerance"):
        constraints[key] = _finite_number(
            constraints[key], f"constraints.{key}", minimum=0.0
        )
    if constraints["max_weight"] <= 0:
        raise ConfigError("constraints.max_weight must be positive")
    for key in ("max_turnover", "max_tracking_error"):
        if constraints[key] is not None:
            constraints[key] = _finite_number(
                constraints[key], f"constraints.{key}", minimum=0.0
            )
    constraints["sector_active_limit"] = _optional_limit(
        constraints["sector_active_limit"], "constraints.sector_active_limit"
    )
    constraints["factor_active_limit"] = _optional_limit(
        constraints["factor_active_limit"], "constraints.factor_active_limit"
    )
    constraints["industry_active_range"] = _industry_ranges(
        constraints["industry_active_range"]
    )
    constraints["style_active_ranges"] = _style_ranges(
        constraints["style_active_ranges"]
    )
    constraints["candidate_weight_range"] = _candidate_weight_range(
        constraints["candidate_weight_range"]
    )
    if (
        constraints["sector_active_limit"] is not None
        and constraints["industry_active_range"] is not None
    ):
        raise ConfigError(
            "configure only one of sector_active_limit and industry_active_range"
        )
    if (
        constraints["factor_active_limit"] is not None
        and constraints["style_active_ranges"] is not None
    ):
        raise ConfigError(
            "configure only one of factor_active_limit and style_active_ranges"
        )
    if config["schema_version"] in {2, 3, 4, 5}:
        styles = constraints["style_active_ranges"] or {}
        if "SIZE" not in styles or not styles["SIZE"].get("enabled", False):
            raise ConfigError(
                "constraints.style_active_ranges.SIZE must be enabled in schema_version 2, 3, 4, or 5"
            )
    if optimizer["objective_mode"] in {"score_max_te", "lexicographic_signal_cost"}:
        if signal["type"] != "rank_score":
            raise ConfigError(
                "score-based optimizer objectives require signal.type rank_score"
            )
        if constraints["max_tracking_error"] is None:
            raise ConfigError(
                "score-based optimizer objectives require constraints.max_tracking_error"
            )

    if (
        config["schema_version"] == 5
        and optimizer["objective_mode"] != "lexicographic_signal_cost"
    ):
        raise ConfigError(
            "schema_version 5 requires optimizer.objective_mode "
            "lexicographic_signal_cost"
        )

    config["cost_model"]["linear_cost_bps"] = _finite_number(
        config["cost_model"]["linear_cost_bps"],
        "cost_model.linear_cost_bps",
        minimum=0.0,
    )

    top_n = config["baseline"]["top_n"]
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ConfigError("baseline.top_n must be a positive integer")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        supplied = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ConfigError("configuration root must be a mapping")
    return validate_config(_merge_strict(DEFAULT_CONFIG, supplied))
