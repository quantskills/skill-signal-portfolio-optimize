from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.config import DEFAULT_CONFIG, load_config  # noqa: E402
from portfolio_runtime.errors import ConfigError  # noqa: E402
from portfolio_runtime.optimizer import optimize_portfolio  # noqa: E402


def _optimizer(blend: float = 0.20) -> dict[str, object]:
    result = deepcopy(DEFAULT_CONFIG["optimizer"])
    result.update(
        {
            "objective_mode": "blended_alpha_preserving_minimum_variance",
            "solver_backend": "clarabel_socp",
            "fallback_policy": "error",
            "blend_strength": blend,
            "minimum_alpha_capture": 1.0,
        }
    )
    return result


def _constraints() -> dict[str, object]:
    result = deepcopy(DEFAULT_CONFIG["constraints"])
    result.update(
        {
            "max_weight": 1.0,
            "max_active_weight": 1.0,
            "candidate_weight_range": {
                "min_weight": 1.0,
                "max_weight": 1.0,
            },
        }
    )
    return result


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="CVXPY is not installed",
)
def test_alpha_preserving_blend_reduces_risk_without_losing_alpha() -> None:
    tickers = pd.Index(["A", "B", "C", "D"], name="ticker")
    expected = pd.Series([0.04, 0.03, 0.02, 0.01], index=tickers)
    covariance = pd.DataFrame(
        np.diag([0.40, 0.30, 0.20, 0.10]),
        index=tickers,
        columns=tickers,
    )
    benchmark = pd.Series(0.25, index=tickers)
    anchor = pd.Series(0.25, index=tickers)
    tradable = pd.Series(True, index=tickers)
    candidate = pd.Series(True, index=tickers)

    result = optimize_portfolio(
        expected,
        covariance,
        benchmark,
        anchor,
        None,
        None,
        tradable,
        _optimizer(),
        _constraints(),
        anchor_weights=anchor,
        candidate_mask=candidate,
    )

    anchor_alpha = float(anchor @ expected)
    optimized_alpha = float(result.weights @ expected)
    anchor_variance = float(anchor @ covariance @ anchor)
    optimized_variance = float(result.weights @ covariance @ result.weights)
    assert result.constraints["passed"]
    assert optimized_alpha >= anchor_alpha - 1.0e-6
    assert optimized_variance <= anchor_variance + 1.0e-7
    assert result.solver["alpha_capture_ratio"] >= 1.0 - 1.0e-6
    assert result.solver["anchor_reallocation"] <= 0.20 + 1.0e-6


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="CVXPY is not installed",
)
def test_zero_alpha_blend_exactly_returns_anchor() -> None:
    tickers = pd.Index(["A", "B", "C"], name="ticker")
    expected = pd.Series([0.03, 0.02, 0.01], index=tickers)
    covariance = pd.DataFrame(np.eye(3), index=tickers, columns=tickers)
    anchor = pd.Series(1.0 / 3.0, index=tickers)
    result = optimize_portfolio(
        expected,
        covariance,
        anchor,
        anchor,
        None,
        None,
        pd.Series(True, index=tickers),
        _optimizer(blend=0.0),
        _constraints(),
        anchor_weights=anchor,
        candidate_mask=pd.Series(True, index=tickers),
    )
    assert np.allclose(result.weights, anchor, atol=1.0e-12)
    assert result.solver["backend"] == "anchor_only"
    assert result.solver["alpha_capture_ratio"] == pytest.approx(1.0)


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="CVXPY is not installed",
)
def test_anchor_style_range_limits_blended_weight_deviation() -> None:
    tickers = pd.Index(["A", "B", "C", "D"], name="ticker")
    expected = pd.Series([0.08, 0.04, 0.02, 0.01], index=tickers)
    covariance = pd.DataFrame(np.diag([0.20, 0.30, 0.40, 0.50]), index=tickers, columns=tickers)
    anchor = pd.Series(0.25, index=tickers)
    config = _optimizer(blend=1.0)
    config["minimum_alpha_capture"] = 0.50
    constraints = _constraints()
    constraints["anchor_style_active_ranges"] = {
        "STYLE": {"enabled": True, "lower_active": -0.01, "upper_active": 0.01}
    }
    exposures = pd.DataFrame({"STYLE": [1.0, -1.0, 0.0, 0.0]}, index=tickers)
    result = optimize_portfolio(
        expected, covariance, anchor, anchor, None, exposures,
        pd.Series(True, index=tickers), config, constraints,
        anchor_weights=anchor, candidate_mask=pd.Series(True, index=tickers),
    )
    assert abs(float((result.weights - anchor).dot(exposures["STYLE"]))) <= 0.01 + 1.0e-7


def test_schema8_requires_expected_return_input(tmp_path: Path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["schema_version"] = 8
    config["signal"].update(
        {
            "type": "expected_return",
            "zscore": False,
            "missing_prediction_policy": "role_aware",
        }
    )
    config["covariance"]["risk_form"] = "factor_model"
    config["optimizer"].update(
        {
            "objective_mode": "blended_alpha_preserving_minimum_variance",
            "solver_backend": "clarabel_socp",
            "fallback_policy": "error",
        }
    )
    config["constraints"]["candidate_weight_range"] = {
        "min_weight": 1.0,
        "max_weight": 1.0,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert load_config(path)["schema_version"] == 8

    config["signal"]["type"] = "rank_score"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="signal.type expected_return"):
        load_config(path)
