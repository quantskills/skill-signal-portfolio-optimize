from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.config import DEFAULT_CONFIG, validate_config
from portfolio_runtime.conic_optimizer import clear_conic_problem_cache
from portfolio_runtime.errors import ConfigError
from portfolio_runtime.optimizer import optimize_portfolio
from portfolio_runtime.risk import PortfolioRisk


def factor_risk() -> PortfolioRisk:
    tickers = pd.Index(["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"])
    exposures = pd.DataFrame(
        [
            [1.0, -0.8],
            [1.0, -0.2],
            [1.0, 0.3],
            [1.0, 0.9],
        ],
        index=tickers,
        columns=["MARKET", "SIZE"],
    )
    factor_covariance = pd.DataFrame(
        [[0.025, 0.003], [0.003, 0.015]],
        index=exposures.columns,
        columns=exposures.columns,
    )
    specific_variance = pd.Series(
        [0.010, 0.012, 0.009, 0.011], index=tickers
    )
    return PortfolioRisk(
        form="factor_model",
        tickers=tickers,
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variance=specific_variance,
    )


def optimizer_config(objective_mode: str) -> dict[str, object]:
    config = deepcopy(DEFAULT_CONFIG["optimizer"])
    config.update(
        {
            "objective_mode": objective_mode,
            "solver_backend": "clarabel_socp",
            "fallback_policy": "error",
            "max_iterations": 500,
            "ftol": 1.0e-8,
            "minimum_signal_capture": 0.99,
            "conic_cache_size": 4,
            "warm_start": True,
        }
    )
    return config


def constraint_config() -> dict[str, object]:
    config = deepcopy(DEFAULT_CONFIG["constraints"])
    config.update(
        {
            "max_weight": 0.60,
            "max_active_weight": 0.50,
            "max_turnover": 0.80,
            "max_tracking_error": 0.30,
            "constraint_tolerance": 1.0e-6,
        }
    )
    return config


def solve(objective_mode: str):
    risk = factor_risk()
    index = risk.tickers
    expected_return = pd.Series([0.04, 0.02, -0.01, -0.03], index=index)
    signal_score = pd.Series([1.5, 0.5, -0.5, -1.5], index=index)
    benchmark = pd.Series(0.25, index=index)
    current = benchmark.copy()
    tradable = pd.Series(True, index=index)
    return optimize_portfolio(
        expected_return=expected_return,
        covariance=risk,
        benchmark=benchmark,
        current=current,
        sectors=None,
        exposures=None,
        tradable=tradable,
        optimizer_config=optimizer_config(objective_mode),
        constraint_config=constraint_config(),
        signal_score=signal_score,
        cost_model={"linear_cost_bps": 7.0},
    )


def require_clarabel() -> None:
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in set(cp.installed_solvers()):
        pytest.skip("CLARABEL is not installed")


def test_factor_square_root_matches_structural_and_dense_variance() -> None:
    risk = factor_risk()
    weights = np.asarray([0.13, -0.07, 0.05, -0.11])
    operator_variance = float(np.square(risk.square_root_operator() @ weights).sum())
    dense_variance = float(weights @ risk.dense().to_numpy() @ weights)
    assert operator_variance == pytest.approx(risk.variance(weights), abs=1.0e-14)
    assert operator_variance == pytest.approx(dense_variance, abs=1.0e-14)
    assert risk.square_root_operator().shape == (6, 4)


def test_clarabel_factor_risk_solve_reuses_parameterized_problem() -> None:
    require_clarabel()
    clear_conic_problem_cache()
    first = solve("score_max_te")
    second = solve("score_max_te")

    assert first.constraints["passed"]
    assert second.constraints["passed"]
    assert first.weights.min() >= 0.0
    assert second.weights.min() >= 0.0
    assert first.solver["backend"] == "cvxpy_clarabel_socp"
    assert first.solver["risk_form"] == "factor_model"
    assert first.solver["risk_operator_rows"] == 6
    assert first.solver["problem_is_dpp"] is True
    assert first.solver["problem_cache_hit"] is False
    assert second.solver["problem_cache_hit"] is True
    assert second.solver["problem_cache_key"] == first.solver["problem_cache_key"]
    np.testing.assert_allclose(first.weights, second.weights, atol=1.0e-8)


def test_clarabel_lexicographic_solve_reuses_both_stages() -> None:
    require_clarabel()
    clear_conic_problem_cache()
    first = solve("lexicographic_signal_cost")
    second = solve("lexicographic_signal_cost")

    assert first.constraints["passed"]
    assert second.constraints["passed"]
    assert first.weights.min() >= 0.0
    assert second.weights.min() >= 0.0
    assert first.solver["backend"] == "cvxpy_clarabel"
    assert first.solver["problem_is_dpp"] is True
    assert first.solver["problem_cache_hit"] is False
    assert second.solver["problem_cache_hit"] is True
    assert first.solver["signal_capture_ratio"] >= 0.99 - 1.0e-7
    np.testing.assert_allclose(first.weights, second.weights, atol=1.0e-8)

def solve_blended(
    blend_strength: float,
    *,
    frozen_first: bool = False,
):
    risk = factor_risk()
    index = risk.tickers
    expected_return = pd.Series([0.04, 0.02, -0.01, -0.03], index=index)
    benchmark = pd.Series(0.25, index=index)
    current = benchmark.copy()
    tradable = pd.Series([not frozen_first, True, True, True], index=index)
    anchor = pd.Series([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], index=index)
    candidate_mask = pd.Series([True, True, True, False], index=index)

    optimizer = optimizer_config("blended_minimum_variance")
    optimizer["blend_strength"] = blend_strength
    constraints = constraint_config()
    constraints.update(
        {
            "max_turnover": None,
            "max_tracking_error": None,
            "candidate_weight_range": {"min_weight": 0.95, "max_weight": 1.0},
        }
    )
    result = optimize_portfolio(
        expected_return=expected_return,
        covariance=risk,
        benchmark=benchmark,
        current=current,
        sectors=None,
        exposures=None,
        tradable=tradable,
        optimizer_config=optimizer,
        constraint_config=constraints,
        anchor_weights=anchor,
        candidate_mask=candidate_mask,
    )
    return result, anchor, current, tradable, candidate_mask, risk


def test_zero_blend_is_exact_execution_aware_anchor() -> None:
    result, _, current, _, _, _ = solve_blended(0.0, frozen_first=True)

    expected = pd.Series(
        [current.iloc[0], 0.375, 0.375, 0.0],
        index=result.weights.index,
    )
    np.testing.assert_allclose(result.weights, expected, atol=1.0e-12)
    assert result.solver["backend"] == "anchor_only"
    assert result.solver["blend_strength"] == 0.0
    assert result.constraints["passed"]


def test_conservative_blend_matches_convex_formula_and_reduces_risk() -> None:
    require_clarabel()
    blended, anchor, _, _, candidate_mask, risk = solve_blended(0.10)
    endpoint, _, _, _, _, _ = solve_blended(1.0)

    expected = 0.90 * anchor + 0.10 * endpoint.weights
    np.testing.assert_allclose(blended.weights, expected, atol=1.0e-7)
    assert float(blended.weights[candidate_mask].sum()) >= 0.95 - 1.0e-7
    assert risk.variance(blended.weights) <= risk.variance(anchor) + 1.0e-9
    assert blended.solver["predicted_risk_reduction"] >= -1.0e-8
    assert blended.solver["anchor_reallocation"] <= 0.10 + 1.0e-7
    assert blended.solver["objective_mode"] == "blended_minimum_variance"
    assert blended.constraints["passed"]

def test_blend_endpoint_respects_original_candidate_mask() -> None:
    require_clarabel()
    risk = factor_risk()
    index = risk.tickers
    expected_return = pd.Series(0.0, index=index)
    benchmark = pd.Series(0.25, index=index)
    current = benchmark.copy()
    tradable = pd.Series(True, index=index)
    # The anchor contains a small non-candidate holding. The endpoint must not
    # treat that holding as part of the signal candidate universe.
    anchor = pd.Series([0.475, 0.475, 0.05, 0.0], index=index)
    candidate_mask = pd.Series([True, True, False, False], index=index)
    optimizer = optimizer_config("blended_minimum_variance")
    optimizer["blend_strength"] = 0.20
    constraints = constraint_config()
    constraints.update(
        {
            "max_turnover": None,
            "max_tracking_error": None,
            "candidate_weight_range": {
                "min_weight": 0.95,
                "max_weight": 1.0,
            },
        }
    )

    result = optimize_portfolio(
        expected_return=expected_return,
        covariance=risk,
        benchmark=benchmark,
        current=current,
        sectors=None,
        exposures=None,
        tradable=tradable,
        optimizer_config=optimizer,
        constraint_config=constraints,
        anchor_weights=anchor,
        candidate_mask=candidate_mask,
    )

    assert result.constraints["passed"]
    assert float(result.weights[candidate_mask].sum()) >= 0.95 - 1.0e-7


def test_conservative_blend_preserves_frozen_holdings() -> None:
    require_clarabel()
    result, _, current, tradable, _, _ = solve_blended(0.20, frozen_first=True)

    frozen = ~tradable
    np.testing.assert_allclose(
        result.weights[frozen],
        current[frozen],
        atol=1.0e-8,
    )
    assert result.constraints["passed"]


def schema_v6_config() -> dict[str, object]:
    config = deepcopy(DEFAULT_CONFIG)
    config["schema_version"] = 6
    config["signal"]["missing_prediction_policy"] = "role_aware"
    config["covariance"]["risk_form"] = "factor_model"
    config["optimizer"].update(
        {
            "objective_mode": "blended_minimum_variance",
            "solver_backend": "clarabel_socp",
            "fallback_policy": "error",
            "blend_strength": 0.10,
        }
    )
    config["constraints"]["candidate_weight_range"] = {
        "min_weight": 0.95,
        "max_weight": 1.0,
    }
    return config


def test_schema_v6_accepts_conservative_blend_contract() -> None:
    config = validate_config(schema_v6_config())

    assert config["optimizer"]["blend_strength"] == 0.10
    assert config["constraints"]["max_tracking_error"] is None
    assert config["constraints"]["industry_active_range"] is None


def test_schema_v6_rejects_hard_risk_constraints() -> None:
    config = schema_v6_config()
    config["constraints"]["max_tracking_error"] = 0.06

    with pytest.raises(ConfigError, match="does not allow hard"):
        validate_config(config)


def schema_v7_config() -> dict[str, object]:
    config = schema_v6_config()
    config["schema_version"] = 7
    config["optimizer"].update(
        {
            "objective_mode": "signal_preserving_minimum_variance",
            "minimum_signal_capture": 0.97,
            "turnover_penalty": 0.00035,
            "blend_strength": 0.0,
        }
    )
    return config


def test_schema_v7_accepts_signal_preserving_contract() -> None:
    config = validate_config(schema_v7_config())

    assert config["optimizer"]["minimum_signal_capture"] == 0.97
    assert config["optimizer"]["turnover_penalty"] == 0.00035


def test_signal_preserving_minimum_variance_respects_floor() -> None:
    require_clarabel()
    risk = factor_risk()
    index = risk.tickers
    signal_score = pd.Series([1.5, 0.5, -0.5, -1.5], index=index)
    expected_return = pd.Series(0.0, index=index)
    benchmark = pd.Series(0.25, index=index)
    current = benchmark.copy()
    tradable = pd.Series(True, index=index)
    anchor = pd.Series(
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        index=index,
    )
    candidate_mask = pd.Series([True, True, True, False], index=index)
    optimizer = optimizer_config(
        "signal_preserving_minimum_variance"
    )
    optimizer.update(
        {
            "minimum_signal_capture": 0.97,
            "turnover_penalty": 0.00035,
        }
    )
    constraints = constraint_config()
    constraints.update(
        {
            "max_turnover": None,
            "max_tracking_error": None,
            "candidate_weight_range": {
                "min_weight": 0.95,
                "max_weight": 1.0,
            },
        }
    )

    result = optimize_portfolio(
        expected_return=expected_return,
        covariance=risk,
        benchmark=benchmark,
        current=current,
        sectors=None,
        exposures=None,
        tradable=tradable,
        optimizer_config=optimizer,
        constraint_config=constraints,
        signal_score=signal_score,
        anchor_weights=anchor,
        candidate_mask=candidate_mask,
    )

    anchor_utility = float(signal_score @ (anchor - benchmark))
    final_utility = float(
        signal_score @ (result.weights - benchmark)
    )
    capture = 1.0 - max(
        anchor_utility - final_utility, 0.0
    ) / abs(anchor_utility)
    anchor_active = anchor - benchmark
    optimized_active = result.weights - benchmark

    assert result.constraints["passed"]
    assert capture >= 0.97 - 1.0e-7
    assert result.solver["signal_capture_ratio"] >= 0.97 - 1.0e-7
    assert risk.variance(optimized_active) <= (
        risk.variance(anchor_active) + 1.0e-8
    )
    assert (
        result.solver["objective_mode"]
        == "signal_preserving_minimum_variance"
    )
