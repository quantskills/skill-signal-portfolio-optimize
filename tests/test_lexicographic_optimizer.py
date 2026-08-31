from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.config import DEFAULT_CONFIG  # noqa: E402
from portfolio_runtime.errors import OptimizationError  # noqa: E402
from portfolio_runtime.lexicographic import (  # noqa: E402
    StageSolution,
    select_lexicographic_backend,
    signal_utility,
    signal_utility_floor,
    solve_secondary_cost_problem_scipy,
)
from portfolio_runtime.optimizer import optimize_portfolio  # noqa: E402
from portfolio_runtime.risk import PortfolioRisk  # noqa: E402


def _problem() -> tuple:
    tickers = pd.Index(["A", "B", "C", "D"], name="ticker")
    expected = pd.Series(0.0, index=tickers)
    score = pd.Series([2.0, 1.0, -1.0, -2.0], index=tickers)
    covariance = pd.DataFrame(np.eye(4) * 0.10, index=tickers, columns=tickers)
    benchmark = pd.Series(0.25, index=tickers)
    current = pd.Series([0.10, 0.20, 0.30, 0.40], index=tickers)
    tradable = pd.Series(True, index=tickers)
    optimizer = deepcopy(DEFAULT_CONFIG["optimizer"])
    optimizer.update(
        {
            "objective_mode": "lexicographic_signal_cost",
            "solver_backend": "scipy_slsqp",
            "minimum_signal_capture": 0.90,
            "stability_penalty": 1.0e-8,
            "max_iterations": 2000,
            "ftol": 1.0e-10,
        }
    )
    constraints = deepcopy(DEFAULT_CONFIG["constraints"])
    constraints.update(
        {
            "max_weight": 0.60,
            "max_active_weight": 0.40,
            "max_turnover": 0.30,
            "max_tracking_error": 0.20,
        }
    )
    return expected, score, covariance, benchmark, current, tradable, optimizer, constraints


def _solve(*, order: list[str] | None = None, candidate_range=None, current=True):
    expected, score, covariance, benchmark, current_weights, tradable, optimizer, constraints = _problem()
    if order is not None:
        expected = expected.reindex(order)
        score = score.reindex(order)
        covariance = covariance.reindex(index=order, columns=order)
        benchmark = benchmark.reindex(order)
        current_weights = current_weights.reindex(order)
        tradable = tradable.reindex(order)
    constraints["candidate_weight_range"] = candidate_range
    return optimize_portfolio(
        expected,
        covariance,
        benchmark,
        current_weights if current else None,
        None,
        None,
        tradable,
        optimizer,
        constraints,
        signal_score=score,
        candidate_mask=pd.Series(True, index=expected.index),
        cost_model={"linear_cost_bps": 7.0},
    )


def test_lexicographic_two_stage_captures_signal_and_reduces_turnover() -> None:
    result = _solve()
    assert result.constraints["passed"]
    assert result.solver["primary_signal_utility"] > 0
    assert result.solver["signal_capture_ratio"] >= 0.90 - 1.0e-6
    assert result.solver["one_way_turnover"] < 0.30
    assert result.solver["estimated_transaction_cost"] == pytest.approx(
        result.solver["one_way_turnover"] * 7.0 / 10000.0
    )
    assert result.solver["backend_fallback_used"] is False


def test_lexicographic_is_repeatable_and_ticker_order_invariant() -> None:
    runs = [_solve().weights.sort_index() for _ in range(3)]
    for weights in runs[1:]:
        assert np.max(np.abs(weights - runs[0])) <= 1.0e-8
    reordered = _solve(order=["D", "B", "A", "C"]).weights.sort_index()
    assert np.max(np.abs(reordered - runs[0])) <= 1.0e-8


def test_inactive_candidate_range_does_not_change_unique_solution() -> None:
    unrestricted = _solve().weights.sort_index()
    inactive = _solve(
        candidate_range={"min_weight": 0.0, "max_weight": 1.0}
    ).weights.sort_index()
    assert np.max(np.abs(inactive - unrestricted)) <= 1.0e-8


def test_first_period_without_current_ignores_turnover_cap() -> None:
    result = _solve(current=False)
    assert result.constraints["passed"]
    assert result.constraints["one_way_turnover"] is None
    assert result.solver["estimated_transaction_cost"] is None


def test_frozen_holding_is_preserved_and_bound_exception_is_disclosed() -> None:
    expected, score, covariance, benchmark, current, tradable, optimizer, constraints = _problem()
    current = pd.Series([0.65, 0.15, 0.10, 0.10], index=expected.index)
    tradable.loc["A"] = False
    constraints["max_weight"] = 0.60
    constraints["max_active_weight"] = 0.50
    constraints["max_turnover"] = 0.30
    result = optimize_portfolio(
        expected,
        covariance,
        benchmark,
        current,
        None,
        None,
        tradable,
        optimizer,
        constraints,
        signal_score=score,
        cost_model={"linear_cost_bps": 7.0},
    )
    assert result.weights["A"] == pytest.approx(0.65, abs=1.0e-10)
    assert result.constraints["maximum_frozen_weight_deviation"] <= 1.0e-10
    assert result.constraints["frozen_bound_exceptions"][0]["ticker"] == "A"


def test_candidate_range_is_reduced_by_frozen_outside_holding() -> None:
    expected, score, covariance, benchmark, current, tradable, optimizer, constraints = _problem()
    current = pd.Series([0.30, 0.30, 0.20, 0.20], index=expected.index)
    tradable.loc["C"] = False
    constraints["max_turnover"] = None
    constraints["max_tracking_error"] = None
    constraints["candidate_weight_range"] = {"min_weight": 1.0, "max_weight": 1.0}
    candidate_mask = pd.Series([True, True, False, False], index=expected.index)

    result = optimize_portfolio(
        expected,
        covariance,
        benchmark,
        current,
        None,
        None,
        tradable,
        optimizer,
        constraints,
        signal_score=score,
        candidate_mask=candidate_mask,
        cost_model={"linear_cost_bps": 7.0},
    )

    detail = result.constraints["candidate_weight_range"]
    assert result.constraints["passed"]
    assert result.weights[candidate_mask].sum() == pytest.approx(0.80, abs=1.0e-8)
    assert detail["configured_min_weight"] == 1.0
    assert detail["configured_max_weight"] == 1.0
    assert detail["min_weight"] == pytest.approx(0.80)
    assert detail["max_weight"] == pytest.approx(0.80)
    assert detail["frozen_outside_candidate_weight"] == pytest.approx(0.20)
    assert detail["adjusted_for_frozen_outside"] is True


def test_factor_form_matches_dense_form() -> None:
    expected, score, covariance, benchmark, current, tradable, optimizer, constraints = _problem()
    exposures = pd.DataFrame(
        {"MARKET": [1.0, 1.0, 1.0, 1.0], "SIZE": [-1.0, -0.5, 0.5, 1.0]},
        index=expected.index,
    )
    factor_covariance = pd.DataFrame(
        [[0.02, 0.0], [0.0, 0.01]],
        index=exposures.columns,
        columns=exposures.columns,
    )
    specific = pd.Series(0.05, index=expected.index)
    factor_risk = PortfolioRisk(
        form="factor_model",
        tickers=expected.index,
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variance=specific,
    )
    dense = factor_risk.dense()
    common = dict(
        expected_return=expected,
        benchmark=benchmark,
        current=current,
        sectors=None,
        exposures=None,
        tradable=tradable,
        optimizer_config=optimizer,
        constraint_config=constraints,
        signal_score=score,
        cost_model={"linear_cost_bps": 7.0},
    )
    dense_result = optimize_portfolio(covariance=dense, **common)
    factor_result = optimize_portfolio(covariance=factor_risk, **common)
    assert factor_risk.variance((factor_result.weights - benchmark).to_numpy()) == pytest.approx(
        float(
            (factor_result.weights - benchmark).to_numpy()
            @ dense.to_numpy()
            @ (factor_result.weights - benchmark).to_numpy()
        ),
        abs=1.0e-12,
    )
    assert np.max(np.abs(dense_result.weights - factor_result.weights)) <= 1.0e-7


def test_signal_floor_handles_zero_and_negative_primary_utility() -> None:
    assert signal_utility_floor(0.0, 0.9) == 0.0
    assert signal_utility_floor(-2.0, 0.9) == pytest.approx(-2.2)
    assert signal_utility_floor(2.0, 0.9) == pytest.approx(1.8)


def test_secondary_accepts_primary_turnover_within_final_tolerance() -> None:
    expected, score, covariance, benchmark, current, tradable, optimizer, constraints = _problem()
    constraints["max_turnover"] = 0.10
    constraints["max_tracking_error"] = None
    constraints["constraint_tolerance"] = 1.0e-6
    delta = constraints["max_turnover"] + 0.95 * constraints["constraint_tolerance"]
    primary_weights = current.copy()
    primary_weights.iloc[0] += delta
    primary_weights.iloc[1] -= delta
    primary_utility = signal_utility(primary_weights, benchmark, score)
    primary = StageSolution(
        decision=primary_weights.to_numpy(dtype=float),
        objective_value=-primary_utility,
        iterations=0,
        status=0,
        message="boundary regression fixture",
    )
    risk = PortfolioRisk(
        form="asset_covariance",
        tickers=expected.index,
        asset_covariance=covariance,
    )
    solved = solve_secondary_cost_problem_scipy(
        primary=primary,
        primary_utility=primary_utility,
        utility_floor=signal_utility_floor(primary_utility, 0.99),
        signal_score=score,
        risk=risk,
        benchmark=benchmark,
        current=current,
        sectors=None,
        exposures=None,
        candidate_mask=pd.Series(True, index=expected.index),
        tradable=tradable,
        optimizer_config=optimizer,
        constraint_config=constraints,
        linear_cost_bps=7.0,
    )
    final_turnover = 0.5 * np.abs(solved.decision[: len(current)] - current).sum()
    assert final_turnover <= constraints["max_turnover"] + constraints["constraint_tolerance"]


def test_auto_backend_is_selected_before_solve_without_fallback() -> None:
    assert select_lexicographic_backend("auto") == "scipy_highs_lexicographic"
    with pytest.raises(OptimizationError, match="CLARABEL"):
        select_lexicographic_backend("cvxpy")
