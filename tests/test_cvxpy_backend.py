from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.optimizer import optimize_portfolio  # noqa: E402


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="CVXPY is not installed in the local compatibility environment",
)
def test_cvxpy_score_backend_returns_independently_validated_solution() -> None:
    tickers = pd.Index([f"{value:06d}.SZ" for value in range(1, 5)])
    expected = pd.Series(0.0, index=tickers)
    score = pd.Series([1.0, 0.5, -0.5, -1.0], index=tickers)
    covariance = pd.DataFrame(np.eye(4) * 0.10, index=tickers, columns=tickers)
    benchmark = pd.Series(0.25, index=tickers)
    current = benchmark.copy()
    tradable = pd.Series(True, index=tickers)
    optimizer = {
        "objective_mode": "score_max_te",
        "solver_backend": "cvxpy",
        "risk_aversion": 5.0,
        "turnover_penalty": 0.001,
        "smoothing_epsilon": 1.0e-8,
        "max_iterations": 2000,
        "ftol": 1.0e-8,
    }
    constraints = {
        "max_weight": 0.60,
        "max_active_weight": 0.40,
        "max_turnover": 0.50,
        "max_tracking_error": 0.30,
        "sector_active_limit": None,
        "factor_active_limit": None,
        "industry_active_range": None,
        "style_active_ranges": None,
        "weight_sum_tolerance": 1.0e-8,
        "constraint_tolerance": 1.0e-6,
    }

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
    )

    assert result.solver["backend"] == "cvxpy"
    assert result.constraints["passed"]
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.weights.iloc[0] > result.weights.iloc[-1]
