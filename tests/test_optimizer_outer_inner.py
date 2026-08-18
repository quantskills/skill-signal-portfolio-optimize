from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.config import DEFAULT_CONFIG  # noqa: E402
from portfolio_runtime.optimizer import optimize_portfolio  # noqa: E402


def test_highs_outer_inner_solves_binding_tracking_error() -> None:
    tickers = pd.Index(["A", "B", "C", "D"], name="ticker")
    benchmark = pd.Series(0.25, index=tickers)
    covariance = pd.DataFrame(np.eye(4) * 0.1, index=tickers, columns=tickers)
    expected = pd.Series(0.0, index=tickers)
    score = pd.Series([4.0, 3.0, 2.0, 1.0], index=tickers)
    tradable = pd.Series(True, index=tickers)
    optimizer = deepcopy(DEFAULT_CONFIG["optimizer"])
    optimizer.update(
        {
            "objective_mode": "score_max_te",
            "solver_backend": "scipy_slsqp",
            "max_iterations": 500,
            "ftol": 1.0e-8,
        }
    )
    constraints = deepcopy(DEFAULT_CONFIG["constraints"])
    constraints.update(
        {
            "max_weight": 0.6,
            "max_active_weight": 0.4,
            "max_tracking_error": 0.05,
        }
    )

    result = optimize_portfolio(
        expected,
        covariance,
        benchmark,
        None,
        None,
        None,
        tradable,
        optimizer,
        constraints,
        signal_score=score,
    )

    assert result.constraints["passed"]
    assert result.constraints["tracking_error"] <= 0.050001
    assert result.solver["backend"] == "scipy_highs_outer_inner"
    assert result.solver["risk_anchor_iterations"] > 0
    assert result.solver["certified_optimality_gap"] <= 1.0e-5
