from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from portfolio_runtime.config import DEFAULT_CONFIG
from portfolio_runtime.optimizer import optimize_portfolio
from portfolio_runtime.risk import PortfolioRisk


def test_strict_default_remains_unchanged() -> None:
    assert DEFAULT_CONFIG["optimizer"]["feasibility_mode"] == "strict"


def test_repair_mode_records_attempt_when_strict_is_infeasible() -> None:
    tickers = pd.Index(["A", "B", "C"])
    risk = PortfolioRisk(
        form="asset_covariance",
        tickers=tickers,
        asset_covariance=pd.DataFrame(np.eye(3) * 0.01, index=tickers, columns=tickers),
    )
    optimizer = deepcopy(DEFAULT_CONFIG["optimizer"])
    optimizer.update({
        "objective_mode": "lexicographic_signal_cost",
        "solver_backend": "clarabel_socp",
        "fallback_policy": "error",
        "feasibility_mode": "repair",
    })
    constraints = deepcopy(DEFAULT_CONFIG["constraints"])
    constraints.update({
        "max_weight": 1.0,
        "max_active_weight": 1.0,
        "max_tracking_error": None,
        "max_turnover": None,
        "industry_active_range": {"default": {"lower_active": -1e-9, "upper_active": 1e-9}, "overrides": {}},
        "style_active_ranges": {"SIZE": {"enabled": True, "lower_active": -1e-9, "upper_active": 1e-9}},
        "initial_cash_turnover_exempt": True,
    })
    result = optimize_portfolio(
        expected_return=pd.Series(0.0, index=tickers),
        covariance=risk,
        benchmark=pd.Series([0.8, 0.1, 0.1], index=tickers),
        current=None,
        sectors=pd.Series(["A", "B", "A"], index=tickers),
        exposures=pd.DataFrame({"SIZE": [1.0, -1.0, 0.0]}, index=tickers),
        tradable=pd.Series(True, index=tickers),
        optimizer_config=optimizer,
        constraint_config=constraints,
        signal_score=pd.Series([1.0, 0.5, 0.2], index=tickers),
    )
    assert result.solver["feasibility_mode"] == "repair"
    assert result.solver["repair_attempt"] >= 1
