from __future__ import annotations

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

from portfolio_runtime.config import DEFAULT_CONFIG, load_config
from portfolio_runtime.errors import ConfigError
from portfolio_runtime.optimizer import optimize_portfolio
from portfolio_runtime.risk import PortfolioRisk
from portfolio_runtime.selection import select_risk_aware_equal_weight


def _inputs():
    tickers = pd.Index(list("ABCDEFGH"), name="ticker")
    score = pd.Series(np.arange(8.0, 0.0, -1.0), index=tickers)
    anchor = pd.Series(0.0, index=tickers)
    anchor.loc[list("ABCD")] = 0.25
    candidate = pd.Series(True, index=tickers)
    covariance = pd.DataFrame(
        np.diag([1.0, 1.0, 10.0, 10.0, 0.01, 0.01, 1.0, 1.0]),
        index=tickers,
        columns=tickers,
    )
    return tickers, score, anchor, candidate, covariance


def _selection(risk_penalty: float) -> dict[str, object]:
    return {
        "mode": "risk_aware_boundary",
        "pool_size": 8,
        "protected_top_n": 2,
        "target_holdings": 4,
        "risk_penalty": risk_penalty,
    }


def _constraints() -> dict[str, object]:
    result = deepcopy(DEFAULT_CONFIG["constraints"])
    result.update(
        {
            "max_weight": 1.0,
            "max_active_weight": 1.0,
            "candidate_weight_range": {"min_weight": 1.0, "max_weight": 1.0},
        }
    )
    return result


def test_zero_risk_penalty_returns_exact_anchor() -> None:
    tickers, score, anchor, candidate, _ = _inputs()
    covariance = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    result = select_risk_aware_equal_weight(
        signal_score=score,
        risk_input=covariance,
        anchor_weights=anchor,
        candidate_mask=candidate,
        config=_selection(0.0),
    )
    assert np.allclose(result.weights, anchor)
    assert result.diagnostics["selection_replacement_count"] == 0
    assert result.diagnostics["selection_overlap_ratio"] == pytest.approx(1.0)


def test_execution_anchor_passthrough_defers_only_frozen_stock_targets() -> None:
    tickers, score, anchor, candidate, covariance = _inputs()
    current = anchor.copy()
    current.loc["A"] = 0.30
    current.loc["B"] = 0.20
    tradable = pd.Series(True, index=tickers)
    tradable.loc["A"] = False
    optimizer = deepcopy(DEFAULT_CONFIG["optimizer"])
    optimizer["objective_mode"] = "risk_aware_selection"
    result = optimize_portfolio(
        score * 0.01,
        covariance,
        pd.Series(1.0 / len(tickers), index=tickers),
        current,
        None,
        None,
        tradable,
        optimizer,
        _constraints(),
        signal_score=score,
        anchor_weights=anchor,
        candidate_mask=candidate,
        selection_config=_selection(0.0),
        execution_anchor_passthrough=True,
    )
    assert np.allclose(result.weights, anchor)
    assert result.solver["selection_replacement_count"] == 0
    assert result.solver["execution_anchor_passthrough"] is True
    assert result.solver["execution_deferred_freeze_count"] > 0
    assert not result.constraints["passed"]


def test_execution_anchor_passthrough_keeps_nonzero_risk_constraints() -> None:
    tickers, score, anchor, candidate, covariance = _inputs()
    optimizer = deepcopy(DEFAULT_CONFIG["optimizer"])
    optimizer["objective_mode"] = "risk_aware_selection"
    result = optimize_portfolio(
        score * 0.01, covariance,
        pd.Series(1.0 / len(tickers), index=tickers),
        anchor, None, None, pd.Series(True, index=tickers),
        optimizer, _constraints(), signal_score=score,
        anchor_weights=anchor, candidate_mask=candidate,
        selection_config=_selection(1.0),
        execution_anchor_passthrough=True,
    )
    assert result.constraints["passed"]
    assert result.solver["execution_anchor_passthrough"] is False


def test_risk_penalty_protects_top_names_and_replaces_boundary() -> None:
    _, score, anchor, candidate, covariance = _inputs()
    result = select_risk_aware_equal_weight(
        signal_score=score,
        risk_input=covariance,
        anchor_weights=anchor,
        candidate_mask=candidate,
        config=_selection(1.0),
    )
    selected = set(result.weights[result.weights > 0.0].index)
    assert {"A", "B"}.issubset(selected)
    assert len(selected) == 4
    assert result.diagnostics["selection_replacement_count"] <= 2
    assert float(result.weights.dot(covariance).dot(result.weights)) < float(
        anchor.dot(covariance).dot(anchor)
    )


def test_optimizer_mode_returns_audited_equal_weight_selection() -> None:
    tickers, score, anchor, candidate, covariance = _inputs()
    optimizer = deepcopy(DEFAULT_CONFIG["optimizer"])
    optimizer["objective_mode"] = "risk_aware_selection"
    result = optimize_portfolio(
        score * 0.01,
        covariance,
        pd.Series(1.0 / len(tickers), index=tickers),
        anchor,
        None,
        None,
        pd.Series(True, index=tickers),
        optimizer,
        _constraints(),
        signal_score=score,
        anchor_weights=anchor,
        candidate_mask=candidate,
        cost_model={"linear_cost_bps": 7.0},
        selection_config=_selection(1.0),
    )
    selected = result.weights[result.weights > 0.0]
    assert len(selected) == 4
    assert np.allclose(selected, 0.25)
    assert result.solver["backend"] == "deterministic_risk_aware_selection"
    assert result.solver["selection_replacement_count"] <= 2
    assert result.constraints["passed"]


def test_factor_risk_diagonal_matches_dense_covariance() -> None:
    tickers = pd.Index(["A", "B", "C"], name="ticker")
    exposures = pd.DataFrame(
        [[1.0, 0.2], [1.0, -0.1], [1.0, 0.4]],
        index=tickers,
        columns=["MARKET", "SIZE"],
    )
    factor_covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=exposures.columns,
        columns=exposures.columns,
    )
    specific = pd.Series([0.02, 0.03, 0.01], index=tickers)
    risk = PortfolioRisk(
        form="factor_model",
        tickers=tickers,
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variance=specific,
    )
    assert np.allclose(risk.diagonal(), np.diag(risk.dense()))


def test_schema10_requires_consistent_selection_config(tmp_path: Path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["schema_version"] = 10
    config["signal"].update(
        {"type": "rank_score", "missing_prediction_policy": "role_aware"}
    )
    config["covariance"]["risk_form"] = "factor_model"
    config["optimizer"].update(
        {"objective_mode": "risk_aware_selection", "fallback_policy": "error"}
    )
    config["selection"].update(
        {
            "mode": "risk_aware_boundary",
            "pool_size": 500,
            "protected_top_n": 180,
            "target_holdings": 200,
            "risk_penalty": 0.10,
        }
    )
    config["constraints"].update(
        {
            "max_weight": 0.02,
            "max_active_weight": 0.02,
            "candidate_weight_range": {"min_weight": 0.95, "max_weight": 1.0},
        }
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["schema_version"] == 10

    config["selection"]["protected_top_n"] = 200
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="protected_top_n"):
        load_config(path)
