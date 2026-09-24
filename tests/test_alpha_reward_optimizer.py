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


def _optimizer(
    *,
    reward_weight: float,
    minimum_capture: float = 0.99,
    blend: float = 0.20,
) -> dict[str, object]:
    result = deepcopy(DEFAULT_CONFIG["optimizer"])
    result.update(
        {
            "objective_mode": "blended_alpha_reward_minimum_variance",
            "solver_backend": "clarabel_socp",
            "fallback_policy": "error",
            "blend_strength": blend,
            "minimum_alpha_capture": minimum_capture,
            "alpha_reward_weight": reward_weight,
            "risk_aversion": 1.0,
        }
    )
    return result


def _preserving_optimizer(*, blend: float = 0.20) -> dict[str, object]:
    result = _optimizer(reward_weight=0.0, minimum_capture=1.0, blend=blend)
    result["objective_mode"] = "blended_alpha_preserving_minimum_variance"
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


def _inputs() -> tuple[
    pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    tickers = pd.Index(["A", "B", "C", "D"], name="ticker")
    expected = pd.Series([0.08, 0.04, 0.02, 0.01], index=tickers)
    covariance = pd.DataFrame(
        np.diag([0.20, 0.30, 0.40, 0.50]),
        index=tickers,
        columns=tickers,
    )
    anchor = pd.Series(0.25, index=tickers)
    tradable = pd.Series(True, index=tickers)
    candidate = pd.Series(True, index=tickers)
    return expected, covariance, anchor, tradable, candidate


def _solve(optimizer: dict[str, object]):
    expected, covariance, anchor, tradable, candidate = _inputs()
    return optimize_portfolio(
        expected,
        covariance,
        anchor,
        anchor,
        None,
        None,
        tradable,
        optimizer,
        _constraints(),
        anchor_weights=anchor,
        candidate_mask=candidate,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="CVXPY is not installed",
)
def test_zero_reward_matches_alpha_preserving_endpoint() -> None:
    preserving = _solve(_preserving_optimizer())
    zero_reward = _solve(_optimizer(reward_weight=0.0, minimum_capture=1.0))

    assert np.allclose(preserving.weights, zero_reward.weights, atol=1.0e-7)
    assert zero_reward.solver["alpha_reward_weight"] == 0.0
    assert zero_reward.solver["alpha_reward_contribution"] == pytest.approx(0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="CVXPY is not installed",
)
def test_positive_reward_increases_expected_return_and_respects_floor() -> None:
    risk_only = _solve(_optimizer(reward_weight=0.0))
    rewarded = _solve(_optimizer(reward_weight=0.01))
    expected, covariance, anchor, _, _ = _inputs()

    risk_only_alpha = float(risk_only.weights.dot(expected))
    rewarded_alpha = float(rewarded.weights.dot(expected))
    anchor_alpha = float(anchor.dot(expected))
    rewarded_variance = float(rewarded.weights.dot(covariance).dot(rewarded.weights))
    anchor_variance = float(anchor.dot(covariance).dot(anchor))

    assert rewarded_alpha > risk_only_alpha + 1.0e-6
    assert rewarded_alpha >= 0.99 * anchor_alpha - 1.0e-6
    assert rewarded_variance <= anchor_variance + 1.0e-7
    assert rewarded.solver["alpha_reward_contribution"] > 0.0
    assert rewarded.solver["normalized_alpha_scale"] == pytest.approx(anchor_alpha)
    assert rewarded.constraints["passed"]


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="CVXPY is not installed",
)
def test_reward_mode_allows_audited_risk_return_tradeoff() -> None:
    expected, _, anchor, tradable, candidate = _inputs()
    covariance = pd.DataFrame(
        np.diag([0.50, 0.40, 0.30, 0.20]),
        index=expected.index,
        columns=expected.index,
    )
    rewarded = optimize_portfolio(
        expected,
        covariance,
        anchor,
        anchor,
        None,
        None,
        tradable,
        _optimizer(reward_weight=0.10, blend=0.02),
        _constraints(),
        anchor_weights=anchor,
        candidate_mask=candidate,
    )

    anchor_alpha = float(anchor.dot(expected))
    optimized_alpha = float(rewarded.weights.dot(expected))
    anchor_volatility = float(np.sqrt(anchor.dot(covariance).dot(anchor)))
    optimized_volatility = float(
        np.sqrt(rewarded.weights.dot(covariance).dot(rewarded.weights))
    )

    assert optimized_alpha > anchor_alpha
    assert optimized_volatility > anchor_volatility
    assert optimized_alpha >= 0.99 * anchor_alpha - 1.0e-6
    assert rewarded.solver["predicted_risk_reduction"] < 0.0
    assert rewarded.constraints["passed"]


def test_schema9_requires_positive_alpha_reward(tmp_path: Path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["schema_version"] = 9
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
            "objective_mode": "blended_alpha_reward_minimum_variance",
            "solver_backend": "clarabel_socp",
            "fallback_policy": "error",
            "minimum_alpha_capture": 0.99,
            "alpha_reward_weight": 0.10,
        }
    )
    config["constraints"]["candidate_weight_range"] = {
        "min_weight": 1.0,
        "max_weight": 1.0,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["schema_version"] == 9
    assert loaded["optimizer"]["alpha_reward_weight"] == pytest.approx(0.10)

    config["optimizer"]["alpha_reward_weight"] = 0.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="alpha_reward_weight to be positive"):
        load_config(path)
