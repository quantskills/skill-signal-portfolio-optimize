from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.config import load_config  # noqa: E402
from portfolio_runtime.errors import ConfigError  # noqa: E402
from portfolio_runtime.rolling import run_rolling_experiment  # noqa: E402


def _v5_config(*, risk_form: str = "factor_model", capture: float = 0.995) -> dict:
    return {
        "schema_version": 5,
        "signal": {
            "type": "rank_score",
            "higher_is_better": True,
            "winsorize_mad": 5.0,
            "zscore": True,
            "rank_transform": "normal_score",
            "rank_power": 1.0,
            "annualized_alpha_scale": 0.05,
            "missing_prediction_policy": "error_except_frozen",
        },
        "covariance": {
            "risk_form": risk_form,
            "annualized": True,
            "periods_per_year": 252,
            "eigenvalue_floor": 1.0e-10,
            "symmetry_tolerance": 1.0e-10,
        },
        "optimizer": {
            "objective_mode": "lexicographic_signal_cost",
            "solver_backend": "auto",
            "risk_aversion": 5.0,
            "turnover_penalty": 0.001,
            "smoothing_epsilon": 1.0e-8,
            "max_iterations": 2000,
            "ftol": 1.0e-9,
            "minimum_signal_capture": capture,
            "stability_penalty": 1.0e-8,
        },
        "cost_model": {"linear_cost_bps": 7.0},
        "constraints": {
            "max_weight": 0.60,
            "max_active_weight": 0.40,
            "max_turnover": 0.30,
            "max_tracking_error": 0.30,
            "sector_active_limit": None,
            "factor_active_limit": None,
            "industry_active_range": None,
            "style_active_ranges": {
                "SIZE": {
                    "enabled": True,
                    "target_active": 0.0,
                    "tolerance": 1.0,
                }
            },
            "candidate_weight_range": None,
            "weight_sum_tolerance": 1.0e-8,
            "constraint_tolerance": 1.0e-6,
        },
        "baseline": {"top_n": 2},
    }


def test_schema_v5_parses_and_rejects_invalid_capture(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_v5_config(), sort_keys=False), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["schema_version"] == 5
    assert loaded["constraints"]["candidate_weight_range"] is None
    assert loaded["cost_model"]["linear_cost_bps"] == 7.0

    invalid = _v5_config(capture=1.01)
    path.write_text(yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="must not exceed 1"):
        load_config(path)


def test_factor_form_rolling_uses_one_cost_source_and_writes_v1_contract(
    tmp_path: Path,
) -> None:
    rebalance_dates = [20230102, 20230104]
    all_dates = [20230102, 20230103, 20230104, 20230105]
    tickers = ["A", "B", "C", "D"]
    signal = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "prediction": 4.0 - index}
            for date in rebalance_dates
            for index, ticker in enumerate(tickers)
        ]
    )
    benchmark = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "benchmark_weight": 0.25}
            for date in rebalance_dates
            for ticker in tickers
        ]
    )
    returns = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "return": 0.001 * (index + 1) * (1 if position % 2 else -1),
            }
            for position, date in enumerate(all_dates)
            for index, ticker in enumerate(tickers)
        ]
    )
    config_path = tmp_path / "config.yaml"
    signal_path = tmp_path / "signal.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    returns_path = tmp_path / "returns.parquet"
    risk_root = tmp_path / "risk"
    checkpoint_root = tmp_path / "checkpoints"
    output = tmp_path / "output"
    config_path.write_text(
        yaml.safe_dump(_v5_config(), sort_keys=False), encoding="utf-8"
    )
    signal.to_parquet(signal_path, index=False)
    benchmark.to_parquet(benchmark_path, index=False)
    returns.to_parquet(returns_path, index=False)
    exposure = pd.DataFrame(
        {
            "MARKET": [1.0, 1.0, 1.0, 1.0],
            "SIZE": [-1.0, -0.5, 0.5, 1.0],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    factor_covariance = pd.DataFrame(
        [[0.02, 0.0], [0.0, 0.01]],
        index=exposure.columns,
        columns=exposure.columns,
    )
    specific = pd.Series(0.05, index=exposure.index, name="specific_var")
    for date in rebalance_dates:
        date_root = risk_root / f"date={date}"
        date_root.mkdir(parents=True)
        exposure.to_parquet(date_root / "exposures.parquet")
        factor_covariance.to_parquet(date_root / "factor_cov.parquet")
        specific.to_frame().to_parquet(date_root / "specific_var.parquet")

    result = run_rolling_experiment(
        config_path=config_path,
        signal_file=signal_path,
        benchmark_file=benchmark_path,
        asset_returns_file=returns_path,
        exposure_root=risk_root,
        factor_covariance_root=risk_root,
        specific_variance_root=risk_root,
        checkpoint_root=checkpoint_root,
        output_dir=output,
    )

    assert result["status"] == "success"
    summary = json.loads((output / "optimization_summary.json").read_text())
    manifest = json.loads((output / "rolling_manifest.json").read_text())
    diagnostics = pd.read_parquet(output / "optimization_diagnostics.parquet")
    daily = pd.read_parquet(output / "daily_performance.parquet")
    assert summary["risk_form"] == "factor_model"
    assert summary["fallback_count"] == 0
    assert summary["all_constraints_passed"]
    assert summary["signal_capture_ratio"]["minimum"] >= 0.995 - 1.0e-6
    assert manifest["cost_model_resolution"]["source"] == "config"
    assert manifest["transaction_cost_bps"] == 7.0
    assert len(manifest["factor_risk_inputs"]) == 2
    assert not manifest["covariance_inputs"]
    assert diagnostics["risk_form"].eq("factor_model").all()
    optimized = daily.loc[daily["portfolio"].eq("risk_optimized")]
    assert np.allclose(
        optimized["transaction_cost"], optimized["turnover"] * 7.0 / 10000.0
    )
    manifests = list(checkpoint_root.glob("date=*/run=*/checkpoint_manifest.json"))
    assert len(manifests) == 2
    checkpoint = json.loads(manifests[0].read_text())
    assert checkpoint["signature_inputs"]["risk_form"] == "factor_model"
    assert len(checkpoint["signature_inputs"]["runtime_source_sha256"]) == 64
    assert checkpoint["signature_inputs"]["factor_covariance_sha256"]
    assert checkpoint["signature_inputs"]["specific_variance_sha256"]
