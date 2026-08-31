from __future__ import annotations

import json
from copy import deepcopy
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

from portfolio_runtime.backtest import (  # noqa: E402
    add_benchmark_relative_performance,
    backtest_targets,
    summarize_backtest,
)
from portfolio_runtime.config import DEFAULT_CONFIG  # noqa: E402
from portfolio_runtime.errors import InputDataError  # noqa: E402
from portfolio_runtime.rolling import ROLLING_OUTPUT_FILES, run_rolling_experiment  # noqa: E402


def test_backtest_applies_targets_next_day_and_charges_turnover() -> None:
    targets = pd.DataFrame(
        {
            "date": [20230102, 20230102, 20230103, 20230103],
            "ticker": ["A", "B", "A", "B"],
            "portfolio": "risk_optimized",
            "target_weight": [1.0, 0.0, 0.0, 1.0],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [
                20230102,
                20230102,
                20230103,
                20230103,
                20230104,
                20230104,
            ],
            "ticker": ["A", "B"] * 3,
            "return": [0.50, 0.0, 0.10, 0.0, 0.0, 0.20],
        }
    )
    initial = {"risk_optimized": pd.Series({"A": 0.5, "B": 0.5})}

    daily = backtest_targets(
        targets, returns, transaction_cost_bps=100.0, initial_weights=initial
    )

    assert daily["date"].tolist() == ["20230103", "20230104"]
    assert np.isclose(daily.iloc[0]["gross_return"], 0.10)
    assert np.isclose(daily.iloc[0]["turnover"], 0.5)
    assert np.isclose(daily.iloc[0]["transaction_cost"], 0.005)
    assert np.isclose(daily.iloc[1]["gross_return"], 0.20)
    assert np.isclose(daily.iloc[1]["turnover"], 1.0)
    assert summarize_backtest(daily)["portfolios"]["risk_optimized"]["observations"] == 2


def test_backtest_allows_one_held_asset_to_be_written_off() -> None:
    targets = pd.DataFrame(
        {
            "date": [20230102, 20230102],
            "ticker": ["A", "B"],
            "portfolio": "risk_optimized",
            "target_weight": [0.1, 0.9],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [20230102, 20230102, 20230103, 20230103],
            "ticker": ["A", "B", "A", "B"],
            "return": [0.0, 0.0, -1.0, 0.0],
        }
    )

    daily = backtest_targets(targets, returns, transaction_cost_bps=0.0)

    assert np.isclose(daily.iloc[0]["gross_return"], -0.1)
    assert np.isclose(daily.iloc[0]["net_return"], -0.1)


def test_benchmark_relative_performance_and_metrics() -> None:
    daily = pd.DataFrame(
        {
            "date": ["20230103", "20230103", "20230104", "20230104"],
            "portfolio": ["benchmark", "risk_optimized"] * 2,
            "net_return": [0.01, 0.02, -0.01, 0.00],
            "nav": [1.01, 1.02, 0.9999, 1.02],
            "drawdown": [0.0, 0.0, -0.01, 0.0],
            "turnover": 0.0,
            "transaction_cost": 0.0,
        }
    )

    enriched = add_benchmark_relative_performance(daily)
    optimized = enriched.loc[enriched["portfolio"].eq("risk_optimized")]
    metrics = summarize_backtest(enriched)["portfolios"]["risk_optimized"]

    assert np.allclose(optimized["active_return"], [0.01, 0.01])
    assert np.isclose(float(optimized.iloc[-1]["active_nav"]), 1.02 / 0.9999)
    assert metrics["annualized_excess_return"] > 0
    assert np.isclose(metrics["realized_tracking_error"], 0.0)
    assert metrics["information_ratio"] is None


def test_rolling_experiment_writes_stable_outputs(tmp_path: Path) -> None:
    dates = [20230102, 20230103, 20230104, 20230105, 20230106]
    rebalance_dates = [20230102, 20230104]
    tickers = [f"{value:06d}.SZ" for value in range(1, 5)]
    signal = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "prediction": 4 - index}
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
                "return": 0.001 * (index + 1) * (-1 if position % 2 else 1),
            }
            for position, date in enumerate(dates)
            for index, ticker in enumerate(tickers)
        ]
    )
    signal_path = tmp_path / "signal.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    returns_path = tmp_path / "returns.parquet"
    covariance_path = tmp_path / "covariance.parquet"
    exposure_root = tmp_path / "risk_model"
    config_path = tmp_path / "config.yaml"
    signal.to_parquet(signal_path, index=False)
    benchmark.to_parquet(benchmark_path, index=False)
    returns.to_parquet(returns_path, index=False)
    pd.DataFrame(np.eye(4) * 0.10, index=tickers, columns=tickers).to_parquet(
        covariance_path
    )
    for date in rebalance_dates:
        date_root = exposure_root / f"date={date}"
        date_root.mkdir(parents=True)
        pd.DataFrame(
            {"SIZE": [-1.0, -0.5, 0.5, 1.0]},
            index=pd.Index(tickers, name="ticker"),
        ).to_parquet(date_root / "exposures.parquet")
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "optimizer": {
                    "objective_mode": "score_max_te",
                    "solver_backend": "auto",
                    "risk_aversion": 5.0,
                    "turnover_penalty": 0.001,
                    "smoothing_epsilon": 1.0e-8,
                    "max_iterations": 2000,
                    "ftol": 1.0e-10,
                },
                "constraints": {
                    "max_weight": 0.60,
                    "max_active_weight": 0.40,
                    "max_turnover": None,
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
                    "weight_sum_tolerance": 1.0e-8,
                    "constraint_tolerance": 1.0e-6,
                },
                "baseline": {"top_n": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = run_rolling_experiment(
        config_path=config_path,
        signal_file=signal_path,
        covariance_root=covariance_path,
        exposure_root=exposure_root,
        benchmark_file=benchmark_path,
        asset_returns_file=returns_path,
        end_date="20230104",
        output_dir=output,
        transaction_cost_bps=10.0,
    )

    assert result["status"] == "success"
    assert result["rebalance_count"] == 2
    assert {path.name for path in output.iterdir()} == set(ROLLING_OUTPUT_FILES)
    daily = pd.read_parquet(output / "daily_performance.parquet")
    assert set(daily["portfolio"]) == {
        "benchmark",
        "equal_weight_signal",
        "risk_optimized",
    }
    assert {
        "benchmark_net_return",
        "active_return",
        "active_nav",
        "active_drawdown",
    }.issubset(daily.columns)
    diagnostics = pd.read_parquet(output / "optimization_diagnostics.parquet")
    assert diagnostics["constraints_passed"].all()
    assert diagnostics["objective_mode"].eq("score_max_te").all()
    assert diagnostics["solver_backend"].eq("scipy_highs_outer_inner").all()
    assert "tracking_error_slack" in diagnostics.columns
    metrics = json.loads((output / "portfolio_metrics.json").read_text())
    assert metrics["portfolios"]["risk_optimized"]["realized_tracking_error"] is not None
    manifest = json.loads((output / "rolling_manifest.json").read_text())
    assert "next available trading date" in manifest["execution_timing"]
    assert [item["date"] for item in manifest["exposure_inputs"]] == [
        "20230102",
        "20230104",
    ]
    assert manifest["input_cache"]["enabled"] is True
    assert manifest["input_cache"]["file_load_count"] >= 3
    assert manifest["input_cache"]["cache_hit_count"] > 0


def test_rolling_rejects_exposure_file_and_root_together(tmp_path: Path) -> None:
    with pytest.raises(InputDataError, match="only one"):
        run_rolling_experiment(
            config_path=tmp_path / "config.yaml",
            signal_file=tmp_path / "signal.parquet",
            covariance_root=tmp_path / "risk",
            benchmark_file=tmp_path / "benchmark.parquet",
            asset_returns_file=tmp_path / "returns.parquet",
            exposure_file=tmp_path / "exposures.parquet",
            exposure_root=tmp_path / "risk",
            output_dir=tmp_path / "output",
        )



def test_date_table_cache_reuses_file_and_preserves_date_slices(tmp_path: Path) -> None:
    from portfolio_runtime.io import DateTableCache, load_signal

    path = tmp_path / "signal.parquet"
    pd.DataFrame(
        {
            "date": [20230102, 20230102, 20230103],
            "ticker": ["A", "B", "A"],
            "prediction": [1.0, 2.0, 3.0],
        }
    ).to_parquet(path, index=False)
    cache = DateTableCache()

    first = load_signal(path, "20230102", table_cache=cache)
    second = load_signal(path, "20230103", table_cache=cache)

    assert first.to_dict() == {"A": 1.0, "B": 2.0}
    assert second.to_dict() == {"A": 3.0}
    assert cache.statistics()["file_load_count"] == 1
    assert cache.statistics()["cache_hit_count"] == 1


def test_rolling_stockdemo_feedback_uses_actual_executed_holdings(tmp_path: Path) -> None:
    dates = [20230102, 20230103, 20230104, 20230105]
    rebalance_dates = dates[:-1]
    tickers = ["000001.SZ", "000002.SZ"]
    signal = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "prediction": float(2 - ticker_index + date_index),
            }
            for date_index, date in enumerate(rebalance_dates)
            for ticker_index, ticker in enumerate(tickers)
        ]
    )
    benchmark = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "benchmark_weight": 0.5}
            for date in rebalance_dates
            for ticker in tickers
        ]
    )
    returns = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "return": 0.001 * (index + 1)}
            for date in dates
            for index, ticker in enumerate(tickers)
        ]
    )
    market = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "open": 100.0 + 10.0 * index,
                "close": (100.0 + 10.0 * index) * (1.0 + 0.001 * date_index),
                "pre_close": 100.0 + 10.0 * index,
                "twap": 100.0 + 10.0 * index,
                "is_open": True,
                "is_st": False,
                "adj_factor": 1.0,
            }
            for date_index, date in enumerate(rebalance_dates)
            for index, ticker in enumerate(tickers)
        ]
    )
    signal_path = tmp_path / "signal.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    returns_path = tmp_path / "returns.parquet"
    market_path = tmp_path / "market.parquet"
    covariance_path = tmp_path / "covariance.parquet"
    config_path = tmp_path / "config.yaml"
    signal.to_parquet(signal_path, index=False)
    benchmark.to_parquet(benchmark_path, index=False)
    returns.to_parquet(returns_path, index=False)
    market.to_parquet(market_path, index=False)
    pd.DataFrame(np.eye(2) * 0.10, index=tickers, columns=tickers).to_parquet(
        covariance_path
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"].update(
        {"max_weight": 0.80, "max_active_weight": 0.50}
    )
    config["baseline"]["top_n"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = tmp_path / "execution_output"
    result = run_rolling_experiment(
        config_path=config_path,
        signal_file=signal_path,
        covariance_root=covariance_path,
        benchmark_file=benchmark_path,
        asset_returns_file=returns_path,
        stockdemo_market_file=market_path,
        output_dir=output,
    )

    assert result["stockdemo_compat"]["status"] == "success"
    assert result["stockdemo_compat"]["output_dir"] == str(
        output / "stockdemo_compat"
    )
    feedback = pd.read_parquet(output / "execution_feedback.parquet")
    assert feedback["target_date"].dropna().tolist() == ["20230102", "20230103"]
    diagnostics = pd.read_parquet(output / "optimization_diagnostics.parquet")
    assert diagnostics["current_state_source"].tolist() == [
        "configured_initial_or_theoretical_drift",
        "stockdemo_actual",
        "stockdemo_actual",
    ]
    assert diagnostics["actual_cash_weight"].iloc[1:].notna().all()
    stats = pd.read_csv(output / "stockdemo_compat" / "stats.csv")
    assert stats["unrealized_pnl"].iloc[-1] == pytest.approx(
        feedback["unrealized_pnl"].iloc[-1]
    )
    manifest = json.loads((output / "rolling_manifest.json").read_text())
    assert manifest["execution_feedback"]["enabled"] is True
    assert manifest["execution_feedback"]["summary"]["output_dir"] == str(
        output / "stockdemo_compat"
    )
    stockdemo_summary = json.loads(
        (output / "stockdemo_compat" / "summary.json").read_text()
    )
    assert stockdemo_summary["output_dir"] == str(output / "stockdemo_compat")
