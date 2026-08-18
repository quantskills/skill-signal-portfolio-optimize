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

from portfolio_runtime.dynamic_risk import DynamicRiskModelCache  # noqa: E402
from portfolio_runtime.errors import InputDataError  # noqa: E402
from portfolio_runtime.pipeline import build_optimization_universe  # noqa: E402
from portfolio_runtime.rolling import run_rolling_experiment  # noqa: E402


def _risk_inputs(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2022-01-03", periods=100).strftime("%Y%m%d")
    tickers = pd.Index([f"{value:06d}.SZ" for value in range(1, 13)], name="ticker")
    cap = pd.DataFrame(
        np.vstack(
            [
                np.exp(np.linspace(18.0, 22.0, len(tickers))) * np.exp(0.001 * step)
                for step in range(len(dates))
            ]
        ),
        index=dates,
        columns=tickers,
    )
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    returns.iloc[0] = np.nan
    returns_path = root / "risk_returns.parquet"
    cap_path = root / "risk_cap.parquet"
    config_path = root / "risk.yaml"
    returns.to_parquet(returns_path)
    cap.to_parquet(cap_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "industry_mode": "disabled",
                "style_factors": ["SIZE"],
                "style_lookback_days": 50,
                "style_minimum_observations": 20,
                "momentum_skip_days": 5,
                "lookback_days": 80,
                "minimum_regression_periods": 60,
                "minimum_cross_section_assets": 10,
                "minimum_specific_observations": 30,
                "max_specific_imputation_fraction": 0.30,
            }
        ),
        encoding="utf-8",
    )
    return {
        "returns": returns_path,
        "cap": cap_path,
        "config": config_path,
        "date": Path(str(dates[-1])),
        "tickers": Path("unused"),
    }


def _provider(root: Path, inputs: dict[str, Path]) -> DynamicRiskModelCache:
    return DynamicRiskModelCache(
        config_path=inputs["config"],
        returns_file=inputs["returns"],
        market_cap_file=inputs["cap"],
        industry_file=None,
        cache_root=root / "dynamic",
    )


def test_optimization_universe_includes_positive_carry_holding() -> None:
    signal = pd.Series([1.0], index=["A"])
    benchmark = pd.Series([1.0, 0.0], index=["B", "C"])
    current = pd.Series([0.10, 0.0], index=["C", "D"])
    universe = build_optimization_universe(signal, benchmark, current, 1.0e-8)
    assert universe.tolist() == ["A", "B", "C"]


def test_dynamic_cache_reuses_complete_static_risk(tmp_path: Path) -> None:
    inputs = _risk_inputs(tmp_path)
    provider = _provider(tmp_path, inputs)
    universe = pd.Index([f"{value:06d}.SZ" for value in range(1, 9)])
    static = tmp_path / "static"
    static.mkdir()
    pd.DataFrame(np.eye(8), index=universe, columns=universe).to_parquet(
        static / "asset_cov.parquet"
    )
    pd.DataFrame({"SIZE": np.linspace(-1.0, 1.0, 8)}, index=universe).to_parquet(
        static / "exposures.parquet"
    )

    resolved = provider.resolve(
        date=str(inputs["date"]),
        universe=universe,
        static_covariance_file=static / "asset_cov.parquet",
        static_exposure_file=static / "exposures.parquet",
    )

    assert resolved.source == "static_reused"
    assert provider.statistics()["static_reused_count"] == 1
    assert not any((tmp_path / "dynamic").iterdir())


def test_dynamic_cache_builds_missing_assets_and_invalidates_by_universe_and_config(
    tmp_path: Path,
) -> None:
    inputs = _risk_inputs(tmp_path)
    provider = _provider(tmp_path, inputs)
    first_universe = pd.Index([f"{value:06d}.SZ" for value in range(1, 9)])
    incomplete_static = tmp_path / "static"
    incomplete_static.mkdir()
    static_names = first_universe[:-1]
    pd.DataFrame(
        np.eye(len(static_names)), index=static_names, columns=static_names
    ).to_parquet(incomplete_static / "asset_cov.parquet")
    pd.DataFrame({"SIZE": 0.0}, index=static_names).to_parquet(
        incomplete_static / "exposures.parquet"
    )

    first = provider.resolve(
        date=str(inputs["date"]),
        universe=first_universe,
        static_covariance_file=incomplete_static / "asset_cov.parquet",
        static_exposure_file=incomplete_static / "exposures.parquet",
    )
    reused = provider.resolve(
        date=str(inputs["date"]),
        universe=first_universe,
        static_covariance_file=None,
        static_exposure_file=None,
    )
    second = provider.resolve(
        date=str(inputs["date"]),
        universe=first_universe.append(pd.Index(["000009.SZ"])),
        static_covariance_file=None,
        static_exposure_file=None,
    )
    config = yaml.safe_load(inputs["config"].read_text(encoding="utf-8"))
    config["factor_covariance_halflife"] = 31.0
    inputs["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    changed_provider = _provider(tmp_path, inputs)
    changed = changed_provider.resolve(
        date=str(inputs["date"]),
        universe=first_universe,
        static_covariance_file=None,
        static_exposure_file=None,
    )

    assert first.source == "dynamic_built"
    assert reused.source == "dynamic_reused"
    assert second.cache_directory != first.cache_directory
    assert changed.cache_directory != first.cache_directory
    assert pd.read_parquet(first.covariance_file).shape == (8, 8)


def test_positive_current_holding_missing_risk_data_fails(tmp_path: Path) -> None:
    inputs = _risk_inputs(tmp_path)
    provider = _provider(tmp_path, inputs)
    current = pd.Series([0.9, 0.1], index=["000001.SZ", "MISSING"])
    with pytest.raises(InputDataError, match="positive current holding"):
        provider.validate_positive_current_holdings(str(inputs["date"]), current, 1.0e-8)


def test_partial_dynamic_arguments_fail_before_input_loading(tmp_path: Path) -> None:
    with pytest.raises(InputDataError, match="must be supplied together"):
        run_rolling_experiment(
            config_path=tmp_path / "missing.yaml",
            signal_file=tmp_path / "missing.parquet",
            benchmark_file=tmp_path / "missing.parquet",
            asset_returns_file=tmp_path / "missing.parquet",
            output_dir=tmp_path / "output",
            risk_model_config=tmp_path / "risk.yaml",
        )


def _write_static_rolling_fixture(root: Path) -> dict[str, Path]:
    dates = [20230102, 20230103, 20230104, 20230105, 20230106]
    rebalance_dates = [20230102, 20230104]
    tickers = [f"{value:06d}.SZ" for value in range(1, 5)]
    signal = root / "signal.parquet"
    candidates = root / "candidates.parquet"
    benchmark = root / "benchmark.parquet"
    returns = root / "returns.parquet"
    risk = root / "risk"
    config = root / "portfolio.yaml"
    pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "prediction": 4 - position}
            for date in rebalance_dates
            for position, ticker in enumerate(tickers)
        ]
    ).to_parquet(signal, index=False)
    pd.DataFrame(
        [
            {"date": date, "ticker": ticker}
            for date in rebalance_dates
            for ticker in tickers
        ]
    ).to_parquet(candidates, index=False)
    pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "benchmark_weight": 0.25}
            for date in rebalance_dates
            for ticker in tickers
        ]
    ).to_parquet(benchmark, index=False)
    pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "return": 0.001 * (position + 1)}
            for date in dates
            for position, ticker in enumerate(tickers)
        ]
    ).to_parquet(returns, index=False)
    for date in rebalance_dates:
        date_root = risk / f"date={date}"
        date_root.mkdir(parents=True)
        pd.DataFrame(np.eye(4) * 0.1, index=tickers, columns=tickers).to_parquet(
            date_root / "asset_cov.parquet"
        )
        pd.DataFrame(
            {"SIZE": [-1.0, -0.5, 0.5, 1.0]},
            index=pd.Index(tickers, name="ticker"),
        ).to_parquet(
            date_root / "exposures.parquet"
        )
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "optimizer": {"objective_mode": "score_max_te"},
                "constraints": {
                    "max_weight": 0.6,
                    "max_active_weight": 0.4,
                    "max_tracking_error": 0.3,
                    "style_active_ranges": {
                        "SIZE": {"target_active": 0.0, "tolerance": 1.0}
                    },
                },
                "baseline": {"top_n": 2},
            }
        ),
        encoding="utf-8",
    )
    return {
        "signal": signal,
        "candidates": candidates,
        "benchmark": benchmark,
        "returns": returns,
        "risk": risk,
        "config": config,
    }


def test_rolling_checkpoint_resume_reuses_completed_dates(tmp_path: Path) -> None:
    paths = _write_static_rolling_fixture(tmp_path)
    common = {
        "config_path": paths["config"],
        "signal_file": paths["signal"],
        "candidate_file": paths["candidates"],
        "covariance_root": paths["risk"],
        "exposure_root": paths["risk"],
        "benchmark_file": paths["benchmark"],
        "asset_returns_file": paths["returns"],
        "checkpoint_root": tmp_path / "checkpoints",
    }
    run_rolling_experiment(output_dir=tmp_path / "first", **common)
    run_rolling_experiment(output_dir=tmp_path / "second", **common)

    first = pd.read_parquet(tmp_path / "first" / "rebalance_weights.parquet")
    second = pd.read_parquet(tmp_path / "second" / "rebalance_weights.parquet")
    pd.testing.assert_frame_equal(first, second)
    manifest = json.loads(
        (tmp_path / "second" / "rolling_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["checkpoints"]["reused_count"] == 2
    assert manifest["checkpoints"]["built_count"] == 0

    candidate_rows = pd.read_parquet(paths["candidates"])
    candidate_rows.iloc[::-1].to_parquet(paths["candidates"], index=False)
    run_rolling_experiment(output_dir=tmp_path / "third", **common)
    changed_manifest = json.loads(
        (tmp_path / "third" / "rolling_manifest.json").read_text(encoding="utf-8")
    )
    assert changed_manifest["checkpoints"]["reused_count"] == 0
    assert changed_manifest["checkpoints"]["built_count"] == 2
