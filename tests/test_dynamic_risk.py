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
from portfolio_runtime.io import DateTableCache  # noqa: E402
from portfolio_runtime.pipeline import build_optimization_universe  # noqa: E402
from portfolio_runtime.rolling import (  # noqa: E402
    _initialize_monthly_model_universe,
    _period_model_universes,
    _risk_model_date_for,
    _preflight_monthly_risk_coverage,
    run_rolling_experiment,
)


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


def test_monthly_dynamic_cache_reuses_one_model_date_for_daily_universes(
    tmp_path: Path,
) -> None:
    inputs = _risk_inputs(tmp_path)
    provider = _provider(tmp_path, inputs)
    model_date = str(inputs["date"])
    model_universe = pd.Index(
        [f"{value:06d}.SZ" for value in range(1, 10)], name="ticker"
    )

    first = provider.resolve(
        date=model_date,
        universe=model_universe[:8],
        model_date=model_date,
        model_universe=model_universe,
        static_covariance_file=None,
        static_exposure_file=None,
    )
    reused = provider.resolve(
        date="20991231",
        universe=model_universe[1:],
        model_date=model_date,
        model_universe=model_universe,
        static_covariance_file=None,
        static_exposure_file=None,
    )

    assert first.source == "dynamic_built"
    assert reused.source == "dynamic_reused"
    assert reused.cache_directory == first.cache_directory
    assert provider.statistics()["dynamic_built_count"] == 1
    assert provider.statistics()["dynamic_reused_count"] == 1
    manifest = json.loads(
        (first.cache_directory / "risk_model_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["requested_date"] == model_date
    assert manifest["model_universe_asset_count"] == len(model_universe)


def test_available_model_universe_uses_latest_valid_market_cap(tmp_path: Path) -> None:
    inputs = _risk_inputs(tmp_path)
    cap = pd.read_parquet(inputs["cap"])
    suspended = cap.columns[0]
    cap.loc[cap.index[-1], suspended] = np.nan
    cap.to_parquet(inputs["cap"])
    provider = _provider(tmp_path, inputs)

    available = provider.available_model_universe(
        str(inputs["date"]), pd.Index(cap.columns[:8], name="ticker")
    )

    assert suspended in available


def test_monthly_model_universe_includes_carried_holding(tmp_path: Path) -> None:
    inputs = _risk_inputs(tmp_path)
    provider = _provider(tmp_path, inputs)
    tickers = pd.read_parquet(inputs["cap"]).columns
    planned = pd.Index(tickers[:8], name="ticker")
    carried = tickers[8]
    benchmark = pd.Series([1.0], index=[tickers[0]])
    current = pd.Series([0.9, 0.1], index=[tickers[0], carried])

    model_universe = _initialize_monthly_model_universe(
        provider,
        model_date=str(inputs["date"]),
        planned_universe=planned,
        benchmark=benchmark,
        current=current,
        tolerance=1.0e-8,
    )

    assert carried in model_universe


def test_monthly_dynamic_cache_rejects_uncovered_daily_universe(
    tmp_path: Path,
) -> None:
    inputs = _risk_inputs(tmp_path)
    provider = _provider(tmp_path, inputs)
    with pytest.raises(InputDataError, match="does not cover"):
        provider.resolve(
            date=str(inputs["date"]),
            universe=pd.Index(["000001.SZ", "000010.SZ"]),
            model_date=str(inputs["date"]),
            model_universe=pd.Index(["000001.SZ"]),
            static_covariance_file=None,
            static_exposure_file=None,
        )


def test_risk_coverage_explains_missing_industry_and_preflight_fails(
    tmp_path: Path,
) -> None:
    inputs = _risk_inputs(tmp_path)
    config = yaml.safe_load(inputs["config"].read_text(encoding="utf-8"))
    config["industry_mode"] = "required"
    inputs["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    tickers = pd.read_parquet(inputs["cap"]).columns
    missing = tickers[0]
    industry_path = tmp_path / "industry.parquet"
    pd.DataFrame(
        {
            "stock_symbol": tickers[1:],
            "l1_code": ["801010"] * (len(tickers) - 1),
            "in_date": ["20200101"] * (len(tickers) - 1),
            "out_date": [None] * (len(tickers) - 1),
        }
    ).to_parquet(industry_path, index=False)
    provider = DynamicRiskModelCache(
        config_path=inputs["config"],
        returns_file=inputs["returns"],
        market_cap_file=inputs["cap"],
        industry_file=industry_path,
        cache_root=tmp_path / "dynamic-required",
    )
    universe = pd.Index(tickers[:8], name="ticker")
    model_date = str(inputs["date"])
    coverage = provider.model_universe_coverage(model_date, universe)

    assert not bool(coverage.loc[missing, "available"])
    assert coverage.loc[missing, "missing_reasons"] == "missing_asof_industry"

    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        {
            "date": [model_date] * len(universe),
            "ticker": universe,
            "benchmark_weight": [1.0 / len(universe)] * len(universe),
        }
    ).to_parquet(benchmark_path, index=False)
    with pytest.raises(
        InputDataError,
        match=rf"{missing}\(missing_asof_industry\)",
    ):
        _preflight_monthly_risk_coverage(
            provider,
            planned_universes={model_date[:6]: universe},
            benchmark_file=benchmark_path,
            dates=[model_date],
            tolerance=1.0e-8,
            table_cache=DateTableCache(),
        )


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

def test_weekly_model_date_uses_first_selected_date_in_iso_week() -> None:
    dates = [
        "20251229",
        "20251230",
        "20260105",
        "20260106",
    ]

    assert _risk_model_date_for(
        "20251230", dates, "weekly"
    ) == "20251229"
    assert _risk_model_date_for(
        "20260106", dates, "weekly"
    ) == "20260105"


def test_weekly_model_universe_groups_candidates_and_benchmark(
    tmp_path: Path,
) -> None:
    dates = [
        "20230102",
        "20230103",
        "20230109",
    ]
    signal = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "prediction": [1.0, 2.0, 3.0],
        }
    )
    candidates = signal[["date", "ticker"]]
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["600001.SH", "600002.SH", "600003.SH"],
            "benchmark_weight": [1.0, 1.0, 1.0],
        }
    )
    signal_path = tmp_path / "signal.parquet"
    candidate_path = tmp_path / "candidate.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    signal.to_parquet(signal_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    benchmark.to_parquet(benchmark_path, index=False)

    universes = _period_model_universes(
        signal_path,
        candidate_path,
        benchmark_path,
        dates,
        DateTableCache(),
        frequency="weekly",
    )

    assert sorted(universes) == ["2023W01", "2023W02"]
    assert set(universes["2023W01"]) == {
        "000001.SZ",
        "000002.SZ",
        "600001.SH",
        "600002.SH",
    }
    assert set(universes["2023W02"]) == {
        "000003.SZ",
        "600003.SH",
    }
