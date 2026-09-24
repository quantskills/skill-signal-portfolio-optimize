from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_score_to_alpha",
    ROOT / "scripts/prepare_score_to_alpha.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ic_estimation_ignores_prices_after_calibration_end() -> None:
    dates = pd.bdate_range("2025-01-02", periods=12).strftime("%Y%m%d")
    prices = pd.DataFrame(
        {
            "000001": np.linspace(10.0, 13.0, len(dates)),
            "000002": np.linspace(10.0, 12.0, len(dates)),
            "000003": np.linspace(10.0, 11.0, len(dates)),
        },
        index=dates,
    )
    signal = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "code": ticker,
                "prediction": score,
            }
            for date in dates[:5]
            for ticker, score in zip(
                ["000001", "000002", "000003"], [3.0, 2.0, 1.0]
            )
        ]
    )
    kwargs = {
        "calibration_start": dates[0],
        "calibration_end": dates[7],
        "entry_lag_days": 1,
        "return_horizon_days": 2,
        "minimum_cross_section": 3,
    }
    first = MODULE.estimate_daily_ic(signal, prices, **kwargs)
    changed = prices.copy()
    changed.loc[changed.index > dates[7], "000001"] = 1_000_000.0
    second = MODULE.estimate_daily_ic(signal, changed, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert not first.empty
    assert first["label_maturity_date"].max() <= dates[7]


def test_expected_returns_cover_candidates_and_use_specific_risk(
    tmp_path: Path,
) -> None:
    specific_path = tmp_path / "specific_var.parquet"
    pd.Series(
        [0.04, 0.09, 0.16],
        index=pd.Index(["A", "B", "C"], name="ticker"),
        name="specific_var",
    ).to_frame().to_parquet(specific_path)
    signal = pd.DataFrame(
        {
            "date": ["20260105"] * 3,
            "ticker": ["A", "B", "C"],
            "prediction": [3.0, 2.0, 1.0],
        }
    )
    candidates = signal.loc[:, ["date", "ticker"]]
    manifest = {
        "risk_resolutions": [
            {
                "date": "20260105",
                "specific_variance_file": str(specific_path),
            }
        ]
    }
    result, diagnostics = MODULE.materialize_expected_returns(
        signal,
        candidates,
        manifest,
        application_start="20260105",
        application_end="20260105",
        effective_ic=0.05,
        maximum_absolute_alpha=0.20,
    )
    assert set(result["ticker"]) == {"A", "B", "C"}
    assert np.isfinite(result["prediction"]).all()
    assert diagnostics["candidate_missing_rows"] == 0
    assert diagnostics["risk_files"] == 1
