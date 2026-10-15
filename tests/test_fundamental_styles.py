from __future__ import annotations

import subprocess
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

from portfolio_runtime.errors import ConfigError  # noqa: E402
from portfolio_runtime.fundamental_styles import (  # noqa: E402
    build_fundamental_styles,
    load_fundamental_panel,
)
from portfolio_runtime.risk_model import load_risk_config  # noqa: E402


def _panel() -> pd.DataFrame:
    rows = []
    for ticker, scale in (("000001.SZ", 1.0), ("000002.SZ", 2.0), ("000003.SZ", 3.0)):
        rows.extend(
            [
                {
                    "symbol": ticker,
                    "available_date": 20230430,
                    "report_year": 2022,
                    "report_quarter": 4,
                    "is_total_revenue": 100.0 * scale,
                    "is_n_income_attr_p": 10.0 * scale,
                    "bs_total_hldr_eqy_exc_min_int": 50.0 * scale,
                    "bs_total_liab": 80.0 * scale,
                    "bs_total_assets": 160.0 * scale,
                    "bs_money_cap": 30.0 * scale,
                    "bs_total_cur_liab": 20.0 * scale,
                },
                {
                    "symbol": ticker,
                    "available_date": 20240430,
                    "report_year": 2023,
                    "report_quarter": 4,
                    "is_total_revenue": 120.0 * scale,
                    "is_n_income_attr_p": 12.0 * scale,
                    "bs_total_hldr_eqy_exc_min_int": 60.0 * scale,
                    "bs_total_liab": 84.0 * scale,
                    "bs_total_assets": 168.0 * scale,
                    "bs_money_cap": 36.0 * scale,
                    "bs_total_cur_liab": 24.0 * scale,
                },
                # A later revision must not enter the historical 2024-05 view.
                {
                    "symbol": ticker,
                    "available_date": 20240701,
                    "report_year": 2023,
                    "report_quarter": 4,
                    "is_total_revenue": 140.0 * scale,
                    "is_n_income_attr_p": 14.0 * scale,
                    "bs_total_hldr_eqy_exc_min_int": 70.0 * scale,
                    "bs_total_liab": 91.0 * scale,
                    "bs_total_assets": 175.0 * scale,
                    "bs_money_cap": 42.0 * scale,
                    "bs_total_cur_liab": 28.0 * scale,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_fundamental_styles_are_point_in_time_and_use_same_period_growth(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fundamentals.parquet"
    _panel().to_parquet(source, index=False)
    panel = load_fundamental_panel(source)
    tickers = pd.Index(["000001.SZ", "000002.SZ", "000003.SZ"])
    caps = pd.Series([200.0, 400.0, 600.0], index=tickers)
    names = ["EARNYILD", "GROWTH", "BTOP", "LEVERAGE", "LIQUIDTY"]

    before_revision = build_fundamental_styles(
        panel,
        exposure_date="20240501",
        tickers=tickers,
        market_cap=caps,
        requested=names,
    )
    after_revision = build_fundamental_styles(
        panel,
        exposure_date="20240702",
        tickers=tickers,
        market_cap=caps,
        requested=names,
    )

    assert before_revision.loc["000001.SZ", "EARNYILD"] == pytest.approx(12.0 / 200.0)
    assert after_revision.loc["000001.SZ", "EARNYILD"] == pytest.approx(14.0 / 200.0)
    assert before_revision.loc["000001.SZ", "GROWTH"] == pytest.approx(0.20)
    assert before_revision.loc["000002.SZ", "BTOP"] == pytest.approx(120.0 / 400.0)
    assert before_revision.loc["000003.SZ", "LEVERAGE"] == pytest.approx(252.0 / 504.0)
    assert before_revision.loc["000001.SZ", "LIQUIDTY"] == pytest.approx(36.0 / 24.0)


def test_fundamental_styles_require_materialized_pit_file(tmp_path: Path) -> None:
    config = tmp_path / "risk.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "style_factors": ["SIZE", "BTOP"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fundamental_file"):
        load_risk_config(config)

    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "style_factors": ["SIZE", "BTOP"],
                "fundamental_file": "/tmp/materialized_fundamentals.parquet",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_risk_config(config)
    assert loaded["style_factors"] == ["SIZE", "BTOP"]
    assert loaded["fundamental_file"].endswith("materialized_fundamentals.parquet")

def test_materialize_pit_fundamentals_merges_base_and_extended(tmp_path: Path) -> None:
    base = tmp_path / "base" / "report_year=2024"
    extended = tmp_path / "extended" / "report_year=2024"
    base.mkdir(parents=True)
    extended.mkdir(parents=True)
    common = {
        "symbol": ["000001.SZ"],
        "available_date": [20250430],
        "report_year": [2024],
        "report_quarter": [4],
        "quarter": ["2024q4"],
        "if_adjusted": [0],
        "is_warmup": [False],
    }
    pd.DataFrame({**common, "is_n_income_attr_p": [12.0]}).to_parquet(
        base / "data.parquet", index=False
    )
    pd.DataFrame({**common, "is_rd_exp": [3.0]}).to_parquet(
        extended / "data.parquet", index=False
    )
    output = tmp_path / "materialized.parquet"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_pit_fundamentals.py"),
            "--base-root", str(base.parent),
            "--extended-root", str(extended.parent),
            "--output-file", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    materialized = pd.read_parquet(output)
    assert materialized.loc[0, "is_n_income_attr_p"] == pytest.approx(12.0)
    assert materialized.loc[0, "is_rd_exp"] == pytest.approx(3.0)



def test_earnyild_annualizes_current_filing_without_prior_pit_rows(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "available_date": [20250430, 20250430, 20250430],
            "report_year": [2025, 2025, 2025],
            "report_quarter": [1, 1, 1],
            "is_n_income_attr_p": [10.0, 20.0, 30.0],
        }
    )
    source = tmp_path / "pit.parquet"
    panel.to_parquet(source, index=False)
    styles = build_fundamental_styles(
        load_fundamental_panel(source),
        exposure_date="20250701",
        tickers=pd.Index(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]),
        market_cap=pd.Series([100.0, 200.0, 300.0, 400.0], index=["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]),
        requested=["EARNYILD"],
    )
    assert styles.loc["000001.SZ", "EARNYILD"] == pytest.approx(0.4)
    assert styles.loc["000002.SZ", "EARNYILD"] == pytest.approx(0.4)
    assert pd.isna(styles.loc["000004.SZ", "EARNYILD"])
