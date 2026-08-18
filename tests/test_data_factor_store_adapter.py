from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_data_factor_store_inputs import OUTPUT_FILES, prepare  # noqa: E402


def test_prepare_data_factor_store_inputs_uses_point_in_time_return_formula(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    equity_path = (
        store
        / "canonical/equity_daily/schema=v1/year=2023/month=01/data.parquet"
    )
    weight_path = (
        store
        / "canonical/index_weights/symbol=000852.SH/year=2023/data.parquet"
    )
    equity_path.parent.mkdir(parents=True)
    weight_path.parent.mkdir(parents=True)
    tickers = ["000001.SZ", "000002.SZ"]
    pd.DataFrame(
        {
            "date": [20230102, 20230102, 20230103, 20230103],
            "symbol": tickers * 2,
            "close": [10.0, 20.0, 11.0, 18.0],
            "pre_close": [10.0, 20.0, 10.0, 20.0],
            "market_value": [100.0, 200.0, 110.0, 180.0],
            "volume": [1.0, 1.0, 1.0, 0.0],
            "amount": [10.0, 20.0, 11.0, 0.0],
            "trade_status": [1, 1, 1, 1],
        }
    ).to_parquet(equity_path, index=False)
    pd.DataFrame(
        {
            "date": [20230102, 20230102, 20230103, 20230103],
            "stock_symbol": tickers * 2,
            "weight": [0.5, 0.5, 0.5, 0.5],
        }
    ).to_parquet(weight_path, index=False)
    output = tmp_path / "output"
    args = argparse.Namespace(
        store_root=store,
        start_date="20230102",
        end_date="20230103",
        risk_history_start=None,
        index_symbol="000852.SH",
        output_dir=output,
    )

    result = prepare(args)

    assert result["status"] == "success"
    assert {path.name for path in output.iterdir()} == set(OUTPUT_FILES)
    returns = pd.read_parquet(output / "asset_returns.parquet")
    selected = returns.set_index(["date", "ticker"])["return"]
    assert np.isclose(selected.loc[(20230103, "000001.SZ")], 0.10)
    assert np.isclose(selected.loc[(20230103, "000002.SZ")], -0.10)
    tradable = pd.read_parquet(output / "tradability.parquet")
    row = tradable.loc[
        tradable["date"].eq(20230103) & tradable["ticker"].eq("000002.SZ")
    ].iloc[0]
    assert not bool(row["tradable"])
