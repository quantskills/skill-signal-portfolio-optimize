from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from portfolio_runtime.stockdemo_compat import load_stockdemo_market
from portfolio_runtime.target_weight_execution import (
    TargetWeightExecutionConfig,
    run_target_weight_execution,
)


def test_target_weight_execution_writes_standard_accounting(tmp_path: Path) -> None:
    market = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "open": 100.0, "close": 100.0, "pre_close": 100.0, "twap": 100.0, "is_open": True, "is_st": False, "adj_factor": 1.0}
            for date in (20230102, 20230103, 20230104)
            for ticker in ("000001.SZ", "000002.SZ")
        ]
    )
    market_path = tmp_path / "market.parquet"
    market.to_parquet(market_path, index=False)
    loaded = load_stockdemo_market(market_path, start_date=20230102, end_date=20230104)
    targets = pd.DataFrame(
        {
            "date": [20230102, 20230102, 20230103, 20230103],
            "ticker": ["000001.SZ", "000002.SZ"] * 2,
            "target_weight": [0.6, 0.4, 0.4, 0.6],
        }
    )
    output = tmp_path / "result"
    summary = run_target_weight_execution(
        market=loaded, target_weights=targets, output_dir=output,
        config=TargetWeightExecutionConfig(initial_cash=1_000_000.0),
    )
    assert summary["engine"] == "stockdemo_compat"
    assert (output / "stats.csv").exists()
    assert (output / "holdings.csv").exists()
    assert (output / "transaction.csv").exists()
