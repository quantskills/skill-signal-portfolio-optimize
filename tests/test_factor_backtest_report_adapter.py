from pathlib import Path

import pandas as pd

from scripts.portfolio_runtime.factor_backtest_report_adapter import _prepare_stats


def test_prepare_stats_uses_executed_nav_and_benchmark_nav(tmp_path: Path) -> None:
    path = tmp_path / "stats.csv"
    pd.DataFrame(
        {
            "date": [20250102, 20250103],
            "unrealized_pnl": [99.0, 101.0],
            "nav": [0.99, 1.01],
            "benchmark_nav": [1.0, 1.02],
        }
    ).to_csv(path, index=False)

    stats, benchmark = _prepare_stats(path, initial_cash=100.0)

    assert stats["unrealized_pnl"].tolist() == [99.0, 101.0]
    assert benchmark["benchmark"].tolist() == [100.0, 102.0]
    assert stats.index.strftime("%Y%m%d").tolist() == ["20250102", "20250103"]
