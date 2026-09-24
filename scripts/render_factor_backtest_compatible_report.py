#!/usr/bin/env python3
"""Render an executed target-weight portfolio with factor-backtest PNL styling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_runtime.factor_backtest_report_adapter import (
    FactorBacktestReportConfig,
    render_factor_backtest_compatible_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-backtest-root", required=True, type=Path)
    parser.add_argument("--stats-file", required=True, type=Path)
    parser.add_argument("--holdings-file", required=True, type=Path)
    parser.add_argument("--transactions-file", required=True, type=Path)
    parser.add_argument("--signal-file", required=True, type=Path)
    parser.add_argument("--twap-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-name", default="zz1000")
    parser.add_argument("--initial-cash", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = render_factor_backtest_compatible_report(
        FactorBacktestReportConfig(
            factor_backtest_root=args.factor_backtest_root,
            stats_file=args.stats_file,
            holdings_file=args.holdings_file,
            transactions_file=args.transactions_file,
            signal_file=args.signal_file,
            twap_file=args.twap_file,
            output_dir=args.output_dir,
            benchmark_name=args.benchmark_name,
            initial_cash=args.initial_cash,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
