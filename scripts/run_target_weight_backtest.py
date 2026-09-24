#!/usr/bin/env python3
"""Replay supplied target weights with the reusable execution contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_runtime.io import read_table
from portfolio_runtime.stockdemo_compat import load_stockdemo_market, load_terminal_events
from portfolio_runtime.target_weight_execution import (
    TargetWeightExecutionConfig,
    load_frozen_target_weights,
    run_target_weight_execution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a frozen target-weight portfolio.")
    parser.add_argument("--market-file", required=True, type=Path)
    parser.add_argument("--target-weights-file", required=True, type=Path)
    parser.add_argument("--target-portfolio", default="target_weight")
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--twap-file", type=Path)
    parser.add_argument("--terminal-events-file", type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--transaction", type=float, default=1.4)
    parser.add_argument("--initial-cash", type=float, default=100_000_000.0)
    parser.add_argument("--locked-limit", type=float, default=0.095)
    parser.add_argument("--missing-target-policy", choices=("error", "cash"), default="cash")
    parser.add_argument("--missing-held-policy", choices=("error", "carry_forward", "terminal_writeoff"), default="carry_forward")
    parser.add_argument("--include-final-next-execution", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = load_frozen_target_weights(args.target_weights_file, args.target_portfolio)
    targets["date"] = targets["date"].astype(str).str.replace("-", "", regex=False).str[:8]
    targets = targets.loc[targets["date"].between(str(args.start_date), str(args.end_date))].copy()
    execution_end = (
        pd.Timestamp(str(args.end_date)) + timedelta(days=31)
    ).strftime("%Y%m%d")
    market = load_stockdemo_market(
        args.market_file, start_date=args.start_date, end_date=execution_end,
        twap_file=args.twap_file,
    )
    benchmark = None if args.benchmark_file is None else read_table(args.benchmark_file)
    if benchmark is not None and "date" in benchmark:
        benchmark["date"] = benchmark["date"].astype(str).str.replace("-", "", regex=False).str[:8]
    terminal_events = None if args.terminal_events_file is None else load_terminal_events(args.terminal_events_file)
    summary = run_target_weight_execution(
        market=market,
        target_weights=targets,
        benchmark=benchmark,
        output_dir=args.output_dir,
        terminal_events=terminal_events,
        portfolio_name=args.target_portfolio or "target_weight",
        config=TargetWeightExecutionConfig(
            transaction=args.transaction,
            initial_cash=args.initial_cash,
            locked_limit=args.locked_limit,
            missing_target_policy=args.missing_target_policy,
            missing_held_policy=args.missing_held_policy,
            include_final_next_execution=args.include_final_next_execution,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
