#!/usr/bin/env python3
"""Run stockdemo-compatible signal and/or target-weight backtests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.errors import PortfolioOptimizeError  # noqa: E402
from portfolio_runtime.stockdemo_compat import (  # noqa: E402
    StockDemoExecutionConfig,
    load_stockdemo_market,
    load_stockdemo_signal,
    load_target_weights,
    run_stockdemo_compat,
)
from portfolio_runtime.io import read_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay frozen signals/target weights with stockdemo execution rules."
    )
    parser.add_argument("--market-file", required=True, type=Path)
    parser.add_argument("--signal-file", type=Path)
    parser.add_argument("--target-weights-file", type=Path)
    parser.add_argument("--target-portfolio", default="risk_optimized")
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--portfolio", choices=("signal", "target", "both"), default="both")
    parser.add_argument("--longx", type=int, default=200)
    parser.add_argument("--keep", type=float, default=0.8)
    parser.add_argument("--transaction", type=float, default=1.4)
    parser.add_argument("--initial-cash", type=float, default=100_000_000.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _load_benchmark(path: Path | None, start: str, end: str) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = read_table(path)
    if "date" not in frame.columns:
        if "tradeDate" in frame.columns:
            frame = frame.rename(columns={"tradeDate": "date"})
        else:
            raise ValueError("benchmark must contain date or tradeDate")
    frame["date"] = frame["date"].astype(str).str.replace("-", "", regex=False).str[:8]
    frame = frame.loc[frame["date"].between(start, end)].copy()
    if "close" not in frame.columns and "closeIndex" in frame.columns:
        frame = frame.rename(columns={"closeIndex": "close"})
    if "benchmark" not in frame.columns and "close" not in frame.columns:
        raise ValueError("benchmark must contain close, closeIndex, or benchmark")
    return frame


def _clip_dates(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep only source observations that can execute in the requested window."""

    return frame.loc[frame["date"].astype(str).between(start, end)].copy()


def main() -> int:
    args = parse_args()
    source_start = str(args.start_date)
    source_end = str(args.end_date)
    # A source observed on the requested end date executes on the next market
    # date. Load a small calendar buffer so that final execution is retained.
    execution_end = (pd.Timestamp(source_end) + pd.Timedelta(days=31)).strftime("%Y%m%d")
    config = StockDemoExecutionConfig(
        longx=args.longx,
        keep=args.keep,
        transaction=args.transaction,
        initial_cash=args.initial_cash,
    )
    try:
        market = load_stockdemo_market(
            args.market_file, start_date=source_start, end_date=execution_end
        )
        benchmark = _load_benchmark(args.benchmark_file, source_start, execution_end)
        output = args.output_dir.expanduser().resolve()
        summaries: dict[str, object] = {}
        if args.portfolio in {"signal", "both"}:
            if args.signal_file is None:
                raise ValueError("--signal-file is required for portfolio=signal/both")
            signal = load_stockdemo_signal(args.signal_file)
            signal = _clip_dates(signal, source_start, source_end)
            if signal.empty:
                raise ValueError("signal has no rows in the requested date range")
            summaries["signal"] = run_stockdemo_compat(
                market=market,
                signal=signal,
                benchmark=benchmark,
                output_dir=output / "signal_baseline" if args.portfolio == "both" else output,
                config=config,
                portfolio_name="signal_baseline",
            )
        if args.portfolio in {"target", "both"}:
            if args.target_weights_file is None:
                raise ValueError("--target-weights-file is required for portfolio=target/both")
            targets = load_target_weights(args.target_weights_file, args.target_portfolio)
            targets = _clip_dates(targets, source_start, source_end)
            if targets.empty:
                raise ValueError("target weights have no rows in the requested date range")
            summaries["target"] = run_stockdemo_compat(
                market=market,
                targets=targets,
                benchmark=benchmark,
                output_dir=output / "risk_optimized" if args.portfolio == "both" else output,
                config=config,
                portfolio_name=args.target_portfolio,
            )
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps({"status": "success", "engine": "stockdemo_compat", "results": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"status": "success", "output_dir": str(output), "results": summaries}, ensure_ascii=False, sort_keys=True))
        return 0
    except (PortfolioOptimizeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
