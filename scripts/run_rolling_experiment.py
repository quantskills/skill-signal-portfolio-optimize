#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from portfolio_runtime.errors import PortfolioOptimizeError
from portfolio_runtime.rolling import run_rolling_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rolling signal portfolio optimization and next-day backtest."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--candidate-file",
        type=Path,
        help="Optional date-ticker candidate universe; defaults to all signal rows",
    )
    parser.add_argument("--signal-file", required=True, type=Path)
    parser.add_argument(
        "--covariance-root",
        type=Path,
        help="Static covariance file/root; optional when dynamic risk is configured.",
    )
    parser.add_argument("--benchmark-file", required=True, type=Path)
    parser.add_argument("--asset-returns-file", required=True, type=Path)
    parser.add_argument("--sector-file", type=Path)
    parser.add_argument("--exposure-file", type=Path)
    parser.add_argument(
        "--exposure-root",
        type=Path,
        help="Date-partitioned root containing date=YYYYMMDD/exposures.parquet",
    )
    parser.add_argument("--factor-covariance-root", type=Path)
    parser.add_argument("--specific-variance-root", type=Path)
    parser.add_argument("--tradability-file", type=Path)
    parser.add_argument("--initial-weights-file", type=Path)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--transaction-cost-bps", type=float)
    parser.add_argument("--risk-model-config", type=Path)
    parser.add_argument("--risk-returns-file", type=Path)
    parser.add_argument("--risk-market-cap-file", type=Path)
    parser.add_argument("--risk-industry-file", type=Path)
    parser.add_argument("--dynamic-risk-cache-root", type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Persistent date/signature checkpoints used to resume rolling optimization.",
    )
    parser.add_argument(
        "--stockdemo-market-file",
        type=Path,
        help=(
            "Enable execution-aware rolling state using Stockdemo-compatible "
            "next-day order accounting."
        ),
    )
    parser.add_argument(
        "--stockdemo-twap-file",
        type=Path,
        help="Optional long or legacy wide TWAP/trade_price table for execution feedback.",
    )
    parser.add_argument(
        "--stockdemo-transaction",
        type=float,
        default=1.4,
        help="Stockdemo round-trip transaction setting in per-thousand units.",
    )
    parser.add_argument(
        "--stockdemo-initial-cash", type=float, default=100_000_000.0
    )
    parser.add_argument(
        "--stockdemo-turnover-mode",
        choices=("normal", "flex"),
        default="flex",
        help="Top-N replacement rule used by Stockdemo execution feedback.",
    )
    parser.add_argument(
        "--stockdemo-missing-target-policy",
        choices=("error", "cash"),
        default="error",
        help="Handle positive target tickers missing on execution day: error or leave in cash.",
    )
    parser.add_argument(
        "--stockdemo-missing-held-policy",
        choices=("error", "carry_forward", "terminal_writeoff"),
        default="carry_forward",
        help="Handle held tickers absent from the execution market: fail, carry the last close, or use an explicit terminal write-off.",
    )
    parser.add_argument(
        "--stockdemo-terminal-events-file",
        type=Path,
        help="JSON manifest containing explicit date/ticker terminal_events.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_rolling_experiment(
            config_path=args.config,
            signal_file=args.signal_file,
            candidate_file=args.candidate_file,
            covariance_root=args.covariance_root,
            benchmark_file=args.benchmark_file,
            asset_returns_file=args.asset_returns_file,
            sector_file=args.sector_file,
            exposure_file=args.exposure_file,
            exposure_root=args.exposure_root,
            factor_covariance_root=args.factor_covariance_root,
            specific_variance_root=args.specific_variance_root,
            tradability_file=args.tradability_file,
            initial_weights_file=args.initial_weights_file,
            start_date=args.start_date,
            end_date=args.end_date,
            rebalance_every=args.rebalance_every,
            transaction_cost_bps=args.transaction_cost_bps,
            risk_model_config=args.risk_model_config,
            risk_returns_file=args.risk_returns_file,
            risk_market_cap_file=args.risk_market_cap_file,
            risk_industry_file=args.risk_industry_file,
            dynamic_risk_cache_root=args.dynamic_risk_cache_root,
            checkpoint_root=args.checkpoint_root,
            stockdemo_market_file=args.stockdemo_market_file,
            stockdemo_twap_file=args.stockdemo_twap_file,
            stockdemo_transaction=args.stockdemo_transaction,
            stockdemo_initial_cash=args.stockdemo_initial_cash,
            stockdemo_turnover_mode=args.stockdemo_turnover_mode,
            stockdemo_missing_target_policy=args.stockdemo_missing_target_policy,
            stockdemo_missing_held_policy=args.stockdemo_missing_held_policy,
            stockdemo_terminal_events_file=args.stockdemo_terminal_events_file,
            output_dir=args.output_dir,
        )
    except PortfolioOptimizeError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
