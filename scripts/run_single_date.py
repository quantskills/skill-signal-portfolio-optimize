#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from portfolio_runtime.errors import PortfolioOptimizeError
from portfolio_runtime.pipeline import run_single_date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize one final stock signal for one rebalance date."
    )
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--signal-file", required=True, help="Long-form signal CSV/Parquet")
    parser.add_argument(
        "--candidate-file",
        help="Optional date-ticker candidate universe; defaults to all signal rows",
    )
    parser.add_argument("--covariance-file", required=True, help="Square covariance CSV/Parquet")
    parser.add_argument("--benchmark-file", required=True, help="Long-form benchmark weights")
    parser.add_argument("--current-weights-file", help="Optional current weights")
    parser.add_argument(
        "--industry-file", "--sector-file", dest="sector_file",
        help="Optional industry labels (--sector-file is a compatibility alias)",
    )
    parser.add_argument("--exposure-file", help="Optional style exposures")
    parser.add_argument("--tradability-file", help="Optional tradability flags")
    parser.add_argument("--date", required=True, help="Rebalance date")
    parser.add_argument("--output-dir", required=True, help="New or empty output directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_single_date(
            config_path=args.config,
            signal_file=args.signal_file,
            covariance_file=args.covariance_file,
            candidate_file=args.candidate_file,
            benchmark_file=args.benchmark_file,
            current_weights_file=args.current_weights_file,
            sector_file=args.sector_file,
            exposure_file=args.exposure_file,
            tradability_file=args.tradability_file,
            requested_date=args.date,
            output_dir=args.output_dir,
        )
    except PortfolioOptimizeError as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
