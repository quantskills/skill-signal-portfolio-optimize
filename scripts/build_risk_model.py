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
from portfolio_runtime.risk_model import build_risk_model_from_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate an open structural multifactor asset covariance."
    )
    parser.add_argument("--config", required=True, help="Risk-model YAML config")
    parser.add_argument("--returns-file", required=True, help="Date x ticker return panel")
    parser.add_argument("--market-cap-file", required=True, help="Date x ticker market-cap panel")
    parser.add_argument(
        "--industry-file",
        help="Optional interval industry history; required only by industry_mode=required",
    )
    parser.add_argument(
        "--universe-file",
        required=True,
        help="Long-form date-ticker file defining target assets",
    )
    parser.add_argument("--date", required=True, help="Risk model as-of date")
    parser.add_argument("--output-dir", required=True, help="New or empty output directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_risk_model_from_files(
            config_path=args.config,
            returns_file=args.returns_file,
            market_cap_file=args.market_cap_file,
            industry_file=args.industry_file,
            universe_file=args.universe_file,
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
