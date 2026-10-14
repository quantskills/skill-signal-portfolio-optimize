#!/usr/bin/env python3
"""Render style and industry active-exposure charts from a rolling result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_runtime.exposure_report import ExposureReportConfig, render_exposure_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--industry-frequency", choices=("rebalance", "weekly", "monthly"), default="monthly")
    parser.add_argument("--title-prefix", default="Portfolio exposure")
    parser.add_argument("--portfolio", default="risk_optimized")
    args = parser.parse_args()
    report = render_exposure_report(
        ExposureReportConfig(
            exposure_file=args.exposure_file,
            output_dir=args.output_dir,
            industry_frequency=args.industry_frequency,
            title_prefix=args.title_prefix,
            portfolio=args.portfolio,
        )
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
