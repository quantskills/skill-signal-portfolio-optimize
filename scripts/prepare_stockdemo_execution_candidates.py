#!/usr/bin/env python3
"""Filter optimization candidates against the next Stockdemo execution date."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_runtime.errors import InputDataError
from portfolio_runtime.io import normalize_date, normalize_ticker, read_table


def _read_market_keys(path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise InputDataError(f"market file does not exist: {source}")
    if source.is_dir():
        files = sorted(source.glob("*.parquet"))
        if not files:
            raise InputDataError(f"market directory has no parquet files: {source}")
        source = files[0]
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(source), format="parquet")
        schema = dataset.schema
        if "date" not in schema.names:
            raise InputDataError("market file must contain date")
        if "symbol" not in schema.names and "ticker" not in schema.names:
            raise InputDataError("market file must contain symbol or ticker")
        columns = ["date"] + [
            name for name in ("symbol", "ticker") if name in schema.names
        ]
        date_type = schema.field("date").type
        if pa.types.is_integer(date_type):
            lower: object = int(start_date)
            upper: object = int(end_date)
        else:
            lower = start_date
            upper = end_date
        table = dataset.to_table(
            filter=(ds.field("date") >= lower) & (ds.field("date") <= upper),
            columns=columns,
        )
    except InputDataError:
        raise
    except Exception as exc:
        raise InputDataError(f"cannot read market keys: {exc}") from exc
    frame = table.to_pandas()
    frame["date"] = frame["date"].map(normalize_date)
    if "symbol" in frame.columns:
        values = frame["symbol"].where(frame["symbol"].notna(), frame.get("ticker"))
    else:
        values = frame["ticker"]
    frame["ticker"] = values.map(normalize_ticker)
    result = frame[["date", "ticker"]].copy()
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError("market contains duplicate date/ticker keys")
    if result.empty:
        raise InputDataError("market has no rows in requested date range")
    return result


def _load_date_ticker(path: str | Path, label: str) -> pd.DataFrame:
    frame = read_table(path).copy()
    required = {"date", "ticker"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputDataError(f"{label} missing column(s): {', '.join(missing)}")
    frame["date"] = frame["date"].map(normalize_date)
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    result = frame[["date", "ticker"]].copy()
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError(f"{label} contains duplicate date/ticker rows")
    return result


def _next_execution_dates(market: pd.DataFrame, source_dates: list[str]) -> dict[str, str]:
    calendar = sorted(market["date"].unique().tolist())
    result: dict[str, str] = {}
    for date in source_dates:
        position = bisect_right(calendar, date)
        if position >= len(calendar):
            raise InputDataError(f"market has no execution date after {date}")
        result[date] = calendar[position]
    return result


def prepare_candidates(
    *,
    candidate_file: str | Path,
    market_file: str | Path,
    output_file: str | Path,
    exclusions_file: str | Path,
    report_file: str | Path,
    benchmark_file: str | Path | None = None,
    signal_file: str | Path | None = None,
    industry_file: str | Path | None = None,
    tradability_file: str | Path | None = None,
) -> dict[str, Any]:
    candidates = _load_date_ticker(candidate_file, "candidate universe")
    start_date = str(candidates["date"].min())
    end_date = str(candidates["date"].max())
    market_end = normalize_date(pd.Timestamp(end_date) + pd.Timedelta(days=10))
    market = _read_market_keys(market_file, start_date, market_end)
    source_dates = sorted(candidates["date"].unique().tolist())
    execution_dates = _next_execution_dates(market, source_dates)
    available = {
        date: set(group["ticker"])
        for date, group in market.groupby("date", sort=False)
    }
    candidates = candidates.copy()
    candidates["execution_date"] = candidates["date"].map(execution_dates)
    candidates["execution_available"] = [
        ticker in available[execution_date]
        for ticker, execution_date in zip(candidates["ticker"], candidates["execution_date"])
    ]
    excluded = candidates.loc[~candidates["execution_available"]].copy()
    excluded["reason"] = "missing_next_execution_market_row"
    filtered = candidates.loc[candidates["execution_available"], ["date", "ticker"]]

    if filtered.empty:
        raise InputDataError("no candidates remain after execution-date filtering")
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(destination, index=False)
    exclusions_destination = Path(exclusions_file).expanduser().resolve()
    exclusions_destination.parent.mkdir(parents=True, exist_ok=True)
    excluded[["date", "ticker", "execution_date", "reason"]].to_parquet(
        exclusions_destination, index=False
    )

    report: dict[str, Any] = {
        "status": "success",
        "policy": "exclude_candidates_without_next_execution_market_row",
        "candidate_file": str(Path(candidate_file).expanduser().resolve()),
        "market_file": str(Path(market_file).expanduser().resolve()),
        "output_file": str(destination),
        "exclusions_file": str(exclusions_destination),
        "source_date_start": start_date,
        "source_date_end": end_date,
        "market_date_start": str(market["date"].min()),
        "market_date_end": str(market["date"].max()),
        "source_date_count": len(source_dates),
        "market_date_count": int(market["date"].nunique()),
        "candidate_rows_before": int(len(candidates)),
        "candidate_rows_after": int(len(filtered)),
        "excluded_rows": int(len(excluded)),
        "excluded_unique_tickers": sorted(excluded["ticker"].unique().tolist()),
        "excluded_sample": excluded[
            ["date", "ticker", "execution_date", "reason"]
        ].to_dict("records"),
        "market_duplicate_keys": int(market.duplicated(["date", "ticker"]).sum()),
    }

    if signal_file is not None:
        signal = _load_date_ticker(signal_file, "signal")
        signal_keys = pd.MultiIndex.from_frame(signal[["date", "ticker"]])
        candidate_keys = pd.MultiIndex.from_frame(candidates[["date", "ticker"]])
        report["signal_missing_candidate_rows"] = int(
            (~candidate_keys.isin(signal_keys)).sum()
        )
    if benchmark_file is not None:
        benchmark = read_table(benchmark_file).copy()
        if not {"date", "ticker", "benchmark_weight"}.issubset(benchmark.columns):
            raise InputDataError(
                "benchmark must contain date, ticker, benchmark_weight"
            )
        benchmark["date"] = benchmark["date"].map(normalize_date)
        benchmark["ticker"] = benchmark["ticker"].map(normalize_ticker)
        benchmark = benchmark.loc[benchmark["benchmark_weight"].astype(float) > 1e-12]
        missing_benchmark = []
        for date, group in benchmark.groupby("date", sort=True):
            if date not in execution_dates:
                continue
            execution_date = execution_dates[date]
            for ticker in group["ticker"]:
                if ticker not in available[execution_date]:
                    missing_benchmark.append(
                        {"date": date, "execution_date": execution_date, "ticker": ticker}
                    )
        report["positive_benchmark_missing_next_execution_rows"] = len(missing_benchmark)
        report["positive_benchmark_missing_next_execution_sample"] = missing_benchmark[:20]
    for label, path in (("industry", industry_file), ("tradability", tradability_file)):
        if path is None:
            continue
        table = _load_date_ticker(path, label)
        table_keys = pd.MultiIndex.from_frame(table[["date", "ticker"]])
        filtered_keys = pd.MultiIndex.from_frame(filtered[["date", "ticker"]])
        report[f"{label}_missing_filtered_candidate_rows"] = int(
            (~filtered_keys.isin(table_keys)).sum()
        )
    report_destination = Path(report_file).expanduser().resolve()
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare candidates with next-execution Stockdemo market coverage."
    )
    parser.add_argument("--candidate-file", required=True, type=Path)
    parser.add_argument("--market-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--exclusions-file", required=True, type=Path)
    parser.add_argument("--report-file", required=True, type=Path)
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--signal-file", type=Path)
    parser.add_argument("--industry-file", type=Path)
    parser.add_argument("--tradability-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = prepare_candidates(
            candidate_file=args.candidate_file,
            market_file=args.market_file,
            output_file=args.output_file,
            exclusions_file=args.exclusions_file,
            report_file=args.report_file,
            benchmark_file=args.benchmark_file,
            signal_file=args.signal_file,
            industry_file=args.industry_file,
            tradability_file=args.tradability_file,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
