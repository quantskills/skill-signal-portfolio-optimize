#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from portfolio_runtime.errors import InputDataError
from portfolio_runtime.io import normalize_date, normalize_ticker, read_table, sha256_file


OUTPUT_FILES = (
    "signal.parquet",
    "signal_full.parquet",
    "signal_candidates.parquet",
    "optimizer_universe.parquet",
    "benchmark_weights.parquet",
    "rebalance_dates.parquet",
    "input_manifest.json",
)


def _quoted_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise InputDataError("column names must be non-empty text")
    return '"' + value.replace('"', '""') + '"'


def _quoted_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _scan_expression(path: Path) -> str:
    literal = _quoted_literal(str(path))
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return f"read_csv_auto({literal}, header=true)"
    if suffix in {".parquet", ".pq"}:
        return f"read_parquet({literal})"
    raise InputDataError(f"unsupported signal format: {path.suffix}")


def _prepare_output_directory(output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise InputDataError(f"output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise InputDataError(f"output directory must be new or empty: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    return destination, temporary


def _load_selected_signal(
    *,
    signal_path: Path,
    eligibility_path: Path,
    ticker_column: str,
    signal_column: str,
    start_date: str,
    end_date: str,
    rebalance_every: int,
    candidate_count: int,
    higher_is_better: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, dict[str, int]]:
    if rebalance_every <= 0:
        raise InputDataError("rebalance_every must be positive")
    if candidate_count <= 0:
        raise InputDataError("candidate_count must be positive")
    scan = _scan_expression(signal_path)
    eligibility_scan = _scan_expression(eligibility_path)
    ticker = _quoted_identifier(ticker_column)
    score = _quoted_identifier(signal_column)
    connection = duckdb.connect()
    try:
        columns = {
            str(row[0])
            for row in connection.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
        }
        missing = {"date", ticker_column, signal_column} - columns
        if missing:
            raise InputDataError(
                "signal source missing column(s): " + ", ".join(sorted(missing))
            )
        eligibility_columns = {
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {eligibility_scan}"
            ).fetchall()
        }
        missing_eligibility = {"date", "ticker", "tradable"} - eligibility_columns
        if missing_eligibility:
            raise InputDataError(
                "eligibility file missing column(s): "
                + ", ".join(sorted(missing_eligibility))
            )
        all_dates = [
            str(int(row[0]))
            for row in connection.execute(
                f"""
                SELECT DISTINCT CAST(date AS BIGINT) AS date
                FROM {scan}
                WHERE CAST(date AS BIGINT) BETWEEN ? AND ?
                ORDER BY date
                """,
                [int(start_date), int(end_date)],
            ).fetchall()
        ]
        selected_dates = all_dates[::rebalance_every]
        if not selected_dates:
            raise InputDataError("no signal dates remain after date filters")
        selected_frame = pd.DataFrame(
            {"date": [int(value) for value in selected_dates]}
        )
        connection.register("selected_dates", selected_frame)
        eligibility = connection.execute(
            f"""
            SELECT CAST(source.date AS BIGINT) AS date,
                   CAST(source.ticker AS VARCHAR) AS ticker,
                   CAST(source.tradable AS BOOLEAN) AS tradable
            FROM {eligibility_scan} AS source
            INNER JOIN selected_dates USING (date)
            ORDER BY date, ticker
            """
        ).fetchdf()
        if eligibility.duplicated(["date", "ticker"]).any():
            raise InputDataError("eligibility file contains duplicate selected keys")
        if eligibility["tradable"].isna().any():
            raise InputDataError("eligibility file contains invalid tradable values")
        connection.register("selected_eligibility", eligibility)
        quality = connection.execute(
            f"""
            WITH selected AS (
                SELECT CAST(source.date AS BIGINT) AS date,
                       CAST(source.{ticker} AS VARCHAR) AS ticker,
                       TRY_CAST(source.{score} AS DOUBLE) AS prediction
                FROM {scan} AS source
                INNER JOIN selected_dates USING (date)
            )
            SELECT COUNT(*) AS rows,
                   COUNT(*) FILTER (
                       WHERE ticker IS NULL OR TRIM(ticker) = ''
                   ) AS invalid_tickers,
                   COUNT(*) FILTER (
                       WHERE prediction IS NULL OR NOT ISFINITE(prediction)
                   ) AS invalid_predictions,
                   COUNT(*) - COUNT(DISTINCT (date, ticker)) AS duplicate_keys,
                   COUNT(DISTINCT date) AS dates,
                   COUNT(*) FILTER (
                       WHERE EXISTS (
                           SELECT 1 FROM selected_eligibility AS eligibility
                           WHERE eligibility.date = selected.date
                             AND eligibility.ticker = selected.ticker
                             AND eligibility.tradable
                       )
                   ) AS eligible_rows
            FROM selected
            """
        ).fetchone()
        quality_record = {
            "rows": int(quality[0]),
            "invalid_tickers": int(quality[1]),
            "invalid_predictions": int(quality[2]),
            "duplicate_keys": int(quality[3]),
            "dates": int(quality[4]),
            "eligible_rows": int(quality[5]),
        }
        failures = {
            name: value
            for name, value in quality_record.items()
            if name not in {"rows", "dates", "eligible_rows"} and value != 0
        }
        if failures:
            raise InputDataError(f"selected signal rows fail quality checks: {failures}")
        full_signal = connection.execute(
            f"""
            SELECT CAST(source.date AS BIGINT) AS date,
                   CAST(source.{ticker} AS VARCHAR) AS ticker,
                   CAST(source.{score} AS DOUBLE) AS prediction
            FROM {scan} AS source
            INNER JOIN selected_dates USING (date)
            ORDER BY date, ticker
            """
        ).fetchdf()
        direction = "DESC" if higher_is_better else "ASC"
        signal = connection.execute(
            f"""
            WITH selected AS (
                SELECT CAST(source.date AS BIGINT) AS date,
                       CAST(source.{ticker} AS VARCHAR) AS ticker,
                       CAST(source.{score} AS DOUBLE) AS prediction
                FROM {scan} AS source
                INNER JOIN selected_dates USING (date)
                INNER JOIN selected_eligibility AS eligibility
                  ON CAST(source.date AS BIGINT) = eligibility.date
                 AND CAST(source.{ticker} AS VARCHAR) = eligibility.ticker
                 AND eligibility.tradable
            ), ranked AS (
                SELECT date, ticker, prediction,
                       ROW_NUMBER() OVER (
                           PARTITION BY date
                           ORDER BY prediction {direction}, ticker ASC
                       ) AS candidate_rank
                FROM selected
            )
            SELECT date, ticker, prediction
            FROM ranked
            WHERE candidate_rank <= ?
            ORDER BY date, ticker
            """,
            [candidate_count],
        ).fetchdf()
    except InputDataError:
        raise
    except Exception as exc:
        raise InputDataError(f"cannot prepare signal source {signal_path}: {exc}") from exc
    finally:
        connection.close()
    for label, frame in (("full signal", full_signal), ("candidate signal", signal)):
        frame["ticker"] = frame["ticker"].map(normalize_ticker)
        frame["date"] = frame["date"].map(lambda value: int(normalize_date(value)))
        if frame.duplicated(["date", "ticker"]).any():
            raise InputDataError(
                f"ticker normalization creates duplicate {label} keys"
            )
    counts = signal.groupby("date").size().reindex(
        [int(value) for value in selected_dates], fill_value=0
    )
    incomplete = counts.loc[counts.ne(candidate_count)]
    if not incomplete.empty:
        raise InputDataError(
            "signal has fewer candidates than requested on date(s): "
            + str(incomplete.head(10).to_dict())
        )
    eligibility["ticker"] = eligibility["ticker"].map(normalize_ticker)
    eligibility["date"] = eligibility["date"].map(
        lambda value: int(normalize_date(value))
    )
    if eligibility.duplicated(["date", "ticker"]).any():
        raise InputDataError("ticker normalization creates duplicate eligibility keys")
    return full_signal, signal, selected_dates, eligibility, quality_record


def _load_benchmark(
    benchmark_file: str | Path,
    selected_dates: list[str],
    *,
    weight_tolerance: float,
    sum_tolerance: float,
    eligibility: pd.DataFrame,
    exclusion_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = read_table(benchmark_file)
    required = {"date", "ticker", "benchmark_weight"}
    missing = required - set(frame.columns)
    if missing:
        raise InputDataError(
            "benchmark weights missing column(s): " + ", ".join(sorted(missing))
        )
    result = frame.loc[:, ["date", "ticker", "benchmark_weight"]].copy()
    result["date"] = result["date"].map(normalize_date)
    selected = set(selected_dates)
    result = result.loc[result["date"].isin(selected)].copy()
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result["benchmark_weight"] = pd.to_numeric(
        result["benchmark_weight"], errors="coerce"
    )
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError("benchmark weights contain duplicate date-ticker keys")
    if result["benchmark_weight"].isna().any() or not np.isfinite(
        result["benchmark_weight"]
    ).all():
        raise InputDataError("benchmark weights contain missing or non-finite values")
    if result["benchmark_weight"].lt(0.0).any():
        raise InputDataError("benchmark weights contain negative values")
    available_dates = set(result["date"].unique())
    missing_dates = sorted(selected - available_dates)
    if missing_dates:
        raise InputDataError(f"benchmark weights missing selected date(s): {missing_dates}")
    totals = result.groupby("date")["benchmark_weight"].sum()
    deviations = (totals - 1.0).abs()
    invalid_totals = totals.loc[deviations.gt(sum_tolerance)]
    if not invalid_totals.empty:
        raise InputDataError(
            "benchmark weight sums exceed normalization tolerance: "
            + str(invalid_totals.head(10).to_dict())
        )
    result = result.loc[result["benchmark_weight"].gt(weight_tolerance)].copy()
    result["date"] = result["date"].astype(int)
    eligible_keys = eligibility.loc[:, ["date", "ticker"]].drop_duplicates()
    merged = result.merge(
        eligible_keys.assign(market_record_available=True),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    excluded = merged.loc[merged["market_record_available"].isna()].copy()
    excluded_weight = excluded.groupby("date")["benchmark_weight"].sum()
    maximum_excluded_weight = (
        0.0 if excluded_weight.empty else float(excluded_weight.max())
    )
    if maximum_excluded_weight > exclusion_tolerance:
        raise InputDataError(
            "benchmark weight without market records exceeds exclusion tolerance: "
            + str(excluded_weight.loc[excluded_weight.gt(exclusion_tolerance)].to_dict())
        )
    result = merged.loc[merged["market_record_available"].notna(), result.columns].copy()
    positive_totals = result.groupby("date")["benchmark_weight"].transform("sum")
    result["benchmark_weight"] = result["benchmark_weight"] / positive_totals
    summary = {
        "raw_sum_min": float(totals.min()),
        "raw_sum_max": float(totals.max()),
        "raw_max_absolute_deviation": float(deviations.max()),
        "normalization_tolerance": float(sum_tolerance),
        "excluded_without_market_record_count": int(len(excluded)),
        "excluded_without_market_record_weight_max": maximum_excluded_weight,
        "exclusion_tolerance": float(exclusion_tolerance),
    }
    return (
        result.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True),
        summary,
    )


def prepare_inputs(
    *,
    signal_file: str | Path,
    benchmark_file: str | Path,
    eligibility_file: str | Path,
    start_date: object,
    end_date: object,
    rebalance_every: int,
    candidate_count: int,
    output_dir: str | Path,
    ticker_column: str = "symbol",
    signal_column: str = "signal",
    higher_is_better: bool = True,
    weight_tolerance: float = 1.0e-8,
    benchmark_sum_tolerance: float = 1.0e-3,
    benchmark_exclusion_tolerance: float = 5.0e-3,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    start = normalize_date(start_date)
    end = normalize_date(end_date)
    if start > end:
        raise InputDataError("start_date must not exceed end_date")
    if not np.isfinite(weight_tolerance) or weight_tolerance < 0:
        raise InputDataError("weight_tolerance must be finite and non-negative")
    if not np.isfinite(benchmark_sum_tolerance) or benchmark_sum_tolerance < 0:
        raise InputDataError(
            "benchmark_sum_tolerance must be finite and non-negative"
        )
    if (
        not np.isfinite(benchmark_exclusion_tolerance)
        or benchmark_exclusion_tolerance < 0
    ):
        raise InputDataError(
            "benchmark_exclusion_tolerance must be finite and non-negative"
        )
    signal_path = Path(signal_file).expanduser().resolve()
    benchmark_path = Path(benchmark_file).expanduser().resolve()
    eligibility_path = Path(eligibility_file).expanduser().resolve()
    if not signal_path.is_file():
        raise InputDataError(f"input file does not exist: {signal_path}")
    if not benchmark_path.is_file():
        raise InputDataError(f"input file does not exist: {benchmark_path}")
    if not eligibility_path.is_file():
        raise InputDataError(f"input file does not exist: {eligibility_path}")
    full_signal, signal, selected_dates, eligibility, source_quality = (
        _load_selected_signal(
            signal_path=signal_path,
            eligibility_path=eligibility_path,
            ticker_column=ticker_column,
            signal_column=signal_column,
            start_date=start,
            end_date=end,
            rebalance_every=rebalance_every,
            candidate_count=candidate_count,
            higher_is_better=higher_is_better,
        )
    )
    benchmark, benchmark_quality = _load_benchmark(
        benchmark_path,
        selected_dates,
        weight_tolerance=weight_tolerance,
        sum_tolerance=benchmark_sum_tolerance,
        eligibility=eligibility,
        exclusion_tolerance=benchmark_exclusion_tolerance,
    )
    universe = pd.concat(
        [
            signal.loc[:, ["date", "ticker"]],
            benchmark.loc[:, ["date", "ticker"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["date", "ticker"])
    universe = universe.sort_values(["date", "ticker"], kind="stable").reset_index(
        drop=True
    )
    full_signal_counts = full_signal.groupby("date").size()
    signal_counts = signal.groupby("date").size()
    benchmark_counts = benchmark.groupby("date").size()
    universe_counts = universe.groupby("date").size()
    dates = pd.DataFrame(
        {
            "date": [int(value) for value in selected_dates],
            "ordinal": np.arange(1, len(selected_dates) + 1, dtype=int),
            "calibration_count": [
                full_signal_counts[int(value)] for value in selected_dates
            ],
            "signal_count": [signal_counts[int(value)] for value in selected_dates],
            "benchmark_count": [benchmark_counts[int(value)] for value in selected_dates],
            "universe_count": [universe_counts[int(value)] for value in selected_dates],
        }
    )
    destination, temporary = _prepare_output_directory(output_dir)
    manifest = {
        "schema_version": 2,
        "status": "success",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "date_start": selected_dates[0],
        "date_end": selected_dates[-1],
        "source_date_start": start,
        "source_date_end": end,
        "rebalance_every": rebalance_every,
        "rebalance_count": len(selected_dates),
        "candidate_count_per_date": candidate_count,
        "higher_is_better": higher_is_better,
        "ranking_tie_break": "prediction then ticker ascending",
        "signal_semantics": (
            "signal_full is the calibration cross-section; signal_candidates is the "
            "tradable Top-N set; signal.parquet is a legacy candidate-signal alias"
        ),
        "universe_definition": "top_signal_candidates_union_positive_benchmark_constituents",
        "full_signal_rows": int(len(full_signal)),
        "candidate_rows": int(len(signal)),
        "signal_rows": int(len(signal)),
        "benchmark_rows": int(len(benchmark)),
        "optimizer_universe_rows": int(len(universe)),
        "optimizer_universe_count_min": int(dates["universe_count"].min()),
        "optimizer_universe_count_max": int(dates["universe_count"].max()),
        "source_quality": source_quality,
        "benchmark_quality": {
            **benchmark_quality,
            "normalization": "divide_positive_constituent_weights_by_daily_sum",
        },
        "inputs": {
            "signal": {
                "path": str(signal_path),
                "sha256": sha256_file(signal_path),
                "ticker_column": ticker_column,
                "signal_column": signal_column,
            },
            "benchmark": {
                "path": str(benchmark_path),
                "sha256": sha256_file(benchmark_path),
            },
            "eligibility": {
                "path": str(eligibility_path),
                "sha256": sha256_file(eligibility_path),
                "candidate_rule": "tradable is true",
                "benchmark_rule": "date-ticker market record exists",
            },
        },
        "outputs": list(OUTPUT_FILES),
    }
    try:
        signal.to_parquet(temporary / "signal.parquet", index=False)
        full_signal.to_parquet(temporary / "signal_full.parquet", index=False)
        signal.loc[:, ["date", "ticker"]].to_parquet(
            temporary / "signal_candidates.parquet", index=False
        )
        universe.to_parquet(temporary / "optimizer_universe.parquet", index=False)
        benchmark.to_parquet(temporary / "benchmark_weights.parquet", index=False)
        dates.to_parquet(temporary / "rebalance_dates.parquet", index=False)
        (temporary / "input_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "success",
        "rebalance_count": len(selected_dates),
        "full_signal_rows": int(len(full_signal)),
        "candidate_rows": int(len(signal)),
        "signal_rows": int(len(signal)),
        "optimizer_universe_rows": int(len(universe)),
        "universe_count_min": int(dates["universe_count"].min()),
        "universe_count_max": int(dates["universe_count"].max()),
        "output_dir": str(destination),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare frozen Alpha191+LGBM Top-N rolling optimizer inputs."
    )
    parser.add_argument("--signal-file", required=True)
    parser.add_argument("--benchmark-file", required=True)
    parser.add_argument("--eligibility-file", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=200)
    parser.add_argument("--ticker-column", default="symbol")
    parser.add_argument("--signal-column", default="signal")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--weight-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--benchmark-sum-tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--benchmark-exclusion-tolerance", type=float, default=5.0e-3
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = prepare_inputs(
            signal_file=args.signal_file,
            benchmark_file=args.benchmark_file,
            eligibility_file=args.eligibility_file,
            start_date=args.start_date,
            end_date=args.end_date,
            rebalance_every=args.rebalance_every,
            candidate_count=args.candidate_count,
            output_dir=args.output_dir,
            ticker_column=args.ticker_column,
            signal_column=args.signal_column,
            higher_is_better=not args.lower_is_better,
            weight_tolerance=args.weight_tolerance,
            benchmark_sum_tolerance=args.benchmark_sum_tolerance,
            benchmark_exclusion_tolerance=args.benchmark_exclusion_tolerance,
        )
    except Exception as exc:
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
