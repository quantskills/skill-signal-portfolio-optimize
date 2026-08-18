#!/usr/bin/env python3
"""Adapt an existing candidate universe and asset covariance to the v0.1 contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_runtime.errors import InputDataError
from portfolio_runtime.io import normalize_date, normalize_ticker, read_table, sha256_file


OUTPUT_FILES = (
    "signal.parquet",
    "covariance.parquet",
    "benchmark_weights.parquet",
    "input_manifest.json",
)


def _prepare_output_directory(output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise InputDataError(
                f"output path exists and is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise InputDataError(f"output directory must be new or empty: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    return destination, temporary


def _load_covariance(path: str | Path) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise InputDataError(f"input file does not exist: {file_path}")
    try:
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path, index_col=0)
        elif file_path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(file_path)
        else:
            raise InputDataError(f"unsupported covariance format: {file_path.suffix}")
    except InputDataError:
        raise
    except Exception as exc:
        raise InputDataError(f"cannot read covariance {file_path}: {exc}") from exc
    if frame.shape[0] != frame.shape[1]:
        raise InputDataError(f"covariance must be square, got {frame.shape}")
    frame.index = pd.Index([normalize_ticker(value) for value in frame.index])
    frame.columns = pd.Index([normalize_ticker(value) for value in frame.columns])
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise InputDataError("covariance has duplicate row or column tickers")
    if set(frame.index) != set(frame.columns):
        raise InputDataError("covariance row and column ticker sets differ")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise InputDataError("covariance contains missing or non-finite values")
    return numeric.reindex(index=frame.columns, columns=frame.columns).astype(float)


def prepare_inputs(
    *,
    candidate_file: str | Path,
    covariance_file: str | Path,
    requested_date: object,
    output_dir: str | Path,
    ticker_column: str = "symbol",
    signal_column: str = "alpha_score",
) -> dict[str, object]:
    date = normalize_date(requested_date)
    candidates = read_table(candidate_file)
    required = {"date", ticker_column, signal_column}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise InputDataError(
            f"candidate file missing column(s): {', '.join(missing)}"
        )
    normalized_dates = candidates["date"].map(normalize_date)
    selected = candidates.loc[
        normalized_dates.eq(date), [ticker_column, signal_column]
    ].copy()
    if selected.empty:
        raise InputDataError(f"candidate file has no rows for {date}")
    selected["ticker"] = selected[ticker_column].map(normalize_ticker)
    if selected["ticker"].duplicated().any():
        duplicates = selected.loc[
            selected["ticker"].duplicated(keep=False), "ticker"
        ].unique()
        raise InputDataError(
            f"candidate file has duplicate ticker(s): {list(duplicates[:10])}"
        )
    selected["prediction"] = pd.to_numeric(selected[signal_column], errors="coerce")
    if not np.isfinite(selected["prediction"].to_numpy(dtype=float)).all():
        raise InputDataError("candidate signal contains missing or non-finite values")
    signal = selected.set_index("ticker")["prediction"].astype(float)

    covariance = _load_covariance(covariance_file)
    extra = covariance.index.difference(signal.index)
    if len(extra):
        raise InputDataError(
            f"covariance contains ticker(s) outside candidate universe: {list(extra[:10])}"
        )
    excluded = signal.index.difference(covariance.index).sort_values()
    universe = covariance.index
    signal_output = pd.DataFrame(
        {
            "date": date,
            "ticker": universe,
            "prediction": signal.reindex(universe).to_numpy(dtype=float),
        }
    )
    benchmark_output = pd.DataFrame(
        {
            "date": date,
            "ticker": universe,
            "benchmark_weight": np.full(len(universe), 1.0 / len(universe)),
        }
    )

    destination, temporary = _prepare_output_directory(output_dir)
    candidate_path = Path(candidate_file).expanduser().resolve()
    covariance_path = Path(covariance_file).expanduser().resolve()
    manifest = {
        "schema_version": 1,
        "status": "success",
        "requested_date": date,
        "candidate_count": int(len(signal)),
        "asset_count": int(len(universe)),
        "excluded_without_covariance_count": int(len(excluded)),
        "excluded_without_covariance_tickers": excluded.tolist(),
        "benchmark_semantics": "equal_weight_covariance_eligible_signal_universe",
        "benchmark_is_market_index": False,
        "inputs": {
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
                "ticker_column": ticker_column,
                "signal_column": signal_column,
            },
            "covariance": {
                "path": str(covariance_path),
                "sha256": sha256_file(covariance_path),
            },
        },
        "outputs": list(OUTPUT_FILES),
    }
    try:
        signal_output.to_parquet(temporary / "signal.parquet", index=False)
        covariance.to_parquet(temporary / "covariance.parquet", index=True)
        benchmark_output.to_parquet(
            temporary / "benchmark_weights.parquet", index=False
        )
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
    return {**manifest, "output_dir": str(destination)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one-date v0.1 inputs using an equal-weight benchmark over "
            "the covariance-eligible signal universe."
        )
    )
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--covariance-file", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ticker-column", default="symbol")
    parser.add_argument("--signal-column", default="alpha_score")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = prepare_inputs(
            candidate_file=args.candidate_file,
            covariance_file=args.covariance_file,
            requested_date=args.date,
            output_dir=args.output_dir,
            ticker_column=args.ticker_column,
            signal_column=args.signal_column,
        )
    except InputDataError as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
