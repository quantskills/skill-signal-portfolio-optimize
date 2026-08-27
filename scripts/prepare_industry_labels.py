#!/usr/bin/env python3
"""Materialize point-in-time industry labels for portfolio constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_runtime.errors import InputDataError
from portfolio_runtime.io import normalize_date, normalize_ticker, read_table, sha256_file


_HISTORY_COLUMNS = {"stock_symbol", "l1_code", "in_date", "out_date"}
_NORMALIZED_HISTORY_COLUMNS = {"ticker", "industry", "in_date", "out_date_normalized"}
_MISSING_POLICIES = {"error", "exclude"}


def _optional_date(value: object) -> str:
    if value is None or pd.isna(value) or str(value).strip() in {"", "<NA>", "nan", "None"}:
        return "99991231"
    return normalize_date(value)


def load_interval_history(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    columns = set(frame.columns)
    if _HISTORY_COLUMNS.issubset(columns):
        result = frame.loc[:, ["stock_symbol", "l1_code", "in_date", "out_date"]].rename(
            columns={"stock_symbol": "ticker", "l1_code": "industry"}
        )
    elif _NORMALIZED_HISTORY_COLUMNS.issubset(columns):
        result = frame.loc[:, ["ticker", "industry", "in_date", "out_date_normalized"]].rename(
            columns={"out_date_normalized": "out_date"}
        )
    else:
        raise InputDataError(
            "industry history must contain either "
            "stock_symbol/l1_code/in_date/out_date or "
            "ticker/industry/in_date/out_date_normalized"
        )

    result = result.copy()
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result["industry"] = result["industry"].astype("string").str.strip()
    if result["industry"].isna().any() or result["industry"].eq("").any():
        raise InputDataError("industry history contains missing or empty industry codes")
    result["in_date"] = result["in_date"].map(normalize_date)
    result["out_date"] = result["out_date"].map(_optional_date)
    if (result["out_date"] < result["in_date"]).any():
        raise InputDataError("industry history contains out_date before in_date")

    result = result.sort_values(["ticker", "in_date", "out_date"], kind="stable")
    for ticker, group in result.groupby("ticker", sort=False):
        previous_end = group["out_date"].shift()
        overlaps = group["in_date"].le(previous_end).fillna(False)
        if overlaps.any():
            raise InputDataError(f"industry history has overlapping intervals for {ticker}")
    return result.reset_index(drop=True)


def load_universe(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    required = {"date", "ticker"}
    missing = required - set(frame.columns)
    if missing:
        raise InputDataError("universe missing column(s): " + ", ".join(sorted(missing)))
    result = frame.loc[:, ["date", "ticker"]].copy()
    result["date"] = result["date"].map(normalize_date)
    result["ticker"] = result["ticker"].map(normalize_ticker)
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError("universe contains duplicate date/ticker rows")
    if result.empty:
        raise InputDataError("universe is empty")
    return result.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def prepare_industry_labels(
    *,
    history_file: str | Path,
    universe_file: str | Path,
    output_file: str | Path,
    coverage_file: str | Path | None = None,
    manifest_file: str | Path | None = None,
    minimum_coverage: float = 1.0,
    missing_policy: str = "error",
    filtered_universe_file: str | Path | None = None,
    exclusions_file: str | Path | None = None,
    minimum_remaining_count: int = 1,
) -> dict[str, Any]:
    if not np.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
        raise InputDataError("minimum_coverage must be between 0 and 1")
    if missing_policy not in _MISSING_POLICIES:
        raise InputDataError(
            "missing_policy must be one of: " + ", ".join(sorted(_MISSING_POLICIES))
        )
    if (
        isinstance(minimum_remaining_count, bool)
        or not isinstance(minimum_remaining_count, int)
        or minimum_remaining_count <= 0
    ):
        raise InputDataError("minimum_remaining_count must be a positive integer")
    history = load_interval_history(history_file)
    universe = load_universe(universe_file)

    label_parts: list[pd.DataFrame] = []
    filtered_universe_parts: list[pd.DataFrame] = []
    exclusion_rows: list[dict[str, str]] = []
    coverage_rows: list[dict[str, Any]] = []
    for date, requested in universe.groupby("date", sort=True):
        active = history.loc[
            history["in_date"].le(date) & history["out_date"].ge(date),
            ["ticker", "industry"],
        ]
        if active["ticker"].duplicated().any():
            raise InputDataError(f"multiple active industries exist on {date}")
        merged = requested.merge(active, on="ticker", how="left", validate="one_to_one")
        mapped = merged["industry"].notna() & merged["industry"].astype("string").str.strip().ne("")
        mapped_count = int(mapped.sum())
        constituent_count = int(len(merged))
        coverage = mapped_count / constituent_count if constituent_count else 0.0
        missing = merged.loc[~mapped, "ticker"].tolist()
        coverage_rows.append(
            {
                "date": date,
                "constituents": constituent_count,
                "industry_mapped": mapped_count,
                "industry_unmapped": constituent_count - mapped_count,
                "coverage": coverage,
                "coverage_after_filter": 1.0 if mapped_count else 0.0,
            }
        )
        if coverage < minimum_coverage and missing_policy == "error":
            sample = ", ".join(missing[:10])
            raise InputDataError(
                f"industry coverage on {date} is {coverage:.4%}, below "
                f"minimum_coverage={minimum_coverage:.4%}; missing: {sample}"
            )
        if missing_policy == "exclude" and mapped_count < minimum_remaining_count:
            sample = ", ".join(missing[:10])
            raise InputDataError(
                f"no industry-valid candidates remain on {date}; "
                f"mapped={mapped_count}, minimum_remaining_count={minimum_remaining_count}; "
                f"missing: {sample}"
            )
        filtered_universe_parts.append(merged.loc[mapped, ["date", "ticker"]])
        exclusion_rows.extend(
            {"date": date, "ticker": ticker, "reason": "missing_industry_asof"}
            for ticker in missing
        )
        label_parts.append(
            merged.loc[mapped, ["date", "ticker", "industry"]].rename(
                columns={"industry": "sector"}
            )
        )

    labels = pd.concat(label_parts, ignore_index=True)
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(destination, index=False)

    filtered_destination = None
    exclusions_destination = None
    if missing_policy == "exclude":
        filtered_destination = (
            destination.with_name(f"{destination.stem}.filtered_universe.parquet")
            if filtered_universe_file is None
            else Path(filtered_universe_file).expanduser().resolve()
        )
        filtered_destination.parent.mkdir(parents=True, exist_ok=True)
        filtered_universe = pd.concat(filtered_universe_parts, ignore_index=True)
        filtered_universe.to_parquet(filtered_destination, index=False)

        exclusions_destination = (
            destination.with_name(f"{destination.stem}.exclusions.parquet")
            if exclusions_file is None
            else Path(exclusions_file).expanduser().resolve()
        )
        exclusions_destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            exclusion_rows, columns=["date", "ticker", "reason"]
        ).to_parquet(exclusions_destination, index=False)

    coverage_destination = (
        destination.with_name(f"{destination.stem}.coverage.parquet")
        if coverage_file is None
        else Path(coverage_file).expanduser().resolve()
    )
    coverage_destination.parent.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_parquet(coverage_destination, index=False)

    manifest_destination = (
        destination.with_name(f"{destination.stem}.manifest.json")
        if manifest_file is None
        else Path(manifest_file).expanduser().resolve()
    )
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "success",
        "history_file": str(Path(history_file).expanduser().resolve()),
        "history_sha256": sha256_file(history_file),
        "universe_file": str(Path(universe_file).expanduser().resolve()),
        "universe_sha256": sha256_file(universe_file),
        "output_file": str(destination),
        "coverage_file": str(coverage_destination),
        "minimum_coverage": float(minimum_coverage),
        "missing_policy": missing_policy,
        "coverage_check_enforced": bool(missing_policy == "error"),
        "filtered_universe_file": (
            None if filtered_destination is None else str(filtered_destination)
        ),
        "exclusions_file": (
            None if exclusions_destination is None else str(exclusions_destination)
        ),
        "excluded_row_count": int(len(exclusion_rows)),
        "excluded_unique_ticker_count": int(
            len({row["ticker"] for row in exclusion_rows})
        ),
        "original_universe_rows": int(len(universe)),
        "filtered_universe_rows": int(
            sum(len(part) for part in filtered_universe_parts)
        ),
        "minimum_remaining_count": int(minimum_remaining_count),
        "date_start": str(coverage["date"].min()),
        "date_end": str(coverage["date"].max()),
        "date_count": int(len(coverage)),
        "label_rows": int(len(labels)),
        "minimum_observed_coverage": float(coverage["coverage"].min()),
        "maximum_observed_coverage": float(coverage["coverage"].max()),
        "minimum_observed_coverage_after_filter": float(
            coverage["coverage_after_filter"].min()
        ),
    }
    manifest_destination.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize strict point-in-time industry labels from interval history."
    )
    parser.add_argument("--history-file", required=True, type=Path)
    parser.add_argument("--universe-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--coverage-file", type=Path)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--minimum-coverage", type=float, default=1.0)
    parser.add_argument(
        "--missing-policy",
        choices=sorted(_MISSING_POLICIES),
        default="error",
        help="error on missing labels, or exclude only missing candidates",
    )
    parser.add_argument("--filtered-universe-file", type=Path)
    parser.add_argument("--exclusions-file", type=Path)
    parser.add_argument("--minimum-remaining-count", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = prepare_industry_labels(
            history_file=args.history_file,
            universe_file=args.universe_file,
            output_file=args.output_file,
            coverage_file=args.coverage_file,
            manifest_file=args.manifest_file,
            minimum_coverage=args.minimum_coverage,
            missing_policy=args.missing_policy,
            filtered_universe_file=args.filtered_universe_file,
            exclusions_file=args.exclusions_file,
            minimum_remaining_count=args.minimum_remaining_count,
        )
    except InputDataError as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
