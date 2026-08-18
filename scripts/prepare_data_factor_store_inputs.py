#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


OUTPUT_FILES = (
    "returns.parquet",
    "market_cap.parquet",
    "asset_returns.parquet",
    "benchmark_weights.parquet",
    "tradability.parquet",
    "data_manifest.json",
)


def parse_date(value: str) -> int:
    parsed = pd.to_datetime(str(value), errors="raise")
    return int(parsed.strftime("%Y%m%d"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare portable optimizer inputs from the canonical data-factor store."
    )
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--risk-history-start")
    parser.add_argument("--index-symbol", default="000852.SH")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> dict[str, object]:
    store = args.store_root.expanduser().resolve()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    history_start = parse_date(args.risk_history_start or args.start_date)
    if history_start > start or start > end:
        raise ValueError("require risk_history_start <= start_date <= end_date")
    equity_glob = (
        store / "canonical/equity_daily/schema=v1/year=*/month=*/data.parquet"
    )
    weight_glob = (
        store
        / "canonical/index_weights"
        / f"symbol={args.index_symbol}"
        / "year=*/data.parquet"
    )
    if not list((store / "canonical/equity_daily/schema=v1").glob("year=*/month=*/data.parquet")):
        raise FileNotFoundError(f"canonical equity data is missing under {store}")
    if not list(
        (store / "canonical/index_weights" / f"symbol={args.index_symbol}").glob(
            "year=*/data.parquet"
        )
    ):
        raise FileNotFoundError(f"canonical index weights are missing for {args.index_symbol}")

    connection = duckdb.connect()
    equity = connection.execute(
        """
        SELECT CAST(date AS BIGINT) AS date,
               CAST(symbol AS VARCHAR) AS ticker,
               CAST(close AS DOUBLE) AS close,
               CAST(pre_close AS DOUBLE) AS pre_close,
               CAST(market_value AS DOUBLE) AS market_value,
               CAST(volume AS DOUBLE) AS volume,
               CAST(amount AS DOUBLE) AS amount,
               CAST(trade_status AS DOUBLE) AS trade_status
        FROM read_parquet(?, hive_partitioning=false)
        WHERE date BETWEEN ? AND ?
        ORDER BY date, ticker
        """,
        [str(equity_glob), history_start, end],
    ).fetchdf()
    weights = connection.execute(
        """
        SELECT CAST(date AS BIGINT) AS date,
               CAST(stock_symbol AS VARCHAR) AS ticker,
               CAST(weight AS DOUBLE) AS benchmark_weight
        FROM read_parquet(?, hive_partitioning=false)
        WHERE date BETWEEN ? AND ?
        ORDER BY date, ticker
        """,
        [str(weight_glob), start, end],
    ).fetchdf()
    connection.close()
    if equity.empty:
        raise ValueError("canonical equity query returned no rows")
    if weights.empty:
        raise ValueError("canonical benchmark query returned no rows")
    if equity.duplicated(["date", "ticker"]).any():
        raise ValueError("canonical equity data contain duplicate date-ticker keys")
    if weights.duplicated(["date", "ticker"]).any():
        raise ValueError("canonical benchmark weights contain duplicate keys")

    valid_price = equity["close"].gt(0) & equity["pre_close"].gt(0)
    equity["return"] = np.nan
    equity.loc[valid_price, "return"] = (
        equity.loc[valid_price, "close"] / equity.loc[valid_price, "pre_close"] - 1.0
    )
    equity["tradable"] = (
        equity["trade_status"].eq(0)
        & equity["volume"].gt(0)
        & equity["amount"].gt(0)
    )
    returns_long = equity.loc[
        equity["return"].notna() & np.isfinite(equity["return"]),
        ["date", "ticker", "return"],
    ].copy()
    if returns_long["return"].le(-1.0).any():
        raise ValueError("computed returns contain values at or below -100%")
    cap_long = equity.loc[
        equity["market_value"].gt(0) & np.isfinite(equity["market_value"]),
        ["date", "ticker", "market_value"],
    ]
    returns_wide = returns_long.pivot(index="date", columns="ticker", values="return")
    cap_wide = cap_long.pivot(index="date", columns="ticker", values="market_value")
    returns_wide.index = returns_wide.index.astype(str)
    cap_wide.index = cap_wide.index.astype(str)
    asset_returns = returns_long.loc[returns_long["date"].between(start, end)].copy()
    tradability = equity.loc[
        equity["date"].between(start, end), ["date", "ticker", "tradable"]
    ].copy()

    destination = args.output_dir.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"output directory must be new or empty: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        returns_wide.to_parquet(temporary / "returns.parquet")
        cap_wide.to_parquet(temporary / "market_cap.parquet")
        asset_returns.to_parquet(temporary / "asset_returns.parquet", index=False)
        weights.to_parquet(temporary / "benchmark_weights.parquet", index=False)
        tradability.to_parquet(temporary / "tradability.parquet", index=False)
        manifest = {
            "schema_version": 1,
            "status": "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "store_root": str(store),
            "history_start": history_start,
            "start_date": start,
            "end_date": end,
            "index_symbol": args.index_symbol,
            "return_formula": "close / pre_close - 1",
            "tradability_rule": "trade_status == 0 and volume > 0 and amount > 0",
            "rows": {
                "asset_returns": int(len(asset_returns)),
                "benchmark_weights": int(len(weights)),
                "tradability": int(len(tradability)),
            },
            "outputs": list(OUTPUT_FILES),
        }
        (temporary / "data_manifest.json").write_text(
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
        "output_dir": str(destination),
        "asset_return_rows": int(len(asset_returns)),
        "benchmark_weight_rows": int(len(weights)),
        "tradability_rows": int(len(tradability)),
    }


def main() -> int:
    args = parse_args()
    try:
        result = prepare(args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
