#!/usr/bin/env python3
"""Materialize partitioned PIT fundamentals into one risk-model input parquet."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from portfolio_runtime.errors import InputDataError


KEY_COLUMNS = [
    "symbol",
    "available_date",
    "report_year",
    "report_quarter",
    "quarter",
    "if_adjusted",
    "is_warmup",
]


def _files(root: Path) -> list[Path]:
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise InputDataError(f"no parquet files found below {root}")
    return paths


def _read(root: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in _files(root)]
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--extended-root", type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    args = parser.parse_args()
    base_root = args.base_root.expanduser().resolve()
    if not base_root.is_dir():
        raise InputDataError(f"base-root is not a directory: {base_root}")
    output = args.output_file.expanduser().resolve()
    if output.exists():
        raise InputDataError(f"output-file already exists: {output}")

    base = _read(base_root)
    missing = sorted(set(KEY_COLUMNS) - set(base.columns))
    if missing:
        raise InputDataError("base PIT data missing key(s): " + ", ".join(missing))
    result = base
    if args.extended_root is not None:
        extended_root = args.extended_root.expanduser().resolve()
        if not extended_root.is_dir():
            raise InputDataError(f"extended-root is not a directory: {extended_root}")
        extended = _read(extended_root)
        missing = sorted(set(KEY_COLUMNS) - set(extended.columns))
        if missing:
            raise InputDataError(
                "extended PIT data missing key(s): " + ", ".join(missing)
            )
        if base.duplicated(KEY_COLUMNS).any() or extended.duplicated(KEY_COLUMNS).any():
            raise InputDataError("PIT source contains duplicate report-version keys")
        additions = [column for column in extended.columns if column not in base.columns]
        result = base.merge(
            extended[KEY_COLUMNS + additions],
            on=KEY_COLUMNS,
            how="outer",
            validate="one_to_one",
        )
    result = result.sort_values(
        ["available_date", "symbol", "report_year", "report_quarter"], kind="stable"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".parquet", dir=output.parent)[1])
    try:
        result.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "status": "success",
                "output_file": str(output),
                "rows": int(len(result)),
                "columns": result.columns.tolist(),
                "available_date_min": str(result["available_date"].min()),
                "available_date_max": str(result["available_date"].max()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputDataError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
