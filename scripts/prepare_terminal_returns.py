#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_runtime.errors import InputDataError, PortfolioOptimizeError
from portfolio_runtime.io import normalize_date, normalize_ticker, read_table, sha256_file


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_terminal_returns(
    *,
    input_file: str | Path,
    output_file: str | Path,
    manifest_file: str | Path,
    end_date: object,
    terminal_return: float = -1.0,
) -> dict[str, object]:
    source = Path(input_file).expanduser().resolve()
    destination = Path(output_file).expanduser().resolve()
    manifest_path = Path(manifest_file).expanduser().resolve()
    date_end = normalize_date(end_date)
    if not np.isfinite(terminal_return) or terminal_return < -1.0:
        raise InputDataError("terminal_return must be finite and at least -1")
    input_hash = sha256_file(source)
    expected = {
        "input_sha256": input_hash,
        "end_date": date_end,
        "terminal_return": float(terminal_return),
    }
    if destination.exists() or manifest_path.exists():
        if not destination.is_file() or not manifest_path.is_file():
            raise InputDataError("terminal-return output is incomplete")
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputDataError(f"invalid terminal-return manifest: {exc}") from exc
        if all(prior.get(key) == value for key, value in expected.items()):
            if prior.get("output_sha256") != sha256_file(destination):
                raise InputDataError("terminal-return output hash does not match manifest")
            return {
                "status": "success",
                "cache_status": "reused",
                "output_file": str(destination),
                "terminal_event_count": int(prior["terminal_event_count"]),
            }
        raise InputDataError("terminal-return output exists with stale inputs")

    frame = read_table(source)
    required = {"date", "ticker", "return"}
    missing = required - set(frame.columns)
    if missing:
        raise InputDataError(
            "asset returns missing column(s): " + ", ".join(sorted(missing))
        )
    result = frame.loc[:, ["date", "ticker", "return"]].copy()
    result["date"] = result["date"].map(normalize_date)
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result["return"] = pd.to_numeric(result["return"], errors="coerce")
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError("asset returns contain duplicate date/ticker rows")
    if not np.isfinite(result["return"].to_numpy(dtype=float)).all():
        raise InputDataError("asset returns contain missing or non-finite values")
    calendar = sorted(date for date in result["date"].unique() if date <= date_end)
    if not calendar:
        raise InputDataError(f"asset returns have no dates through {date_end}")
    positions = {date: position for position, date in enumerate(calendar)}
    last_dates = result.loc[result["date"].le(date_end)].groupby("ticker")["date"].max()
    events: list[dict[str, object]] = []
    for ticker, last_date in last_dates.items():
        position = positions[last_date]
        if position + 1 >= len(calendar):
            continue
        events.append(
            {
                "date": calendar[position + 1],
                "ticker": ticker,
                "return": float(terminal_return),
                "last_observed_date": last_date,
            }
        )
    additions = pd.DataFrame(
        events, columns=["date", "ticker", "return", "last_observed_date"]
    )
    combined = pd.concat(
        [result, additions.loc[:, ["date", "ticker", "return"]]],
        ignore_index=True,
    ).sort_values(["date", "ticker"], kind="stable")
    if combined.duplicated(["date", "ticker"]).any():
        raise InputDataError("terminal-return derivation created duplicate rows")

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".terminal-returns-", dir=destination.parent))
    temporary_output = temporary_dir / destination.name
    temporary_manifest = temporary_dir / manifest_path.name
    try:
        combined.to_parquet(temporary_output, index=False)
        manifest = {
            "schema_version": 1,
            "status": "success",
            **expected,
            "input_file": str(source),
            "output_file": str(destination),
            "input_row_count": int(len(result)),
            "output_row_count": int(len(combined)),
            "terminal_event_count": int(len(events)),
            "terminal_events": events,
            "assumption": (
                "A ticker whose observed return series ends permanently before end_date "
                "receives the configured terminal return on the next market calendar date. "
                "This is a conservative delisting/write-off assumption, not an observed return."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_sha256": sha256_file(temporary_output),
        }
        _write_json(temporary_manifest, manifest)
        os.replace(temporary_output, destination)
        os.replace(temporary_manifest, manifest_path)
        temporary_dir.rmdir()
    except BaseException:
        if temporary_output.exists():
            temporary_output.unlink()
        if temporary_manifest.exists():
            temporary_manifest.unlink()
        if temporary_dir.exists():
            temporary_dir.rmdir()
        raise
    return {
        "status": "success",
        "cache_status": "built",
        "output_file": str(destination),
        "terminal_event_count": len(events),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add explicit conservative terminal returns to permanently ended series."
    )
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--manifest-file", required=True, type=Path)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--terminal-return", type=float, default=-1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = prepare_terminal_returns(
            input_file=args.input_file,
            output_file=args.output_file,
            manifest_file=args.manifest_file,
            end_date=args.end_date,
            terminal_return=args.terminal_return,
        )
    except PortfolioOptimizeError as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
