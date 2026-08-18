#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from portfolio_runtime.errors import InputDataError, PortfolioOptimizeError
from portfolio_runtime.io import normalize_date, read_table, sha256_file
from portfolio_runtime.risk_model import OUTPUT_FILES, build_risk_model_from_files


ROOT_MANIFEST = "rolling_risk_manifest.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _select_dates(
    universe_file: str | Path,
    *,
    start_date: object | None,
    end_date: object | None,
    rebalance_every: int,
) -> list[str]:
    if rebalance_every <= 0:
        raise InputDataError("rebalance_every must be positive")
    frame = read_table(universe_file)
    if "date" not in frame.columns or "ticker" not in frame.columns:
        raise InputDataError("universe file must contain date and ticker columns")
    dates = sorted({normalize_date(value) for value in frame["date"]})
    if start_date is not None:
        lower = normalize_date(start_date)
        dates = [value for value in dates if value >= lower]
    if end_date is not None:
        upper = normalize_date(end_date)
        dates = [value for value in dates if value <= upper]
    selected = dates[::rebalance_every]
    if not selected:
        raise InputDataError("no universe dates remain after date filters")
    return selected


def _input_records(
    *,
    config_path: str | Path,
    returns_file: str | Path,
    market_cap_file: str | Path,
    industry_file: str | Path | None,
    universe_file: str | Path,
) -> dict[str, dict[str, str]]:
    supplied: dict[str, str | Path] = {
        "config": config_path,
        "returns": returns_file,
        "market_cap": market_cap_file,
        "universe": universe_file,
    }
    if industry_file is not None:
        supplied["industry"] = industry_file
    records: dict[str, dict[str, str]] = {}
    for name, value in supplied.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise InputDataError(f"input file does not exist: {path}")
        records[name] = {"path": str(path), "sha256": sha256_file(path)}
    return records


def _completed_cache_matches(
    date_output: Path,
    date: str,
    inputs: dict[str, dict[str, str]],
) -> bool:
    if not date_output.exists():
        return False
    if not date_output.is_dir():
        raise InputDataError(f"risk-model cache path is not a directory: {date_output}")
    entries = list(date_output.iterdir())
    if not entries:
        return False
    missing = [name for name in OUTPUT_FILES if not (date_output / name).is_file()]
    manifest_path = date_output / "risk_model_manifest.json"
    if missing:
        raise InputDataError(
            f"incomplete risk-model cache for {date}; missing {missing}: {date_output}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputDataError(f"invalid risk-model cache manifest for {date}: {exc}") from exc
    if manifest.get("status") != "success" or manifest.get("requested_date") != date:
        raise InputDataError(f"invalid completed risk-model cache for {date}: {date_output}")
    cached_inputs = manifest.get("inputs")
    if not isinstance(cached_inputs, dict):
        raise InputDataError(f"risk-model cache has no input records for {date}")
    changed = set(cached_inputs) ^ set(inputs)
    changed.update(
        name
        for name, record in inputs.items()
        if cached_inputs.get(name, {}).get("sha256") != record["sha256"]
    )
    if changed:
        raise InputDataError(
            f"stale risk-model cache for {date}; changed inputs {sorted(changed)}: "
            f"{date_output}"
        )
    return True


def build_rolling_risk_models(
    *,
    config_path: str | Path,
    returns_file: str | Path,
    market_cap_file: str | Path,
    universe_file: str | Path,
    output_root: str | Path,
    industry_file: str | Path | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    rebalance_every: int = 1,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    root = Path(output_root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise InputDataError(f"output root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    dates = _select_dates(
        universe_file,
        start_date=start_date,
        end_date=end_date,
        rebalance_every=rebalance_every,
    )
    inputs = _input_records(
        config_path=config_path,
        returns_file=returns_file,
        market_cap_file=market_cap_file,
        industry_file=industry_file,
        universe_file=universe_file,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started.isoformat(),
        "completed_at": None,
        "selected_dates": dates,
        "selected_count": len(dates),
        "built_dates": [],
        "skipped_dates": [],
        "failed_date": None,
        "date_filters": {
            "start_date": None if start_date is None else normalize_date(start_date),
            "end_date": None if end_date is None else normalize_date(end_date),
            "rebalance_every": rebalance_every,
        },
        "inputs": inputs,
        "date_output_pattern": "date=YYYYMMDD",
        "date_outputs": list(OUTPUT_FILES),
    }
    manifest_path = root / ROOT_MANIFEST
    _write_json_atomic(manifest_path, manifest)
    active_date: str | None = None
    try:
        for position, date in enumerate(dates, start=1):
            active_date = date
            date_output = root / f"date={date}"
            if _completed_cache_matches(date_output, date, inputs):
                manifest["skipped_dates"].append(date)
                action = "skipped"
            else:
                result = build_risk_model_from_files(
                    config_path=config_path,
                    returns_file=returns_file,
                    market_cap_file=market_cap_file,
                    industry_file=industry_file,
                    universe_file=universe_file,
                    requested_date=date,
                    output_dir=date_output,
                )
                manifest["built_dates"].append(date)
                action = "built"
                if result.get("status") != "success":
                    raise InputDataError(f"risk-model build returned non-success for {date}")
            _write_json_atomic(manifest_path, manifest)
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "date": date,
                        "action": action,
                        "position": position,
                        "total": len(dates),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    except Exception as exc:
        manifest["status"] = "error"
        manifest["failed_date"] = active_date
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json_atomic(manifest_path, manifest)
        raise
    manifest["status"] = "success"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, manifest)
    return {
        "status": "success",
        "selected_count": len(dates),
        "built_count": len(manifest["built_dates"]),
        "skipped_count": len(manifest["skipped_dates"]),
        "date_start": dates[0],
        "date_end": dates[-1],
        "output_root": str(root),
        "manifest": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and resume date-partitioned structural risk-model caches."
    )
    parser.add_argument("--config", required=True, help="Risk-model YAML config")
    parser.add_argument("--returns-file", required=True, help="Date x ticker return panel")
    parser.add_argument("--market-cap-file", required=True, help="Date x ticker market-cap panel")
    parser.add_argument("--industry-file", help="Optional interval industry history")
    parser.add_argument(
        "--universe-file", required=True, help="Long-form date-ticker universe or signal file"
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_rolling_risk_models(
            config_path=args.config,
            returns_file=args.returns_file,
            market_cap_file=args.market_cap_file,
            industry_file=args.industry_file,
            universe_file=args.universe_file,
            start_date=args.start_date,
            end_date=args.end_date,
            rebalance_every=args.rebalance_every,
            output_root=args.output_root,
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
