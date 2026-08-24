#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from portfolio_runtime.config import load_config
from portfolio_runtime.errors import InputDataError, PortfolioOptimizeError
from portfolio_runtime.io import sha256_file
from portfolio_runtime.rolling import _runtime_source_hash, run_rolling_experiment
from portfolio_runtime.sweep import (
    apply_dotted_overrides,
    flatten_sweep_metrics,
    load_sweep_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible rolling parameter sweep over portfolio configs."
    )
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--matrix-file", required=True, type=Path)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--signal-file", required=True, type=Path)
    parser.add_argument("--covariance-root", type=Path)
    parser.add_argument("--benchmark-file", required=True, type=Path)
    parser.add_argument("--asset-returns-file", required=True, type=Path)
    parser.add_argument("--sector-file", type=Path)
    parser.add_argument("--exposure-file", type=Path)
    parser.add_argument("--exposure-root", type=Path)
    parser.add_argument("--factor-covariance-root", type=Path)
    parser.add_argument("--specific-variance-root", type=Path)
    parser.add_argument("--tradability-file", type=Path)
    parser.add_argument("--initial-weights-file", type=Path)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--transaction-cost-bps", type=float)
    parser.add_argument("--risk-model-config", type=Path)
    parser.add_argument("--risk-returns-file", type=Path)
    parser.add_argument("--risk-market-cap-file", type=Path)
    parser.add_argument("--risk-industry-file", type=Path)
    parser.add_argument("--dynamic-risk-cache-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "sweep_summary.csv", index=False)
    (root / "sweep_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _completed_metrics(output: Path, config_path: Path) -> dict[str, Any] | None:
    if not output.exists():
        return None
    manifest_path = output / "rolling_manifest.json"
    metrics_path = output / "portfolio_metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise InputDataError(f"incomplete sweep run output exists at {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("inputs", {}).get("config", {}).get("sha256")
    if (
        manifest.get("status") != "success"
        or recorded != sha256_file(config_path)
        or manifest.get("runtime_source_sha256") != _runtime_source_hash()
    ):
        raise InputDataError(f"stale sweep run output exists at {output}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    try:
        base = load_config(args.base_config)
        variants = load_sweep_variants(args.matrix_file)
        root = args.output_root.expanduser().resolve()
        config_root = root / "configs"
        run_root = root / "runs"
        config_root.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for position, variant in enumerate(variants, start=1):
            name = variant["name"]
            resolved = apply_dotted_overrides(base, variant["overrides"])
            config_path = config_root / f"{name}.yaml"
            config_path.write_text(
                yaml.safe_dump(resolved, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            output = run_root / name
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "stage": "parameter_sweep",
                        "variant": name,
                        "position": position,
                        "total": len(variants),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            try:
                metrics = _completed_metrics(output, config_path)
                if metrics is None:
                    result = run_rolling_experiment(
                        config_path=config_path,
                        signal_file=args.signal_file,
                        candidate_file=args.candidate_file,
                        covariance_root=args.covariance_root,
                        benchmark_file=args.benchmark_file,
                        asset_returns_file=args.asset_returns_file,
                        sector_file=args.sector_file,
                        exposure_file=args.exposure_file,
                        exposure_root=args.exposure_root,
                        factor_covariance_root=args.factor_covariance_root,
                        specific_variance_root=args.specific_variance_root,
                        tradability_file=args.tradability_file,
                        initial_weights_file=args.initial_weights_file,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        rebalance_every=args.rebalance_every,
                        transaction_cost_bps=args.transaction_cost_bps,
                        risk_model_config=args.risk_model_config,
                        risk_returns_file=args.risk_returns_file,
                        risk_market_cap_file=args.risk_market_cap_file,
                        risk_industry_file=args.risk_industry_file,
                        dynamic_risk_cache_root=args.dynamic_risk_cache_root,
                        checkpoint_root=(
                            None
                            if args.checkpoint_root is None
                            else args.checkpoint_root / name
                        ),
                        output_dir=output,
                    )
                    metrics = {"portfolios": result["portfolios"]}
                rows.append(flatten_sweep_metrics(name, metrics))
            except PortfolioOptimizeError as exc:
                rows.append(
                    {
                        "variant": name,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                _write_summary(root, rows)
                if not args.continue_on_error:
                    raise
            _write_summary(root, rows)
    except (PortfolioOptimizeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    error_count = sum(row["status"] == "error" for row in rows)
    print(
        json.dumps(
            {
                "status": "success" if error_count == 0 else "completed_with_errors",
                "variant_count": len(rows),
                "success_count": len(rows) - error_count,
                "error_count": error_count,
                "output_root": str(root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
