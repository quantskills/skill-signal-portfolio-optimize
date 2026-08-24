from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .backtest import (
    add_benchmark_relative_performance,
    backtest_targets,
    drift_weights,
    load_asset_returns,
    summarize_backtest,
)
from .config import load_config
from .cost import resolve_linear_cost_bps
from .dynamic_risk import DynamicRiskModelCache
from .errors import InputDataError
from .io import (
    load_candidate_universe,
    load_signal,
    load_tradability,
    load_weight_series,
    normalize_date,
    read_table,
    sha256_file,
)
from .pipeline import OUTPUT_FILES as SINGLE_DATE_OUTPUT_FILES
from .pipeline import build_optimization_universe, run_single_date


ROLLING_OUTPUT_FILES = (
    "rebalance_weights.parquet",
    "daily_performance.parquet",
    "exposure_timeseries.parquet",
    "optimization_diagnostics.parquet",
    "portfolio_metrics.json",
    "rolling_manifest.json",
    "optimization_summary.json",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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


def resolve_covariance_path(root: str | Path, date: str) -> Path:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / f"date={date}" / "asset_cov.parquet",
        path / f"date={date}" / "covariance.parquet",
        path / f"{date}.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InputDataError(
        f"no covariance file exists for {date}; checked {[str(item) for item in candidates]}"
    )


def resolve_exposure_path(root: str | Path, date: str) -> Path:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / f"date={date}" / "exposures.parquet",
        path / f"date={date}" / "style_exposures.parquet",
        path / f"{date}.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InputDataError(
        f"no exposure file exists for {date}; checked {[str(item) for item in candidates]}"
    )


def resolve_factor_covariance_path(root: str | Path, date: str) -> Path:
    path = Path(root).expanduser().resolve()
    candidates = (path,) if path.is_file() else (
        path / f"date={date}" / "factor_cov.parquet",
        path / f"{date}.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InputDataError(
        f"no factor covariance file exists for {date}; checked "
        f"{[str(item) for item in candidates]}"
    )


def resolve_specific_variance_path(root: str | Path, date: str) -> Path:
    path = Path(root).expanduser().resolve()
    candidates = (path,) if path.is_file() else (
        path / f"date={date}" / "specific_var.parquet",
        path / f"{date}.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InputDataError(
        f"no specific variance file exists for {date}; checked "
        f"{[str(item) for item in candidates]}"
    )


def _select_rebalance_dates(
    signal_file: str | Path,
    *,
    start_date: object | None,
    end_date: object | None,
    rebalance_every: int,
) -> list[str]:
    if rebalance_every <= 0:
        raise InputDataError("rebalance_every must be positive")
    frame = read_table(signal_file)
    if "date" not in frame.columns:
        raise InputDataError("signal file must contain date")
    dates = sorted({normalize_date(value) for value in frame["date"]})
    if start_date is not None:
        lower = normalize_date(start_date)
        dates = [value for value in dates if value >= lower]
    if end_date is not None:
        upper = normalize_date(end_date)
        dates = [value for value in dates if value <= upper]
    selected = dates[::rebalance_every]
    if not selected:
        raise InputDataError("no signal dates remain after rolling date filters")
    return selected


def _initial_weights(
    first_date: str,
    benchmark_file: str | Path,
    initial_weights_file: str | Path | None,
) -> pd.Series:
    if initial_weights_file is None:
        return load_weight_series(
            benchmark_file, first_date, "benchmark_weight", "benchmark"
        )
    return load_weight_series(
        initial_weights_file, first_date, "current_weight", "initial weights"
    )


def _diagnostic_rows(
    date: str,
    constraint_path: Path,
    risk_path: Path,
    signal_path: Path,
    optimization_summary_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    constraint_payload = json.loads(constraint_path.read_text(encoding="utf-8"))
    risk_payload = json.loads(risk_path.read_text(encoding="utf-8"))
    optimized = constraint_payload["risk_optimized"]
    constraints = optimized["constraints"]
    signal_payload = json.loads(signal_path.read_text(encoding="utf-8"))
    optimization_summary = json.loads(
        optimization_summary_path.read_text(encoding="utf-8")
    )
    solver = optimized["solver"]
    risk = risk_payload["risk_optimized"]
    slacks = constraints.get("constraint_slacks", {})
    row = {
        "date": date,
        "solver_backend": solver.get("backend"),
        "solver_name": solver.get("solver"),
        "objective_mode": solver.get("objective_mode"),
        "solver_iterations": solver.get("iterations"),
        "objective_value": solver.get("objective_value"),
        "primary_signal_utility": optimization_summary.get("primary_signal_utility"),
        "signal_utility_floor": optimization_summary.get("signal_utility_floor"),
        "signal_utility_solver_floor": optimization_summary.get(
            "signal_utility_solver_floor"
        ),
        "signal_utility_tolerance": optimization_summary.get(
            "signal_utility_tolerance"
        ),
        "primary_optimality_gap": optimization_summary.get("primary_optimality_gap"),
        "secondary_optimality_gap": optimization_summary.get("secondary_optimality_gap"),
        "final_signal_utility": optimization_summary.get("final_signal_utility"),
        "signal_capture_ratio": optimization_summary.get("signal_capture_ratio"),
        "minimum_signal_capture": optimization_summary.get("minimum_signal_capture"),
        "estimated_transaction_cost": optimization_summary.get("estimated_transaction_cost"),
        "turnover_saved": optimization_summary.get("turnover_saved"),
        "risk_form": optimization_summary.get("risk_form"),
        "backend_fallback_used": bool(solver.get("backend_fallback_used", False)),
        "binding_constraints": json.dumps(
            optimization_summary.get("binding_constraints", []), sort_keys=True
        ),
        "constraints_passed": constraints["passed"],
        "violation_count": len(constraints["violations"]),
        "binding_constraint_count": len(constraints["binding_constraints"]),
        "one_way_turnover": constraints["one_way_turnover"],
        "tracking_error": constraints["tracking_error"],
        "turnover_slack": slacks.get("max_turnover"),
        "tracking_error_slack": slacks.get("max_tracking_error"),
        "max_weight_slack": slacks.get("max_weight"),
        "max_active_weight_slack": slacks.get("max_active_weight"),
        "candidate_weight_lower_slack": slacks.get("candidate_weight_lower"),
        "candidate_weight_upper_slack": slacks.get("candidate_weight_upper"),
        "expected_return": risk["expected_return"],
        "active_volatility": risk["active_volatility"],
        "calibration_asset_count": signal_payload["calibration_asset_count"],
        "candidate_asset_count": signal_payload["candidate_asset_count"],
        "optimization_asset_count": signal_payload["optimization_asset_count"],
        "prediction_coverage": signal_payload["optimization_prediction_coverage"],
        "candidate_target_weight": signal_payload["portfolio_candidate_weight"][
            "risk_optimized"
        ],
    }
    exposure_rows: list[dict[str, Any]] = []
    for kind, values in (
        ("industry", constraints["industry_exposures"]),
        ("style", constraints["style_exposures"]),
    ):
        for name, detail in values.items():
            exposure_rows.append(
                {
                    "date": date,
                    "exposure_type": kind,
                    "name": name,
                    **detail,
                }
            )
    return row, exposure_rows


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_source_hash() -> str:
    root = Path(__file__).resolve().parent
    return _canonical_hash(
        {path.name: sha256_file(path) for path in sorted(root.glob("*.py"))}
    )


def _optional_covariance_path(root: str | Path | None, date: str) -> Path | None:
    if root is None:
        return None
    try:
        return resolve_covariance_path(root, date)
    except InputDataError:
        return None


def _optional_exposure_path(root: str | Path | None, date: str) -> Path | None:
    if root is None:
        return None
    try:
        return resolve_exposure_path(root, date)
    except InputDataError:
        return None


def _checkpoint_directory(
    root: Path, date: str, signature: str
) -> Path:
    return root / f"date={date}" / f"run={signature[:16]}"


def _checkpoint_is_complete(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    required = [*SINGLE_DATE_OUTPUT_FILES, "checkpoint_manifest.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise InputDataError(f"incomplete rolling checkpoint at {path}; missing {missing}")
    try:
        payload = json.loads(
            (path / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise InputDataError(f"invalid rolling checkpoint at {path}: {exc}") from exc
    if payload.get("status") != "success" or payload.get("signature") != signature:
        raise InputDataError(f"stale rolling checkpoint at {path}")
    return True


def run_rolling_experiment(
    *,
    config_path: str | Path,
    signal_file: str | Path,
    candidate_file: str | Path | None = None,
    benchmark_file: str | Path,
    asset_returns_file: str | Path,
    output_dir: str | Path,
    covariance_root: str | Path | None = None,
    sector_file: str | Path | None = None,
    exposure_file: str | Path | None = None,
    exposure_root: str | Path | None = None,
    factor_covariance_root: str | Path | None = None,
    specific_variance_root: str | Path | None = None,
    tradability_file: str | Path | None = None,
    initial_weights_file: str | Path | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    rebalance_every: int = 1,
    transaction_cost_bps: float | None = None,
    risk_model_config: str | Path | None = None,
    risk_returns_file: str | Path | None = None,
    risk_market_cap_file: str | Path | None = None,
    risk_industry_file: str | Path | None = None,
    dynamic_risk_cache_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    if exposure_file is not None and exposure_root is not None:
        raise InputDataError("configure only one of exposure_file and exposure_root")
    dynamic_required = {
        "risk_model_config": risk_model_config,
        "risk_returns_file": risk_returns_file,
        "risk_market_cap_file": risk_market_cap_file,
        "dynamic_risk_cache_root": dynamic_risk_cache_root,
    }
    supplied_dynamic = [name for name, value in dynamic_required.items() if value is not None]
    if supplied_dynamic and len(supplied_dynamic) != len(dynamic_required):
        missing = [name for name, value in dynamic_required.items() if value is None]
        raise InputDataError(
            "dynamic risk arguments must be supplied together; missing " + ", ".join(missing)
        )
    if risk_industry_file is not None and not supplied_dynamic:
        raise InputDataError("risk_industry_file requires complete dynamic risk arguments")
    portfolio_config = load_config(config_path)
    risk_form = portfolio_config["covariance"]["risk_form"]
    effective_cost_bps, cost_resolution = resolve_linear_cost_bps(
        portfolio_config, transaction_cost_bps
    )
    dynamic_enabled = len(supplied_dynamic) == len(dynamic_required)
    if not dynamic_enabled and risk_form == "asset_covariance" and covariance_root is None:
        raise InputDataError("covariance_root is required for asset_covariance risk")
    if not dynamic_enabled and risk_form == "factor_model":
        if exposure_file is None and exposure_root is None:
            raise InputDataError("factor_model risk requires exposure_file or exposure_root")
        if factor_covariance_root is None or specific_variance_root is None:
            raise InputDataError(
                "factor_model risk requires factor_covariance_root and "
                "specific_variance_root"
            )
    dynamic_cache = None
    if dynamic_enabled:
        dynamic_cache = DynamicRiskModelCache(
            config_path=risk_model_config,
            returns_file=risk_returns_file,
            market_cap_file=risk_market_cap_file,
            industry_file=risk_industry_file,
            cache_root=dynamic_risk_cache_root,
        )
    tolerance = float(portfolio_config["constraints"]["weight_sum_tolerance"])
    date_source = signal_file if candidate_file is None else candidate_file
    dates = _select_rebalance_dates(
        date_source,
        start_date=start_date,
        end_date=end_date,
        rebalance_every=rebalance_every,
    )
    asset_returns = load_asset_returns(str(asset_returns_file))
    if end_date is not None:
        asset_returns = asset_returns.loc[
            asset_returns["date"].le(normalize_date(end_date))
        ].copy()
    returns_wide = asset_returns.pivot(
        index="date", columns="ticker", values="return"
    ).sort_index()
    calendar = returns_wide.index.astype(str).tolist()
    positions = {date: position for position, date in enumerate(calendar)}
    absent_dates = [date for date in dates if date not in positions]
    if absent_dates:
        raise InputDataError(f"rebalance dates are absent from returns: {absent_dates[:10]}")
    if positions[dates[-1]] + 1 >= len(calendar):
        raise InputDataError("last rebalance date has no following execution date")

    initial = _initial_weights(dates[0], benchmark_file, initial_weights_file)
    current = initial.copy()
    previous_target: pd.Series | None = None
    previous_date: str | None = None
    weight_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    covariance_inputs: list[dict[str, str]] = []
    exposure_inputs: list[dict[str, str]] = []
    factor_risk_inputs: list[dict[str, str]] = []
    risk_resolutions: list[dict[str, Any]] = []
    checkpoint_reused_count = 0
    checkpoint_built_count = 0
    destination, temporary = _prepare_output_directory(output_dir)
    working = Path(tempfile.mkdtemp(prefix=".rolling-work-", dir=destination.parent))
    checkpoint_path = (
        None if checkpoint_root is None else Path(checkpoint_root).expanduser().resolve()
    )
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    runtime_source_sha256 = _runtime_source_hash()
    base_signature_inputs = {
        "implementation_version": __version__,
        "runtime_source_sha256": runtime_source_sha256,
        "config_sha256": sha256_file(config_path),
        "signal_sha256": sha256_file(signal_file),
        "candidate_sha256": (
            None if candidate_file is None else sha256_file(candidate_file)
        ),
        "benchmark_sha256": sha256_file(benchmark_file),
        "tradability_sha256": (
            None if tradability_file is None else sha256_file(tradability_file)
        ),
        "sector_sha256": None if sector_file is None else sha256_file(sector_file),
        "exposure_file_sha256": (
            None if exposure_file is None else sha256_file(exposure_file)
        ),
    }
    try:
        for position, date in enumerate(dates, start=1):
            if previous_target is not None and previous_date is not None:
                drift_dates = calendar[positions[previous_date] + 1 : positions[date] + 1]
                current = drift_weights(previous_target, returns_wide, drift_dates)
            current_path = working / f"current-{date}.parquet"
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": current.index,
                    "current_weight": current.to_numpy(dtype=float),
                }
            ).to_parquet(current_path, index=False)
            risk_source = "static"
            risk_fingerprint = ""
            risk_asset_count: int | None = None
            covariance_path: Path | None = None
            factor_covariance_path: Path | None = None
            specific_variance_path: Path | None = None
            if dynamic_cache is not None:
                candidates = (
                    load_signal(signal_file, date).index
                    if candidate_file is None
                    else load_candidate_universe(candidate_file, date)
                )
                benchmark = load_weight_series(
                    benchmark_file, date, "benchmark_weight", "benchmark"
                )
                universe = build_optimization_universe(
                    candidates, benchmark, current, tolerance
                )
                dynamic_cache.validate_positive_current_holdings(
                    date, current, tolerance
                )
                if tradability_file is not None:
                    load_tradability(tradability_file, date, universe)
                static_covariance = _optional_covariance_path(covariance_root, date)
                if exposure_file is not None:
                    static_exposure = Path(exposure_file).expanduser().resolve()
                else:
                    static_exposure = _optional_exposure_path(exposure_root, date)
                resolved = dynamic_cache.resolve(
                    date=date,
                    universe=universe,
                    static_covariance_file=static_covariance,
                    static_exposure_file=static_exposure,
                )
                date_exposure_file = resolved.exposure_file
                if risk_form == "asset_covariance":
                    covariance_path = resolved.covariance_file
                else:
                    factor_covariance_path = resolved.factor_covariance_file
                    specific_variance_path = resolved.specific_variance_file
                risk_source = resolved.source
                risk_fingerprint = resolved.fingerprint
                risk_asset_count = resolved.asset_count
            else:
                date_exposure_file = exposure_file
                if exposure_root is not None:
                    date_exposure_file = resolve_exposure_path(exposure_root, date)
                if risk_form == "asset_covariance":
                    covariance_path = resolve_covariance_path(covariance_root, date)
                else:
                    factor_covariance_path = resolve_factor_covariance_path(
                        factor_covariance_root, date
                    )
                    specific_variance_path = resolve_specific_variance_path(
                        specific_variance_root, date
                    )
                risk_fingerprint = _canonical_hash(
                    {
                        "risk_form": risk_form,
                        "covariance": (
                            None if covariance_path is None else sha256_file(covariance_path)
                        ),
                        "factor_covariance": (
                            None
                            if factor_covariance_path is None
                            else sha256_file(factor_covariance_path)
                        ),
                        "specific_variance": (
                            None
                            if specific_variance_path is None
                            else sha256_file(specific_variance_path)
                        ),
                        "exposure": (
                            None
                            if date_exposure_file is None
                            else sha256_file(date_exposure_file)
                        ),
                    }
                )
            if covariance_path is not None:
                covariance_inputs.append(
                    {
                        "date": date,
                        "path": str(covariance_path),
                        "sha256": sha256_file(covariance_path),
                        "source": risk_source,
                    }
                )
            if date_exposure_file is not None:
                exposure_inputs.append(
                    {
                        "date": date,
                        "path": str(date_exposure_file),
                        "sha256": sha256_file(date_exposure_file),
                    }
                )
            if factor_covariance_path is not None and specific_variance_path is not None:
                factor_risk_inputs.append(
                    {
                        "date": date,
                        "factor_covariance_path": str(factor_covariance_path),
                        "factor_covariance_sha256": sha256_file(factor_covariance_path),
                        "specific_variance_path": str(specific_variance_path),
                        "specific_variance_sha256": sha256_file(specific_variance_path),
                        "source": risk_source,
                    }
                )
            risk_resolutions.append(
                {
                    "date": date,
                    "source": risk_source,
                    "fingerprint": risk_fingerprint,
                    "asset_count": risk_asset_count,
                    "risk_form": risk_form,
                    "covariance_file": (
                        None if covariance_path is None else str(covariance_path)
                    ),
                    "factor_covariance_file": (
                        None if factor_covariance_path is None else str(factor_covariance_path)
                    ),
                    "specific_variance_file": (
                        None if specific_variance_path is None else str(specific_variance_path)
                    ),
                    "exposure_file": (
                        None if date_exposure_file is None else str(date_exposure_file)
                    ),
                }
            )
            signature_payload = {
                **base_signature_inputs,
                "date": date,
                "current_weights_sha256": sha256_file(current_path),
                "risk_fingerprint": risk_fingerprint,
                "risk_form": risk_form,
                "covariance_sha256": (
                    None if covariance_path is None else sha256_file(covariance_path)
                ),
                "factor_covariance_sha256": (
                    None
                    if factor_covariance_path is None
                    else sha256_file(factor_covariance_path)
                ),
                "specific_variance_sha256": (
                    None
                    if specific_variance_path is None
                    else sha256_file(specific_variance_path)
                ),
                "exposure_sha256": (
                    None
                    if date_exposure_file is None
                    else sha256_file(date_exposure_file)
                ),
            }
            signature = _canonical_hash(signature_payload)
            if checkpoint_path is None:
                date_output = working / f"date={date}"
                reusable = False
            else:
                date_output = _checkpoint_directory(checkpoint_path, date, signature)
                reusable = _checkpoint_is_complete(date_output, signature)
            if reusable:
                checkpoint_reused_count += 1
            else:
                if checkpoint_path is None:
                    build_output = date_output
                else:
                    date_output.parent.mkdir(parents=True, exist_ok=True)
                    build_output = Path(
                        tempfile.mkdtemp(
                            prefix=f".{date_output.name}.tmp-", dir=date_output.parent
                        )
                    )
                try:
                    run_single_date(
                        config_path=config_path,
                        signal_file=signal_file,
                        candidate_file=candidate_file,
                        covariance_file=covariance_path,
                        factor_covariance_file=factor_covariance_path,
                        specific_variance_file=specific_variance_path,
                        transaction_cost_bps=transaction_cost_bps,
                        benchmark_file=benchmark_file,
                        current_weights_file=current_path,
                        sector_file=sector_file,
                        exposure_file=date_exposure_file,
                        tradability_file=tradability_file,
                        requested_date=date,
                        output_dir=build_output,
                    )
                    if checkpoint_path is not None:
                        _write_json(
                            build_output / "checkpoint_manifest.json",
                            {
                                "status": "success",
                                "date": date,
                                "signature": signature,
                                "signature_inputs": signature_payload,
                                "completed_at": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(
                (datetime.now(timezone.utc) - started).total_seconds()
            ),
                            },
                        )
                        os.replace(build_output, date_output)
                    checkpoint_built_count += 1
                except BaseException:
                    if checkpoint_path is not None and build_output.exists():
                        shutil.rmtree(build_output)
                    raise
            weights = pd.read_parquet(date_output / "target_weights.parquet")
            weight_frames.append(weights)
            previous_target = weights.loc[
                weights["portfolio"].eq("risk_optimized")
            ].set_index("ticker")["target_weight"].astype(float)
            previous_date = date
            diagnostic, exposures = _diagnostic_rows(
                date,
                date_output / "constraint_diagnostics.json",
                date_output / "risk_summary.json",
                date_output / "signal_diagnostics.json",
                date_output / "optimization_summary.json",
            )
            diagnostic_rows.append(diagnostic)
            exposure_rows.extend(exposures)
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "stage": "portfolio_optimization",
                        "date": date,
                        "position": position,
                        "total": len(dates),
                        "risk_source": risk_source,
                        "checkpoint_reused": reusable,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

        rebalance_weights = pd.concat(weight_frames, ignore_index=True)
        benchmark_targets: list[pd.DataFrame] = []
        for date in dates:
            benchmark = load_weight_series(
                benchmark_file, date, "benchmark_weight", "benchmark"
            )
            benchmark_targets.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "ticker": benchmark.index,
                        "portfolio": "benchmark",
                        "target_weight": benchmark.to_numpy(dtype=float),
                    }
                )
            )
        target_contract = pd.concat(
            [
                rebalance_weights.loc[
                    :, ["date", "ticker", "portfolio", "target_weight"]
                ],
                *benchmark_targets,
            ],
            ignore_index=True,
        )
        initial_map = {
            "risk_optimized": initial,
            "equal_weight_signal": initial,
            "benchmark": initial,
        }
        daily = add_benchmark_relative_performance(
            backtest_targets(
                target_contract,
                asset_returns,
                transaction_cost_bps=effective_cost_bps,
                initial_weights=initial_map,
            )
        )
        metrics = summarize_backtest(daily)
        diagnostics = pd.DataFrame(diagnostic_rows)
        exposures = pd.DataFrame(exposure_rows)
        if exposures.empty:
            exposures = pd.DataFrame(
                columns=[
                    "date",
                    "exposure_type",
                    "name",
                    "benchmark_exposure",
                    "portfolio_exposure",
                    "active_exposure",
                    "lower_bound",
                    "upper_bound",
                    "lower_slack",
                    "upper_slack",
                    "binding",
                    "passed",
                ]
            )
        binding_counts: dict[str, int] = {}
        for encoded in diagnostics.get("binding_constraints", pd.Series(dtype=str)).dropna():
            for name in json.loads(encoded):
                binding_counts[str(name)] = binding_counts.get(str(name), 0) + 1
        capture_values = pd.to_numeric(
            diagnostics.get("signal_capture_ratio", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        utility_loss_values = (
            pd.to_numeric(diagnostics["primary_signal_utility"], errors="coerce")
            - pd.to_numeric(diagnostics["final_signal_utility"], errors="coerce")
        ).dropna()
        rolling_optimization_summary = {
            "objective_mode": portfolio_config["optimizer"]["objective_mode"],
            "risk_form": risk_form,
            "solver_backends": sorted(
                diagnostics["solver_backend"].dropna().astype(str).unique().tolist()
            ),
            "rebalance_count": len(dates),
            "all_constraints_passed": bool(diagnostics["constraints_passed"].all()),
            "fallback_count": int(diagnostics["backend_fallback_used"].sum()),
            "binding_constraint_counts": binding_counts,
            "binding_constraint_ratios": {
                name: float(count / len(dates)) for name, count in binding_counts.items()
            },
            "signal_capture_ratio": {
                "minimum": None if capture_values.empty else float(capture_values.min()),
                "mean": None if capture_values.empty else float(capture_values.mean()),
                "maximum": None if capture_values.empty else float(capture_values.max()),
            },
            "signal_utility_loss": {
                "total": (
                    None if utility_loss_values.empty else float(utility_loss_values.sum())
                ),
                "mean": (
                    None if utility_loss_values.empty else float(utility_loss_values.mean())
                ),
            },
            "runtime_seconds": float(
                (datetime.now(timezone.utc) - started).total_seconds()
            ),
            "total_one_way_turnover_at_rebalances": float(
                pd.to_numeric(diagnostics["one_way_turnover"], errors="coerce").sum()
            ),
            "total_estimated_transaction_cost": float(
                pd.to_numeric(
                    diagnostics["estimated_transaction_cost"], errors="coerce"
                ).sum()
            ),
            "total_turnover_saved": float(
                pd.to_numeric(diagnostics["turnover_saved"], errors="coerce").sum()
            ),
            "cost_model": cost_resolution,
            "risk_cache": (
                {"enabled": False}
                if dynamic_cache is None
                else {"enabled": True, **dynamic_cache.statistics()}
            ),
        }
        rebalance_weights.to_parquet(
            temporary / "rebalance_weights.parquet", index=False
        )
        daily.to_parquet(temporary / "daily_performance.parquet", index=False)
        exposures.to_parquet(temporary / "exposure_timeseries.parquet", index=False)
        diagnostics.to_parquet(
            temporary / "optimization_diagnostics.parquet", index=False
        )
        _write_json(temporary / "portfolio_metrics.json", metrics)
        _write_json(
            temporary / "optimization_summary.json", rolling_optimization_summary
        )

        supplied_paths = {
            "config": config_path,
            "signal": signal_file,
            "candidates": candidate_file,
            "benchmark": benchmark_file,
            "asset_returns": asset_returns_file,
            "sectors": sector_file,
            "exposures": exposure_file,
            "tradability": tradability_file,
            "initial_weights": initial_weights_file,
        }
        manifest_inputs = {
            name: {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": sha256_file(Path(path).expanduser().resolve()),
            }
            for name, path in supplied_paths.items()
            if path is not None
        }
        manifest = {
            "schema_version": 4,
            "implementation_version": __version__,
            "runtime_source_sha256": runtime_source_sha256,
            "status": "success",
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "rebalance_dates": dates,
            "rebalance_count": len(dates),
            "transaction_cost_bps": float(effective_cost_bps),
            "cost_model_resolution": cost_resolution,
            "risk_form": risk_form,
            "execution_timing": (
                "target formed on rebalance date t and applied before the asset return "
                "on the next available trading date"
            ),
            "benchmark_method": "simulated_from_supplied_rebalance_weights",
            "inputs": manifest_inputs,
            "covariance_inputs": covariance_inputs,
            "exposure_inputs": exposure_inputs,
            "factor_risk_inputs": factor_risk_inputs,
            "risk_resolutions": risk_resolutions,
            "dynamic_risk": (
                {"enabled": False}
                if dynamic_cache is None
                else {"enabled": True, **dynamic_cache.statistics()}
            ),
            "checkpoints": {
                "root": None if checkpoint_path is None else str(checkpoint_path),
                "reused_count": checkpoint_reused_count,
                "built_count": checkpoint_built_count,
            },
            "outputs": list(ROLLING_OUTPUT_FILES),
        }
        manifest["output_sha256"] = {
            name: sha256_file(temporary / name)
            for name in ROLLING_OUTPUT_FILES
            if name != "rolling_manifest.json" and (temporary / name).is_file()
        }
        _write_json(temporary / "rolling_manifest.json", manifest)
        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if working.exists():
            shutil.rmtree(working)
    return {
        "status": "success",
        "rebalance_count": len(dates),
        "date_start": dates[0],
        "date_end": dates[-1],
        "output_dir": str(destination),
        "portfolios": metrics["portfolios"],
    }
