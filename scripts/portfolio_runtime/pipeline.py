from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import load_config
from .cost import resolve_linear_cost_bps
from .diagnostics import constraint_report, portfolio_metrics
from .errors import InputDataError
from .io import (
    DateTableCache,
    load_candidate_universe,
    load_covariance,
    load_exposures,
    load_labels,
    load_signal,
    load_weight_series,
    normalize_date,
    normalize_ticker,
    resolve_tradability,
    sha256_file,
)
from .optimizer import build_equal_weight_baseline, optimize_portfolio
from .risk import PortfolioRisk, load_factor_risk
from .signal import calibrate_signal


OUTPUT_FILES = (
    "target_weights.parquet",
    "constraint_diagnostics.json",
    "risk_summary.json",
    "signal_diagnostics.json",
    "run_manifest.json",
    "optimization_summary.json",
)


def build_optimization_universe(
    candidates: pd.Series | pd.Index,
    benchmark: pd.Series,
    current: pd.Series | None,
    tolerance: float,
) -> pd.Index:
    """Return the exact ticker set that the optimizer must cover."""
    candidate_names = (
        candidates.index if isinstance(candidates, pd.Series) else candidates
    )
    universe_names = set(candidate_names)
    universe_names.update(benchmark[benchmark.abs() > tolerance].index)
    if current is not None:
        universe_names.update(current[current.abs() > tolerance].index)
    return pd.Index(sorted(universe_names), name="ticker")


def resolve_optimization_tradability(
    *,
    candidates: pd.Index,
    benchmark: pd.Series,
    current: pd.Series | None,
    tolerance: float,
    tradability_file: str | Path | None,
    requested_date: str,
    missing_security_policy: str,
    table_cache: DateTableCache | None = None,
) -> tuple[pd.Index, pd.Index, pd.Series, pd.Index, pd.Index]:
    """Resolve an auditable universe without treating missing rows as tradable."""
    if missing_security_policy not in {"error", "freeze_last"}:
        raise InputDataError(
            "missing_security_policy must be error or freeze_last"
        )
    resolved_candidates = pd.Index(candidates, name="ticker")
    universe = build_optimization_universe(
        resolved_candidates, benchmark, current, tolerance
    )
    if tradability_file is None:
        tradable = pd.Series(True, index=universe, name="tradable", dtype=bool)
        empty = pd.Index([], name="ticker")
        return resolved_candidates, universe, tradable, empty, empty

    tradable, missing = resolve_tradability(
        tradability_file,
        requested_date,
        universe,
        missing_security_policy=missing_security_policy,
        table_cache=table_cache,
    )
    excluded_candidates = resolved_candidates.intersection(missing)
    if len(excluded_candidates):
        resolved_candidates = resolved_candidates.difference(excluded_candidates)
        if resolved_candidates.empty:
            raise InputDataError(
                "missing tradability records leave no eligible candidates on "
                f"{requested_date}"
            )
        universe = build_optimization_universe(
            resolved_candidates, benchmark, current, tolerance
        )
        tradable, missing = resolve_tradability(
            tradability_file,
            requested_date,
            universe,
            missing_security_policy=missing_security_policy,
            table_cache=table_cache,
        )
    return resolved_candidates, universe, tradable, missing, excluded_candidates


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _ticker_index_hash(values: pd.Index) -> str:
    encoded = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_weight_vector(
    weights: pd.Series,
    label: str,
    *,
    require_full_investment: bool,
    tolerance: float,
) -> None:
    if (weights < -tolerance).any():
        raise InputDataError(f"{label} contains negative weights")
    total = float(weights.sum())
    if require_full_investment and abs(total - 1.0) > tolerance:
        raise InputDataError(f"{label} weights sum to {total:.12g}, expected 1")
    if not require_full_investment and total > 1.0 + tolerance:
        raise InputDataError(f"{label} weights sum to more than one: {total:.12g}")


def _prepare_output_directory(output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise InputDataError(f"output path exists and is not a directory: {destination}")
        entries = list(destination.iterdir())
        if entries:
            raise InputDataError(f"output directory must be new or empty: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    return destination, temporary


def _publish_output(destination: Path, temporary: Path) -> None:
    if destination.exists():
        destination.rmdir()
    os.replace(temporary, destination)



def resolve_missing_prediction_roles(
    calibrated: pd.DataFrame,
    *,
    universe: pd.Index,
    candidates: pd.Index,
    benchmark: pd.Series,
    current: pd.Series | None,
    tradable: pd.Series,
    policy: str,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Resolve missing predictions by portfolio role without inventing alpha."""
    missing = calibrated["signal_score"].isna()
    candidate_mask = pd.Series(
        universe.isin(candidates), index=universe, name="is_candidate", dtype=bool
    )
    held = (
        pd.Series(False, index=universe, dtype=bool)
        if current is None
        else current.gt(tolerance)
    )
    frozen_missing = missing & ~tradable
    exit_only = missing & tradable & held & ~candidate_mask
    benchmark_only = (
        missing
        & tradable
        & ~held
        & ~candidate_mask
        & benchmark.abs().gt(tolerance)
    )

    if policy == "neutral":
        invalid = pd.Series(False, index=universe, dtype=bool)
        exit_only = pd.Series(False, index=universe, name="exit_only", dtype=bool)
    elif policy == "error_except_frozen":
        invalid = missing & ~frozen_missing
        exit_only = pd.Series(False, index=universe, name="exit_only", dtype=bool)
    elif policy == "role_aware":
        invalid = (missing & candidate_mask) | (
            missing
            & ~candidate_mask
            & ~(frozen_missing | exit_only | benchmark_only)
        )
    else:
        raise InputDataError(f"unsupported missing prediction policy: {policy}")

    if invalid.any():
        candidate_missing = invalid & candidate_mask
        label = (
            "candidate ticker(s)"
            if candidate_missing.any()
            else "optimization ticker(s)"
        )
        examples = universe[candidate_missing if candidate_missing.any() else invalid]
        raise InputDataError(
            f"{label} missing full-universe prediction: {list(examples[:10])}"
        )

    resolved = calibrated.copy()
    resolved["signal_score"] = resolved["signal_score"].fillna(0.0)
    resolved["expected_return"] = resolved["expected_return"].fillna(0.0)
    exit_only = exit_only.rename("exit_only")
    diagnostics = {
        "missing_prediction_count": int(missing.sum()),
        "allowed_frozen_missing_prediction_count": int(frozen_missing.sum()),
        "exit_only_missing_prediction_count": int(exit_only.sum()),
        "benchmark_only_neutral_prediction_count": int(benchmark_only.sum()),
        "exit_only_tickers": universe[exit_only].tolist(),
        "benchmark_only_neutral_tickers": universe[benchmark_only].tolist(),
    }
    return resolved, exit_only, diagnostics

def run_single_date(
    *,
    config_path: str | Path,
    signal_file: str | Path,
    candidate_file: str | Path | None = None,
    candidate_universe: pd.Index | None = None,
    covariance_file: str | Path | None,
    benchmark_file: str | Path,
    requested_date: object,
    output_dir: str | Path,
    current_weights_file: str | Path | None = None,
    sector_file: str | Path | None = None,
    exposure_file: str | Path | None = None,
    factor_covariance_file: str | Path | None = None,
    specific_variance_file: str | Path | None = None,
    transaction_cost_bps: float | None = None,
    tradability_file: str | Path | None = None,
    missing_security_policy: str = "error",
    table_cache: DateTableCache | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    date = normalize_date(requested_date)
    config = load_config(config_path)
    effective_cost_bps, cost_resolution = resolve_linear_cost_bps(
        config, transaction_cost_bps
    )
    signal = load_signal(signal_file, date, table_cache=table_cache)
    calibrated_signal, signal_diagnostics = calibrate_signal(
        signal, config["signal"], date
    )
    tolerance = float(config["constraints"]["weight_sum_tolerance"])
    source_candidates = (
        pd.Index(signal.index, name="ticker")
        if candidate_file is None
        else load_candidate_universe(candidate_file, date, table_cache=table_cache)
    )
    if candidate_universe is None:
        candidates = source_candidates
    else:
        candidates = pd.Index(
            sorted({normalize_ticker(value) for value in candidate_universe}),
            name="ticker",
        )
        if candidates.empty:
            raise InputDataError("candidate universe override cannot be empty")
        outside_source = candidates.difference(source_candidates)
        if len(outside_source):
            raise InputDataError(
                "candidate universe override contains ticker(s) outside the source "
                f"candidate universe: {list(outside_source[:10])}"
            )
    missing_candidates = candidates.difference(signal.index)
    if len(missing_candidates):
        raise InputDataError(
            "candidate universe contains ticker(s) missing from full signal: "
            f"{list(missing_candidates[:10])}"
        )

    benchmark_input = load_weight_series(
        benchmark_file, date, "benchmark_weight", "benchmark", table_cache=table_cache
    )
    _validate_weight_vector(
        benchmark_input,
        "benchmark",
        require_full_investment=True,
        tolerance=tolerance,
    )

    current_input = None
    if current_weights_file is not None:
        current_input = load_weight_series(
            current_weights_file, date, "current_weight", "current weights",
            table_cache=table_cache,
        )
        _validate_weight_vector(
            current_input,
            "current",
            require_full_investment=True,
            tolerance=tolerance,
        )

    (
        candidates,
        universe,
        tradable,
        synthetic_nontradable,
        excluded_missing_candidates,
    ) = resolve_optimization_tradability(
        candidates=candidates,
        benchmark=benchmark_input,
        current=current_input,
        tolerance=tolerance,
        tradability_file=tradability_file,
        requested_date=date,
        missing_security_policy=missing_security_policy,
        table_cache=table_cache,
    )
    benchmark = benchmark_input.reindex(universe, fill_value=0.0)
    current = (
        None if current_input is None else current_input.reindex(universe, fill_value=0.0)
    )
    calibrated = calibrated_signal.reindex(universe)

    industry_constraint = (
        config["constraints"]["sector_active_limit"] is not None
        or config["constraints"]["industry_active_range"] is not None
    )
    if industry_constraint and sector_file is None:
        raise InputDataError(
            "sector_file is required when an industry constraint is configured"
        )
    sectors = None if sector_file is None else load_labels(
        sector_file, date, "sector", "sectors", universe, table_cache=table_cache
    )

    style_constraint = (
        config["constraints"]["factor_active_limit"] is not None
        or config["constraints"]["style_active_ranges"] is not None
    )
    if style_constraint and exposure_file is None:
        raise InputDataError(
            "exposure_file is required when a style constraint is configured"
        )
    required_factors = None
    if config["constraints"]["style_active_ranges"] is not None:
        required_factors = {
            name
            for name, specification in config["constraints"]["style_active_ranges"].items()
            if specification.get("enabled", False)
        }
    exposures = None if exposure_file is None else load_exposures(
        exposure_file, date, universe, required_factors, table_cache=table_cache
    )

    if (~tradable).any() and current is None:
        raise InputDataError("current_weights_file is required for non-tradable assets")

    missing_prediction = calibrated["signal_score"].isna()
    candidate_mask = pd.Series(
        universe.isin(candidates), index=universe, name="is_candidate", dtype=bool
    )
    policy = config["signal"]["missing_prediction_policy"]
    calibrated, exit_only_mask, missing_prediction_diagnostics = (
        resolve_missing_prediction_roles(
            calibrated,
            universe=universe,
            candidates=candidates,
            benchmark=benchmark,
            current=current,
            tradable=tradable,
            policy=policy,
            tolerance=tolerance,
        )
    )

    risk_form = config["covariance"]["risk_form"]
    if risk_form == "asset_covariance":
        if covariance_file is None:
            raise InputDataError("covariance_file is required for asset_covariance risk")
        if factor_covariance_file is not None or specific_variance_file is not None:
            raise InputDataError("dense and factor-form risk inputs are mutually exclusive")
        covariance, covariance_diagnostics = load_covariance(
            covariance_file,
            universe,
            annualized=config["covariance"]["annualized"],
            periods_per_year=config["covariance"]["periods_per_year"],
            eigenvalue_floor=config["covariance"]["eigenvalue_floor"],
            symmetry_tolerance=config["covariance"]["symmetry_tolerance"],
        )
        risk: pd.DataFrame | PortfolioRisk = covariance
        covariance_diagnostics = {
            "risk_form": "asset_covariance", **covariance_diagnostics
        }
    else:
        if covariance_file is not None:
            raise InputDataError("dense and factor-form risk inputs are mutually exclusive")
        if exposure_file is None or factor_covariance_file is None or specific_variance_file is None:
            raise InputDataError(
                "factor_model risk requires exposure_file, factor_covariance_file, "
                "and specific_variance_file"
            )
        risk = load_factor_risk(
            exposure_file=str(exposure_file),
            factor_covariance_file=str(factor_covariance_file),
            specific_variance_file=str(specific_variance_file),
            universe=universe,
            requested_date=date,
            annualized=config["covariance"]["annualized"],
            periods_per_year=config["covariance"]["periods_per_year"],
            eigenvalue_floor=config["covariance"]["eigenvalue_floor"],
            symmetry_tolerance=config["covariance"]["symmetry_tolerance"],
        )
        covariance_diagnostics = dict(risk.diagnostics or {})

    signal_baseline = build_equal_weight_baseline(
        calibrated_signal.loc[candidates, "signal_score"],
        config["baseline"]["top_n"],
    )
    baseline = signal_baseline.reindex(universe, fill_value=0.0)
    optimized = optimize_portfolio(
        calibrated["expected_return"],
        risk,
        benchmark,
        current,
        sectors,
        exposures,
        tradable,
        config["optimizer"],
        config["constraints"],
        signal_score=calibrated["signal_score"],
        anchor_weights=baseline,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
        cost_model={"linear_cost_bps": effective_cost_bps},
    )

    baseline_constraints = constraint_report(
        baseline,
        benchmark,
        current,
        risk,
        sectors,
        exposures,
        tradable,
        config["constraints"],
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    risk_baseline = portfolio_metrics(
        baseline, calibrated["expected_return"], risk, benchmark
    )
    risk_optimized = portfolio_metrics(
        optimized.weights, calibrated["expected_return"], risk, benchmark
    )
    risk_summary = {
        "covariance": covariance_diagnostics,
        "equal_weight_signal": risk_baseline,
        "risk_optimized": risk_optimized,
        "optimized_minus_equal_weight": {
            key: float(risk_optimized[key] - risk_baseline[key])
            for key in risk_baseline
        },
    }
    baseline_turnover = (
        None if current is None else float(0.5 * (baseline - current).abs().sum())
    )
    optimization_summary = {
        "objective_mode": config["optimizer"]["objective_mode"],
        "backend": optimized.solver.get("backend"),
        "risk_form": risk_form,
        "blend_strength": optimized.solver.get("blend_strength"),
        "anchor_asset_count": optimized.solver.get("anchor_asset_count"),
        "anchor_predicted_volatility": optimized.solver.get(
            "anchor_predicted_volatility"
        ),
        "minimum_variance_predicted_volatility": optimized.solver.get(
            "minimum_variance_predicted_volatility"
        ),
        "optimized_predicted_volatility": optimized.solver.get("optimized_predicted_volatility"),
        "predicted_risk_reduction": optimized.solver.get("predicted_risk_reduction"),
        "anchor_reallocation": optimized.solver.get("anchor_reallocation"),
        "maximum_anchor_weight_deviation": optimized.solver.get("maximum_anchor_weight_deviation"),
        "primary_signal_utility": optimized.solver.get("primary_signal_utility"),
        "signal_utility_floor": optimized.solver.get("signal_utility_floor"),
        "signal_utility_solver_floor": optimized.solver.get(
            "signal_utility_solver_floor"
        ),
        "signal_utility_tolerance": optimized.solver.get(
            "signal_utility_tolerance"
        ),
        "primary_optimality_gap": optimized.solver.get("primary_optimality_gap"),
        "secondary_optimality_gap": optimized.solver.get("secondary_optimality_gap"),
        "final_signal_utility": optimized.solver.get("final_signal_utility"),
        "signal_capture_ratio": optimized.solver.get("signal_capture_ratio"),
        "minimum_signal_capture": optimized.solver.get("minimum_signal_capture"),
        "one_way_turnover": optimized.solver.get("one_way_turnover"),
        "baseline_one_way_turnover": baseline_turnover,
        "turnover_saved": (
            None
            if baseline_turnover is None or optimized.solver.get("one_way_turnover") is None
            else float(baseline_turnover - optimized.solver["one_way_turnover"])
        ),
        "estimated_transaction_cost": optimized.solver.get("estimated_transaction_cost"),
        "cost_model": cost_resolution,
        "constraint_slacks": optimized.constraints.get("constraint_slacks", {}),
        "binding_constraints": optimized.constraints.get("binding_constraints", []),
        "constraint_duals": optimized.solver.get("constraint_duals"),
        "constraint_duals_unavailable_reason": optimized.solver.get(
            "constraint_duals_unavailable_reason"
        ),
        "solver_performance": {
            key: optimized.solver[key]
            for key in (
                "solver_wall_seconds",
                "problem_compile_seconds",
                "problem_cache_hit",
                "problem_cache_key",
                "problem_cache_size",
                "problem_is_dpp",
                "warm_start_requested",
                "risk_operator_rows",
                "tracking_error_constraint_form",
            )
            if key in optimized.solver
        },
        "constraints_passed": bool(optimized.constraints["passed"]),
    }
    constraint_diagnostics = {
        "risk_optimized": {
            "solver": optimized.solver,
            "constraints": optimized.constraints,
        },
        "equal_weight_signal": {
            "solver": None,
            "constraints": baseline_constraints,
        },
    }

    signal_diagnostics.update(
        {
            "calibration_asset_count": int(len(signal)),
            "candidate_asset_count": int(len(candidates)),
            "optimization_asset_count": int(len(universe)),
            "optimization_prediction_coverage": float(
                1.0 - missing_prediction.mean()
            ),
            "missing_prediction_policy": policy,
            "missing_security_policy": missing_security_policy,
            "synthetic_nontradable_asset_count": int(len(synthetic_nontradable)),
            "synthetic_nontradable_tickers": synthetic_nontradable.tolist(),
            "excluded_missing_candidate_count": int(
                len(excluded_missing_candidates)
            ),
            "excluded_missing_candidate_tickers": excluded_missing_candidates.tolist(),
            **missing_prediction_diagnostics,
            "portfolio_candidate_weight": {
                "equal_weight_signal": float(baseline[candidate_mask].sum()),
                "risk_optimized": float(optimized.weights[candidate_mask].sum()),
            },
        }
    )

    common = pd.DataFrame(
        {
            "date": date,
            "ticker": universe,
            "benchmark_weight": benchmark.to_numpy(dtype=float),
            "current_weight": (
                np.nan if current is None else current.to_numpy(dtype=float)
            ),
            "signal_available": ~missing_prediction.to_numpy(dtype=bool),
            "is_candidate": candidate_mask.to_numpy(dtype=bool),
            "exit_only": exit_only_mask.to_numpy(dtype=bool),
            "has_signal": universe.isin(signal.index),
            "raw_prediction": calibrated["raw_prediction"].to_numpy(dtype=float),
            "signal_score": calibrated["signal_score"].to_numpy(dtype=float),
            "expected_return": calibrated["expected_return"].to_numpy(dtype=float),
            "tradable": tradable.to_numpy(dtype=bool),
            "synthetic_nontradable": universe.isin(synthetic_nontradable),
        }
    )
    target_frames = []
    for name, weights in (
        ("equal_weight_signal", baseline),
        ("risk_optimized", optimized.weights),
    ):
        frame = common.copy()
        frame.insert(2, "portfolio", name)
        frame.insert(3, "target_weight", weights.to_numpy(dtype=float))
        target_frames.append(frame)
    target_weights = pd.concat(target_frames, ignore_index=True)

    supplied_paths = {
        "candidates": candidate_file,
        "config": config_path,
        "signal": signal_file,
        "covariance": covariance_file,
        "factor_covariance": factor_covariance_file,
        "specific_variance": specific_variance_file,
        "benchmark": benchmark_file,
        "current_weights": current_weights_file,
        "sectors": sector_file,
        "exposures": exposure_file,
        "tradability": tradability_file,
    }
    inputs = {}
    for name, path in supplied_paths.items():
        if path is None:
            continue
        resolved = Path(path).expanduser().resolve()
        inputs[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}

    completed = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 4,
        "implementation_version": __version__,
        "status": "success",
        "requested_date": date,
        "asset_count": int(len(universe)),
        "signal_asset_count": int(len(signal)),
        "candidate_asset_count": int(len(candidates)),
        "missing_security_resolution": {
            "policy": missing_security_policy,
            "synthetic_nontradable_asset_count": int(len(synthetic_nontradable)),
            "synthetic_nontradable_ticker_sha256": _ticker_index_hash(
                synthetic_nontradable
            ),
            "excluded_candidate_count": int(len(excluded_missing_candidates)),
            "excluded_candidate_ticker_sha256": _ticker_index_hash(
                excluded_missing_candidates
            ),
        },
        "candidate_universe_override": (
            None
            if candidate_universe is None
            else {
                "asset_count": int(len(candidates)),
                "source_asset_count": int(len(source_candidates)),
                "excluded_asset_count": int(
                    len(source_candidates.difference(candidates))
                ),
                "ticker_sha256": _ticker_index_hash(candidates),
            }
        ),
        "benchmark_or_current_only_asset_count": int(
            len(universe.difference(candidates))
        ),
        "benchmark_only_asset_count": int(
            len(universe.difference(signal.index))
        ),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "config": config,
        "cost_model_resolution": cost_resolution,
        "risk_form": risk_form,
        "inputs": inputs,
        "outputs": list(OUTPUT_FILES),
        "runtime": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "pyarrow": _package_version("pyarrow"),
            "pyyaml": _package_version("PyYAML"),
        },
    }

    destination, temporary = _prepare_output_directory(output_dir)
    try:
        target_weights.to_parquet(temporary / "target_weights.parquet", index=False)
        _write_json(temporary / "constraint_diagnostics.json", constraint_diagnostics)
        _write_json(temporary / "risk_summary.json", risk_summary)
        _write_json(temporary / "signal_diagnostics.json", signal_diagnostics)
        _write_json(temporary / "optimization_summary.json", optimization_summary)
        manifest["output_sha256"] = {
            name: sha256_file(temporary / name)
            for name in OUTPUT_FILES
            if name != "run_manifest.json" and (temporary / name).is_file()
        }
        _write_json(temporary / "run_manifest.json", manifest)
        _publish_output(destination, temporary)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "status": "success",
        "date": date,
        "asset_count": int(len(universe)),
        "output_dir": str(destination),
        "optimized_expected_return": risk_optimized["expected_return"],
        "optimized_active_volatility": risk_optimized["active_volatility"],
        "solver_iterations": optimized.solver["iterations"],
    }
