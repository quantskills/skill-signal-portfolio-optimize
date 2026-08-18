from __future__ import annotations

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
from .diagnostics import constraint_report, portfolio_metrics
from .errors import InputDataError
from .io import (
    load_candidate_universe,
    load_covariance,
    load_exposures,
    load_labels,
    load_signal,
    load_tradability,
    load_weight_series,
    normalize_date,
    sha256_file,
)
from .optimizer import build_equal_weight_baseline, optimize_portfolio
from .signal import calibrate_signal


OUTPUT_FILES = (
    "target_weights.parquet",
    "constraint_diagnostics.json",
    "risk_summary.json",
    "signal_diagnostics.json",
    "run_manifest.json",
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


def run_single_date(
    *,
    config_path: str | Path,
    signal_file: str | Path,
    candidate_file: str | Path | None = None,
    covariance_file: str | Path,
    benchmark_file: str | Path,
    requested_date: object,
    output_dir: str | Path,
    current_weights_file: str | Path | None = None,
    sector_file: str | Path | None = None,
    exposure_file: str | Path | None = None,
    tradability_file: str | Path | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    date = normalize_date(requested_date)
    config = load_config(config_path)
    signal = load_signal(signal_file, date)
    calibrated_signal, signal_diagnostics = calibrate_signal(
        signal, config["signal"], date
    )
    tolerance = float(config["constraints"]["weight_sum_tolerance"])
    candidates = (
        pd.Index(signal.index, name="ticker")
        if candidate_file is None
        else load_candidate_universe(candidate_file, date)
    )
    missing_candidates = candidates.difference(signal.index)
    if len(missing_candidates):
        raise InputDataError(
            "candidate universe contains ticker(s) missing from full signal: "
            f"{list(missing_candidates[:10])}"
        )

    benchmark_input = load_weight_series(
        benchmark_file, date, "benchmark_weight", "benchmark"
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
            current_weights_file, date, "current_weight", "current weights"
        )
        _validate_weight_vector(
            current_input,
            "current",
            require_full_investment=True,
            tolerance=tolerance,
        )

    universe = build_optimization_universe(
        candidates, benchmark_input, current_input, tolerance
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
        sector_file, date, "sector", "sectors", universe
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
        exposure_file, date, universe, required_factors
    )

    tradable = pd.Series(True, index=universe, name="tradable", dtype=bool)
    if tradability_file is not None:
        tradable = load_tradability(tradability_file, date, universe)
    if (~tradable).any() and current is None:
        raise InputDataError("current_weights_file is required for non-tradable assets")

    missing_prediction = calibrated["signal_score"].isna()
    allowed_frozen_missing = pd.Series(False, index=universe, dtype=bool)
    if current is not None:
        allowed_frozen_missing = missing_prediction & ~tradable & current.gt(tolerance)
    policy = config["signal"]["missing_prediction_policy"]
    if policy == "error_except_frozen":
        invalid_missing = missing_prediction & ~allowed_frozen_missing
        if invalid_missing.any():
            raise InputDataError(
                "optimization ticker(s) missing full-universe prediction: "
                f"{list(universe[invalid_missing][:10])}"
            )
    calibrated["signal_score"] = calibrated["signal_score"].fillna(0.0)
    calibrated["expected_return"] = calibrated["expected_return"].fillna(0.0)

    covariance, covariance_diagnostics = load_covariance(
        covariance_file,
        universe,
        annualized=config["covariance"]["annualized"],
        periods_per_year=config["covariance"]["periods_per_year"],
        eigenvalue_floor=config["covariance"]["eigenvalue_floor"],
        symmetry_tolerance=config["covariance"]["symmetry_tolerance"],
    )

    signal_baseline = build_equal_weight_baseline(
        calibrated_signal.loc[candidates, "signal_score"],
        config["baseline"]["top_n"],
    )
    baseline = signal_baseline.reindex(universe, fill_value=0.0)
    optimized = optimize_portfolio(
        calibrated["expected_return"],
        covariance,
        benchmark,
        current,
        sectors,
        exposures,
        tradable,
        config["optimizer"],
        config["constraints"],
        signal_score=calibrated["signal_score"],
    )

    baseline_constraints = constraint_report(
        baseline,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        config["constraints"],
    )
    risk_baseline = portfolio_metrics(
        baseline, calibrated["expected_return"], covariance, benchmark
    )
    risk_optimized = portfolio_metrics(
        optimized.weights, calibrated["expected_return"], covariance, benchmark
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

    candidate_mask = universe.isin(candidates)
    signal_diagnostics.update(
        {
            "calibration_asset_count": int(len(signal)),
            "candidate_asset_count": int(len(candidates)),
            "optimization_asset_count": int(len(universe)),
            "optimization_prediction_coverage": float(
                1.0 - missing_prediction.mean()
            ),
            "missing_prediction_policy": policy,
            "allowed_frozen_missing_prediction_count": int(
                allowed_frozen_missing.sum()
            ),
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
            "is_candidate": candidate_mask,
            "has_signal": universe.isin(signal.index),
            "raw_prediction": calibrated["raw_prediction"].to_numpy(dtype=float),
            "signal_score": calibrated["signal_score"].to_numpy(dtype=float),
            "expected_return": calibrated["expected_return"].to_numpy(dtype=float),
            "tradable": tradable.to_numpy(dtype=bool),
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
        "schema_version": 3,
        "implementation_version": __version__,
        "status": "success",
        "requested_date": date,
        "asset_count": int(len(universe)),
        "signal_asset_count": int(len(signal)),
        "candidate_asset_count": int(len(candidates)),
        "benchmark_or_current_only_asset_count": int(
            len(universe.difference(candidates))
        ),
        "benchmark_only_asset_count": int(
            len(universe.difference(signal.index))
        ),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "config": config,
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
