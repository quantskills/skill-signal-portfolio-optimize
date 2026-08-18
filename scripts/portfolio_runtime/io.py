from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .errors import InputDataError


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq"}


def normalize_date(value: object) -> str:
    if value is None or pd.isna(value):
        raise InputDataError("date cannot be missing")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    compact = text.replace("-", "").replace("/", "")
    if len(compact) >= 8 and compact[:8].isdigit():
        candidate = compact[:8]
    else:
        try:
            candidate = pd.Timestamp(value).strftime("%Y%m%d")
        except (TypeError, ValueError) as exc:
            raise InputDataError(f"cannot parse date value {value!r}") from exc
    try:
        pd.Timestamp(candidate)
    except ValueError as exc:
        raise InputDataError(f"invalid date value {value!r}") from exc
    return candidate


def normalize_ticker(value: object) -> str:
    if value is None or pd.isna(value):
        raise InputDataError("ticker cannot be missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    ticker = str(value).strip()
    if not ticker:
        raise InputDataError("ticker cannot be empty")
    return ticker


def read_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise InputDataError(f"input file does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise InputDataError(f"unsupported input format {suffix}: {file_path}")
    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)
        return pd.read_parquet(file_path)
    except Exception as exc:
        raise InputDataError(f"cannot read {file_path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise InputDataError(f"{label} missing column(s): {', '.join(sorted(missing))}")


def select_date_rows(
    frame: pd.DataFrame,
    requested_date: str,
    label: str,
    *,
    date_optional: bool = False,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        if date_optional:
            return frame.copy()
        raise InputDataError(f"{label} missing column: date")
    normalized = frame["date"].map(normalize_date)
    selected = frame.loc[normalized == requested_date].copy()
    if selected.empty:
        raise InputDataError(f"{label} has no rows for {requested_date}")
    selected["date"] = requested_date
    return selected


def _normalize_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    _require_columns(frame, ["ticker"], label)
    result = frame.copy()
    result["ticker"] = result["ticker"].map(normalize_ticker)
    duplicates = result["ticker"].duplicated(keep=False)
    if duplicates.any():
        values = sorted(result.loc[duplicates, "ticker"].unique())
        raise InputDataError(f"{label} has duplicate ticker(s): {values[:10]}")
    return result.set_index("ticker", drop=True)


def load_signal(path: str | Path, requested_date: str) -> pd.Series:
    frame = select_date_rows(read_table(path), requested_date, "signal")
    _require_columns(frame, ["ticker", "prediction"], "signal")
    unexpected = set(frame.columns) - {"date", "ticker", "prediction"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise InputDataError(
            f"signal must contain one prediction column; unexpected: {names}"
        )
    indexed = _normalize_index(frame[["ticker", "prediction"]], "signal")
    values = pd.to_numeric(indexed["prediction"], errors="coerce")
    if values.empty:
        raise InputDataError("signal is empty")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise InputDataError("signal prediction contains missing or non-finite values")
    return values.astype(float).sort_index()


def load_candidate_universe(path: str | Path, requested_date: str) -> pd.Index:
    frame = select_date_rows(read_table(path), requested_date, "candidate universe")
    _require_columns(frame, ["ticker"], "candidate universe")
    indexed = _normalize_index(frame[["ticker"]], "candidate universe")
    if len(indexed.index) == 0:
        raise InputDataError("candidate universe is empty")
    return pd.Index(sorted(indexed.index), name="ticker")


def load_weight_series(
    path: str | Path,
    requested_date: str,
    value_column: str,
    label: str,
    universe: pd.Index | None = None,
    *,
    require_complete: bool = False,
) -> pd.Series:
    frame = select_date_rows(read_table(path), requested_date, label)
    _require_columns(frame, ["ticker", value_column], label)
    indexed = _normalize_index(frame[["ticker", value_column]], label)
    values = pd.to_numeric(indexed[value_column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise InputDataError(f"{label} contains missing or non-finite weights")
    if universe is None:
        return values.astype(float).sort_index()
    missing = universe.difference(indexed.index)
    if require_complete and len(missing):
        raise InputDataError(f"{label} missing optimization ticker(s): {list(missing[:10])}")
    return values.reindex(universe, fill_value=0.0).astype(float)


def load_labels(
    path: str | Path,
    requested_date: str,
    value_column: str,
    label: str,
    universe: pd.Index,
) -> pd.Series:
    frame = select_date_rows(read_table(path), requested_date, label, date_optional=True)
    _require_columns(frame, ["ticker", value_column], label)
    indexed = _normalize_index(frame[["ticker", value_column]], label)
    missing = universe.difference(indexed.index)
    if len(missing):
        raise InputDataError(f"{label} missing optimization ticker(s): {list(missing[:10])}")
    values = indexed[value_column].reindex(universe)
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise InputDataError(f"{label} contains missing or empty labels")
    return values.astype(str)


def load_exposures(
    path: str | Path,
    requested_date: str,
    universe: pd.Index,
    required_factors: set[str] | None = None,
) -> pd.DataFrame:
    frame = select_date_rows(read_table(path), requested_date, "exposures", date_optional=True)
    if "ticker" not in frame.columns and frame.index.name == "ticker":
        frame = frame.reset_index()
    _require_columns(frame, ["ticker"], "exposures")
    available_columns = [
        column for column in frame.columns if column not in {"date", "ticker"}
    ]
    if required_factors is None:
        value_columns = [
            column for column in available_columns
            if not str(column).upper().startswith("INDUSTRY:")
        ]
    else:
        missing_factors = required_factors.difference(available_columns)
        if missing_factors:
            raise InputDataError(
                f"exposures missing configured style factor(s): {sorted(missing_factors)}"
            )
        value_columns = [
            column for column in available_columns if column in required_factors
        ]
    if not value_columns:
        raise InputDataError("exposures has no enabled style factor columns")
    indexed = _normalize_index(frame[["ticker", *value_columns]], "exposures")
    missing = universe.difference(indexed.index)
    if len(missing):
        raise InputDataError(
            f"exposures missing optimization ticker(s): {list(missing[:10])}"
        )
    numeric = indexed[value_columns].apply(pd.to_numeric, errors="coerce").reindex(universe)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise InputDataError("exposures contains missing or non-finite values")
    return numeric.astype(float)


def load_tradability(
    path: str | Path, requested_date: str, universe: pd.Index
) -> pd.Series:
    frame = select_date_rows(read_table(path), requested_date, "tradability", date_optional=True)
    _require_columns(frame, ["ticker", "tradable"], "tradability")
    indexed = _normalize_index(frame[["ticker", "tradable"]], "tradability")
    missing = universe.difference(indexed.index)
    if len(missing):
        raise InputDataError(
            f"tradability missing optimization ticker(s): {list(missing[:10])}"
        )
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}
    parsed: list[bool] = []
    for ticker, value in indexed["tradable"].reindex(universe).items():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in true_values:
            parsed.append(True)
        elif text in false_values:
            parsed.append(False)
        else:
            raise InputDataError(f"invalid tradable value for {ticker}: {value!r}")
    return pd.Series(parsed, index=universe, name="tradable", dtype=bool)


def load_covariance(
    path: str | Path,
    universe: pd.Index,
    *,
    annualized: bool,
    symmetry_tolerance: float,
    periods_per_year: int,
    eigenvalue_floor: float,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise InputDataError(f"input file does not exist: {file_path}")
    try:
        if file_path.suffix.lower() == ".csv":
            raw = pd.read_csv(file_path, index_col=0)
        elif file_path.suffix.lower() in {".parquet", ".pq"}:
            raw = pd.read_parquet(file_path)
        else:
            raise InputDataError(f"unsupported covariance format: {file_path.suffix}")
    except InputDataError:
        raise
    except Exception as exc:
        raise InputDataError(f"cannot read covariance {file_path}: {exc}") from exc

    if raw.shape[0] != raw.shape[1]:
        raise InputDataError(f"covariance must be square, got {raw.shape}")
    raw.index = pd.Index([normalize_ticker(value) for value in raw.index], name="ticker")
    raw.columns = pd.Index([normalize_ticker(value) for value in raw.columns])
    if raw.index.has_duplicates or raw.columns.has_duplicates:
        raise InputDataError("covariance has duplicate row or column tickers")
    if set(raw.index) != set(raw.columns):
        raise InputDataError("covariance row and column ticker sets differ")
    missing = universe.difference(raw.index)
    if len(missing):
        raise InputDataError(
            f"covariance missing optimization ticker(s): {list(missing[:10])}"
        )

    numeric = raw.apply(pd.to_numeric, errors="coerce").reindex(index=universe, columns=universe)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise InputDataError("covariance contains missing or non-finite values")
    if not annualized:
        values = values * periods_per_year

    max_asymmetry = float(np.max(np.abs(values - values.T)))
    if max_asymmetry > symmetry_tolerance:
        raise InputDataError(
            f"covariance asymmetry {max_asymmetry:.6g} exceeds tolerance "
            f"{symmetry_tolerance:.6g}"
        )
    symmetric = 0.5 * (values + values.T)
    before, vectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(before, eigenvalue_floor)
    repaired = (vectors * clipped) @ vectors.T
    repaired = 0.5 * (repaired + repaired.T)
    after = np.linalg.eigvalsh(repaired)

    diagnostics: dict[str, float | bool] = {
        "input_annualized": annualized,
        "annualization_multiplier": 1.0 if annualized else float(periods_per_year),
        "max_asymmetry": max_asymmetry,
        "min_eigenvalue_before": float(before.min()),
        "min_eigenvalue_after": float(after.min()),
        "eigenvalue_floor": float(eigenvalue_floor),
        "repair_frobenius_norm": float(np.linalg.norm(repaired - values, ord="fro")),
        "was_repaired": bool(max_asymmetry > 0.0 or np.any(before < eigenvalue_floor)),
    }
    return pd.DataFrame(repaired, index=universe, columns=universe), diagnostics


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
