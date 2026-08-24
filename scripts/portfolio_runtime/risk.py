from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .errors import InputDataError
from .io import normalize_ticker, read_table, select_date_rows, sha256_file


@dataclass(frozen=True)
class PortfolioRisk:
    """Covariance operator retaining either dense or structural factor form."""

    form: str
    tickers: pd.Index
    asset_covariance: pd.DataFrame | None = None
    exposures: pd.DataFrame | None = None
    factor_covariance: pd.DataFrame | None = None
    specific_variance: pd.Series | None = None
    diagnostics: dict[str, Any] | None = None
    input_hashes: dict[str, str] | None = None

    def variance(self, weights: np.ndarray | pd.Series) -> float:
        values = np.asarray(weights, dtype=float)
        if values.shape != (len(self.tickers),):
            raise InputDataError("risk weights do not match the risk universe")
        if self.form == "asset_covariance":
            assert self.asset_covariance is not None
            sigma = self.asset_covariance.to_numpy(dtype=float)
            return float(values @ sigma @ values)
        assert self.exposures is not None
        assert self.factor_covariance is not None
        assert self.specific_variance is not None
        x = self.exposures.to_numpy(dtype=float)
        f = self.factor_covariance.to_numpy(dtype=float)
        d = self.specific_variance.to_numpy(dtype=float)
        factor_active = x.T @ values
        return float(factor_active @ f @ factor_active + np.dot(d, values**2))

    def gradient(self, weights: np.ndarray | pd.Series) -> np.ndarray:
        values = np.asarray(weights, dtype=float)
        if self.form == "asset_covariance":
            assert self.asset_covariance is not None
            return 2.0 * self.asset_covariance.to_numpy(dtype=float) @ values
        assert self.exposures is not None
        assert self.factor_covariance is not None
        assert self.specific_variance is not None
        x = self.exposures.to_numpy(dtype=float)
        f = self.factor_covariance.to_numpy(dtype=float)
        d = self.specific_variance.to_numpy(dtype=float)
        return 2.0 * (x @ (f @ (x.T @ values)) + d * values)

    def dense(self) -> pd.DataFrame:
        if self.form == "asset_covariance":
            assert self.asset_covariance is not None
            return self.asset_covariance
        assert self.exposures is not None
        assert self.factor_covariance is not None
        assert self.specific_variance is not None
        x = self.exposures.to_numpy(dtype=float)
        f = self.factor_covariance.to_numpy(dtype=float)
        values = x @ f @ x.T + np.diag(self.specific_variance.to_numpy(dtype=float))
        return pd.DataFrame(values, index=self.tickers, columns=self.tickers)


def as_portfolio_risk(value: pd.DataFrame | PortfolioRisk) -> PortfolioRisk:
    if isinstance(value, PortfolioRisk):
        return value
    return PortfolioRisk(
        form="asset_covariance",
        tickers=value.index,
        asset_covariance=value,
        diagnostics={"risk_form": "asset_covariance"},
        input_hashes={},
    )


def _load_factor_covariance(path: str) -> pd.DataFrame:
    frame = read_table(path)
    if frame.shape[0] != frame.shape[1]:
        raise InputDataError(f"factor covariance must be square, got {frame.shape}")
    frame.index = pd.Index([str(value) for value in frame.index], name="factor")
    frame.columns = pd.Index([str(value) for value in frame.columns])
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise InputDataError("factor covariance has duplicate factor labels")
    if set(frame.index) != set(frame.columns):
        raise InputDataError("factor covariance row and column factors differ")
    numeric = frame.apply(pd.to_numeric, errors="coerce").reindex(
        index=frame.index, columns=frame.index
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise InputDataError("factor covariance contains missing or non-finite values")
    return numeric.astype(float)


def load_factor_risk(
    *,
    exposure_file: str,
    factor_covariance_file: str,
    specific_variance_file: str,
    universe: pd.Index,
    requested_date: str,
    annualized: bool,
    periods_per_year: int,
    eigenvalue_floor: float,
    symmetry_tolerance: float,
) -> PortfolioRisk:
    exposure_frame = select_date_rows(
        read_table(exposure_file), requested_date, "factor risk exposures", date_optional=True
    )
    if "ticker" not in exposure_frame.columns and exposure_frame.index.name == "ticker":
        exposure_frame = exposure_frame.reset_index()
    if "ticker" not in exposure_frame.columns:
        raise InputDataError("factor risk exposures missing column: ticker")
    exposure_frame = exposure_frame.copy()
    exposure_frame["ticker"] = exposure_frame["ticker"].map(normalize_ticker)
    if exposure_frame["ticker"].duplicated().any():
        raise InputDataError("factor risk exposures have duplicate tickers")
    exposure_frame = exposure_frame.set_index("ticker")

    factor_covariance = _load_factor_covariance(factor_covariance_file)
    factors = factor_covariance.index.tolist()
    missing_factors = set(factors) - set(exposure_frame.columns)
    if missing_factors:
        raise InputDataError(
            f"factor risk exposures missing factor(s): {sorted(missing_factors)}"
        )
    missing_assets = universe.difference(exposure_frame.index)
    if len(missing_assets):
        raise InputDataError(
            "factor risk exposures missing optimization ticker(s): "
            f"{list(missing_assets[:10])}"
        )
    exposures = exposure_frame[factors].apply(pd.to_numeric, errors="coerce").reindex(universe)
    if not np.isfinite(exposures.to_numpy(dtype=float)).all():
        raise InputDataError("factor risk exposures contain missing or non-finite values")

    specific_frame = select_date_rows(
        read_table(specific_variance_file),
        requested_date,
        "specific variance",
        date_optional=True,
    )
    if "ticker" not in specific_frame.columns and specific_frame.index.name == "ticker":
        specific_frame = specific_frame.reset_index()
    value_columns = [
        name for name in specific_frame.columns if name not in {"date", "ticker"}
    ]
    if "ticker" not in specific_frame.columns or len(value_columns) != 1:
        raise InputDataError(
            "specific variance must contain ticker and exactly one value column"
        )
    specific_frame = specific_frame.copy()
    specific_frame["ticker"] = specific_frame["ticker"].map(normalize_ticker)
    if specific_frame["ticker"].duplicated().any():
        raise InputDataError("specific variance has duplicate tickers")
    specific = pd.to_numeric(
        specific_frame.set_index("ticker")[value_columns[0]], errors="coerce"
    ).reindex(universe)
    if not np.isfinite(specific.to_numpy(dtype=float)).all():
        raise InputDataError("specific variance contains missing or non-finite values")
    if (specific < 0).any():
        raise InputDataError("specific variance contains negative values")

    factor_values = factor_covariance.to_numpy(dtype=float)
    max_asymmetry = float(np.max(np.abs(factor_values - factor_values.T)))
    if max_asymmetry > symmetry_tolerance:
        raise InputDataError(
            f"factor covariance asymmetry {max_asymmetry:.6g} exceeds tolerance "
            f"{symmetry_tolerance:.6g}"
        )
    symmetric = 0.5 * (factor_values + factor_values.T)
    before, vectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(before, eigenvalue_floor)
    repaired = (vectors * clipped) @ vectors.T
    repaired = 0.5 * (repaired + repaired.T)
    multiplier = 1.0 if annualized else float(periods_per_year)
    repaired *= multiplier
    specific = specific.astype(float) * multiplier
    repaired_factor = pd.DataFrame(
        repaired, index=factor_covariance.index, columns=factor_covariance.index
    )
    diagnostics = {
        "risk_form": "factor_model",
        "input_annualized": bool(annualized),
        "annualization_multiplier": multiplier,
        "factor_count": len(factors),
        "max_asymmetry": max_asymmetry,
        "min_eigenvalue_before": float(before.min()),
        "min_eigenvalue_after": float(np.linalg.eigvalsh(repaired).min()),
        "eigenvalue_floor": float(eigenvalue_floor),
        "was_repaired": bool(max_asymmetry > 0.0 or np.any(before < eigenvalue_floor)),
    }
    return PortfolioRisk(
        form="factor_model",
        tickers=universe,
        exposures=exposures.astype(float),
        factor_covariance=repaired_factor,
        specific_variance=specific.rename("specific_var"),
        diagnostics=diagnostics,
        input_hashes={
            "factor_exposures": sha256_file(exposure_file),
            "factor_covariance": sha256_file(factor_covariance_file),
            "specific_variance": sha256_file(specific_variance_file),
        },
    )
