from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import InputDataError
from .io import normalize_ticker
from .risk_model import (
    OUTPUT_FILES,
    RiskModelContext,
    build_risk_model_from_context,
    load_risk_model_context,
)


@dataclass(frozen=True)
class ResolvedRiskModel:
    covariance_file: Path
    exposure_file: Path
    factor_covariance_file: Path
    specific_variance_file: Path
    source: str
    cache_directory: Path
    fingerprint: str
    asset_count: int


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_universe(universe: pd.Index) -> pd.Index:
    values = sorted({normalize_ticker(value) for value in universe})
    if not values:
        raise InputDataError("dynamic risk universe cannot be empty")
    return pd.Index(values, name="ticker")


def _matrix_tickers(path: Path) -> tuple[pd.Index, pd.Index]:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise InputDataError(f"cannot read covariance coverage from {path}: {exc}") from exc
    if frame.shape[0] != frame.shape[1]:
        raise InputDataError(f"covariance must be square, got {frame.shape}: {path}")
    rows = pd.Index([normalize_ticker(value) for value in frame.index])
    columns = pd.Index([normalize_ticker(value) for value in frame.columns])
    return rows, columns


def _exposure_tickers(path: Path) -> pd.Index:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise InputDataError(f"cannot read exposure coverage from {path}: {exc}") from exc
    if "ticker" in frame.columns:
        values = frame["ticker"]
    else:
        values = frame.index
    return pd.Index([normalize_ticker(value) for value in values])


def risk_files_cover_universe(
    covariance_file: Path, exposure_file: Path, universe: pd.Index
) -> bool:
    rows, columns = _matrix_tickers(covariance_file)
    exposures = _exposure_tickers(exposure_file)
    required = set(_normalized_universe(universe))
    return required.issubset(rows) and required.issubset(columns) and required.issubset(exposures)


class DynamicRiskModelCache:
    """Resolve complete static risk files or build an exact-universe dynamic cache."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        returns_file: str | Path,
        market_cap_file: str | Path,
        industry_file: str | Path | None,
        cache_root: str | Path,
    ) -> None:
        self.context = load_risk_model_context(
            config_path=config_path,
            returns_file=returns_file,
            market_cap_file=market_cap_file,
            industry_file=industry_file,
        )
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.static_reused_count = 0
        self.dynamic_reused_count = 0
        self.dynamic_built_count = 0

    def fingerprint(self, date: str, universe: pd.Index) -> tuple[str, str]:
        normalized = _normalized_universe(universe)
        universe_hash = _canonical_hash(normalized.tolist())
        fingerprint = _canonical_hash(
            {
                "date": date,
                "universe": normalized.tolist(),
                "risk_inputs": {
                    name: detail["sha256"]
                    for name, detail in sorted(self.context.inputs.items())
                },
                "risk_config": self.context.config,
            }
        )
        return fingerprint, universe_hash

    def validate_positive_current_holdings(
        self, date: str, current: pd.Series, tolerance: float
    ) -> None:
        held = pd.Index(
            [normalize_ticker(value) for value in current[current > tolerance].index],
            name="ticker",
        )
        if held.empty:
            return
        missing_returns = held.difference(self.context.returns.columns)
        missing_cap = held.difference(self.context.market_cap.columns)
        if len(missing_returns) or len(missing_cap):
            missing = missing_returns.union(missing_cap)
            raise InputDataError(
                "positive current holding(s) missing risk panels: "
                f"{list(missing[:10])}"
            )
        cap_dates = self.context.market_cap.index[
            self.context.market_cap.index <= date
        ]
        if cap_dates.empty:
            raise InputDataError(f"market cap has no as-of row through {date}")
        cap = self.context.market_cap.loc[cap_dates[-1]].reindex(held)
        invalid = cap.isna() | ~np.isfinite(cap) | cap.le(0)
        if invalid.any():
            raise InputDataError(
                "positive current holding(s) have invalid as-of market cap: "
                f"{held[invalid.to_numpy()].tolist()[:10]}"
            )

    def _validate_complete_dynamic_cache(
        self,
        directory: Path,
        *,
        date: str,
        universe: pd.Index,
        fingerprint: str,
    ) -> bool:
        if not directory.exists():
            return False
        missing = [name for name in OUTPUT_FILES if not (directory / name).is_file()]
        if missing:
            raise InputDataError(
                f"incomplete dynamic risk cache at {directory}; missing {missing}"
            )
        manifest_path = directory / "risk_model_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputDataError(
                f"invalid dynamic risk cache manifest at {manifest_path}: {exc}"
            ) from exc
        if (
            manifest.get("status") != "success"
            or manifest.get("requested_date") != date
            or manifest.get("dynamic_cache_fingerprint") != fingerprint
        ):
            raise InputDataError(f"stale dynamic risk cache at {directory}")
        if not risk_files_cover_universe(
            directory / "asset_cov.parquet",
            directory / "exposures.parquet",
            universe,
        ):
            raise InputDataError(f"dynamic risk cache lacks requested universe: {directory}")
        return True

    def resolve(
        self,
        *,
        date: str,
        universe: pd.Index,
        static_covariance_file: Path | None,
        static_exposure_file: Path | None,
    ) -> ResolvedRiskModel:
        normalized = _normalized_universe(universe)
        fingerprint, universe_hash = self.fingerprint(date, normalized)
        if static_covariance_file is not None and static_exposure_file is not None:
            if risk_files_cover_universe(
                static_covariance_file, static_exposure_file, normalized
            ):
                self.static_reused_count += 1
                return ResolvedRiskModel(
                    covariance_file=static_covariance_file,
                    exposure_file=static_exposure_file,
                    factor_covariance_file=static_covariance_file.parent / "factor_cov.parquet",
                    specific_variance_file=static_covariance_file.parent / "specific_var.parquet",
                    source="static_reused",
                    cache_directory=static_covariance_file.parent,
                    fingerprint=fingerprint,
                    asset_count=len(normalized),
                )

        directory = (
            self.cache_root / f"date={date}" / f"universe={fingerprint[:16]}"
        )
        if self._validate_complete_dynamic_cache(
            directory,
            date=date,
            universe=normalized,
            fingerprint=fingerprint,
        ):
            self.dynamic_reused_count += 1
            source = "dynamic_reused"
        else:
            build_risk_model_from_context(
                context=self.context,
                target_universe=normalized,
                requested_date=date,
                output_dir=directory,
                manifest_metadata={
                    "cache_mode": "dynamic_exact_universe",
                    "dynamic_cache_fingerprint": fingerprint,
                    "universe_sha256": universe_hash,
                },
            )
            self.dynamic_built_count += 1
            source = "dynamic_built"
        return ResolvedRiskModel(
            covariance_file=directory / "asset_cov.parquet",
            exposure_file=directory / "exposures.parquet",
            factor_covariance_file=directory / "factor_cov.parquet",
            specific_variance_file=directory / "specific_var.parquet",
            source=source,
            cache_directory=directory,
            fingerprint=fingerprint,
            asset_count=len(normalized),
        )

    def statistics(self) -> dict[str, int]:
        return {
            "static_reused_count": self.static_reused_count,
            "dynamic_reused_count": self.dynamic_reused_count,
            "dynamic_built_count": self.dynamic_built_count,
        }
