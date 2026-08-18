from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .errors import ConfigError, InputDataError, RiskModelError
from .io import normalize_date, normalize_ticker, read_table, sha256_file


DEFAULT_RISK_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "industry_mode": "required",
    "style_factors": ["SIZE"],
    "style_lookback_days": 252,
    "style_minimum_observations": 126,
    "momentum_skip_days": 21,
    "lookback_days": 300,
    "minimum_regression_periods": 120,
    "minimum_cross_section_assets": 200,
    "minimum_specific_observations": 60,
    "return_winsorize_mad": 8.0,
    "size_winsorize_mad": 5.0,
    "regression_weight_power": 0.5,
    "factor_covariance_halflife": 90.0,
    "specific_variance_halflife": 60.0,
    "factor_covariance_diagonal_shrinkage": 0.10,
    "specific_variance_median_shrinkage": 0.10,
    "annualization_periods": 252,
    "factor_eigenvalue_floor": 1.0e-8,
    "specific_variance_floor": 1.0e-6,
    "max_specific_imputation_fraction": 0.20,
}


OUTPUT_FILES = (
    "asset_cov.parquet",
    "factor_cov.parquet",
    "specific_var.parquet",
    "exposures.parquet",
    "factor_returns.parquet",
    "risk_model_manifest.json",
)


@dataclass(frozen=True)
class RiskModelResult:
    asset_cov: pd.DataFrame
    factor_cov: pd.DataFrame
    specific_var: pd.Series
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RiskModelContext:
    config: dict[str, Any]
    returns: pd.DataFrame
    market_cap: pd.DataFrame
    industry_history: pd.DataFrame | None
    inputs: dict[str, dict[str, str]]


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    return result


def load_risk_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        supplied = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read risk-model config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid risk-model YAML in {config_path}: {exc}") from exc
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ConfigError("risk-model configuration root must be a mapping")
    unknown = set(supplied) - set(DEFAULT_RISK_CONFIG)
    if unknown:
        raise ConfigError(
            "unknown risk-model configuration key(s): " + ", ".join(sorted(unknown))
        )
    config = {**DEFAULT_RISK_CONFIG, **supplied}
    if config["schema_version"] != 1:
        raise ConfigError("risk-model schema_version must equal 1")
    integer_fields = (
        "lookback_days",
        "minimum_regression_periods",
        "minimum_cross_section_assets",
        "minimum_specific_observations",
        "annualization_periods",
        "style_lookback_days",
        "style_minimum_observations",
        "momentum_skip_days",
    )
    for name in integer_fields:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{name} must be a positive integer")
    if config["minimum_regression_periods"] > config["lookback_days"]:
        raise ConfigError("minimum_regression_periods cannot exceed lookback_days")
    if config["industry_mode"] not in {"required", "optional", "disabled"}:
        raise ConfigError("industry_mode must be required, optional, or disabled")
    if not isinstance(config["style_factors"], list) or not config["style_factors"]:
        raise ConfigError("style_factors must be a non-empty list")
    styles = [str(value).upper().strip() for value in config["style_factors"]]
    supported_styles = {"SIZE", "BETA", "MOMENTUM", "RESVOL", "NLSIZE"}
    unknown_styles = set(styles) - supported_styles
    if unknown_styles:
        raise ConfigError(f"unsupported style_factors: {sorted(unknown_styles)}")
    if len(set(styles)) != len(styles):
        raise ConfigError("style_factors must be unique")
    config["style_factors"] = styles
    if config["style_minimum_observations"] > config["style_lookback_days"]:
        raise ConfigError(
            "style_minimum_observations cannot exceed style_lookback_days"
        )
    if config["momentum_skip_days"] >= config["style_lookback_days"]:
        raise ConfigError("momentum_skip_days must be less than style_lookback_days")
    for name in (
        "return_winsorize_mad",
        "size_winsorize_mad",
        "regression_weight_power",
        "factor_covariance_halflife",
        "specific_variance_halflife",
        "factor_eigenvalue_floor",
        "specific_variance_floor",
    ):
        config[name] = _finite_number(config[name], name, minimum=0.0)
    for name in (
        "factor_covariance_diagonal_shrinkage",
        "specific_variance_median_shrinkage",
        "max_specific_imputation_fraction",
    ):
        config[name] = _finite_number(config[name], name, minimum=0.0)
        if config[name] > 1.0:
            raise ConfigError(f"{name} must not exceed 1")
    if config["factor_covariance_halflife"] <= 0:
        raise ConfigError("factor_covariance_halflife must be positive")
    if config["specific_variance_halflife"] <= 0:
        raise ConfigError("specific_variance_halflife must be positive")
    return config


def load_wide_panel(path: str | Path, label: str) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise InputDataError(f"input file does not exist: {file_path}")
    try:
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path, index_col=0)
        elif file_path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(file_path)
        else:
            raise InputDataError(f"unsupported {label} format: {file_path.suffix}")
    except InputDataError:
        raise
    except Exception as exc:
        raise InputDataError(f"cannot read {label} {file_path}: {exc}") from exc
    if frame.empty:
        raise InputDataError(f"{label} is empty")
    dates = pd.Index([normalize_date(value) for value in frame.index], name="date")
    tickers = pd.Index([normalize_ticker(value) for value in frame.columns], name="ticker")
    if dates.has_duplicates:
        raise InputDataError(f"{label} has duplicate dates")
    if tickers.has_duplicates:
        raise InputDataError(f"{label} has duplicate tickers")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric.index = dates
    numeric.columns = tickers
    return numeric.sort_index()


def _load_universe(path: str | Path, requested_date: str) -> pd.Index:
    frame = read_table(path)
    if "date" not in frame.columns or "ticker" not in frame.columns:
        raise InputDataError("universe must contain date and ticker columns")
    dates = frame["date"].map(normalize_date)
    selected = frame.loc[dates.eq(requested_date), "ticker"].map(normalize_ticker)
    if selected.empty:
        raise InputDataError(f"universe has no rows for {requested_date}")
    if selected.duplicated().any():
        raise InputDataError("universe contains duplicate tickers")
    return pd.Index(selected.tolist(), name="ticker")


def load_industry_history(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    required = {"stock_symbol", "l1_code", "in_date", "out_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputDataError(
            "industry history missing column(s): " + ", ".join(missing)
        )
    result = frame.loc[:, list(required)].copy()
    result["ticker"] = result["stock_symbol"].map(normalize_ticker)
    result["industry"] = result["l1_code"].astype("string").str.strip()
    if result["industry"].isna().any() or result["industry"].eq("").any():
        raise InputDataError("industry history contains missing industry codes")
    result["in_date"] = result["in_date"].map(normalize_date)
    out_text = result["out_date"].astype("string").str.strip()
    missing_out = result["out_date"].isna() | out_text.isin({"", "<NA>", "nan", "None"})
    result["out_date_normalized"] = "99991231"
    if (~missing_out).any():
        result.loc[~missing_out, "out_date_normalized"] = result.loc[
            ~missing_out, "out_date"
        ].map(normalize_date)
    if (result["out_date_normalized"] < result["in_date"]).any():
        raise InputDataError("industry history contains out_date before in_date")
    result = result[
        ["ticker", "industry", "in_date", "out_date_normalized"]
    ].sort_values(["ticker", "in_date"], kind="stable")
    for ticker, group in result.groupby("ticker", sort=False):
        previous_end = group["out_date_normalized"].shift()
        if (group["in_date"] <= previous_end).fillna(False).any():
            raise InputDataError(f"industry history has overlapping intervals for {ticker}")
    return result.reset_index(drop=True)


def _industry_asof(history: pd.DataFrame, date: str) -> pd.Series:
    active = history.loc[
        history["in_date"].le(date) & history["out_date_normalized"].ge(date),
        ["ticker", "industry"],
    ]
    if active["ticker"].duplicated().any():
        raise RiskModelError(f"multiple active industries exist at {date}")
    return active.set_index("ticker")["industry"]


def _mad_clip(values: pd.Series, threshold: float) -> pd.Series:
    result = values.astype(float)
    if threshold <= 0:
        return result
    median = float(result.median())
    mad = float((result - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        return result
    return result.clip(median - threshold * scale, median + threshold * scale)


def _weighted_standardize(values: pd.Series, weights: pd.Series) -> pd.Series:
    normalized = weights / float(weights.sum())
    mean = float((values * normalized).sum())
    variance = float(((values - mean) ** 2 * normalized).sum())
    if not np.isfinite(variance) or variance <= 0:
        raise RiskModelError("SIZE exposure has no usable cross-sectional variation")
    return (values - mean) / np.sqrt(variance)


def _standardize_style(
    values: pd.Series,
    weights: pd.Series,
    *,
    name: str,
    winsorize_mad: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() < 2:
        raise RiskModelError(f"{name} exposure has too few usable observations")
    numeric = numeric.fillna(float(numeric.median()))
    clipped = _mad_clip(numeric, winsorize_mad)
    try:
        return _weighted_standardize(clipped, weights)
    except RiskModelError as exc:
        raise RiskModelError(f"{name} exposure has no usable cross-sectional variation") from exc


def _style_exposures_asof(
    returns: pd.DataFrame,
    market_cap: pd.DataFrame,
    exposure_date: str,
    tickers: pd.Index,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    cap = market_cap.loc[exposure_date, tickers].astype(float)
    regression_weight = np.power(cap, config["regression_weight_power"])
    log_cap = _mad_clip(np.log(cap), config["size_winsorize_mad"])
    size = _weighted_standardize(log_cap, regression_weight)
    exposures = pd.DataFrame(index=tickers)
    requested = config["style_factors"]
    if "SIZE" in requested:
        exposures["SIZE"] = size
    if "NLSIZE" in requested:
        raw = size.pow(3)
        design = np.column_stack([np.ones(len(size)), size.to_numpy(dtype=float)])
        root_weight = np.sqrt(regression_weight / float(regression_weight.mean()))
        coefficients, _, _, _ = np.linalg.lstsq(
            design * root_weight.to_numpy()[:, None],
            raw.to_numpy(dtype=float) * root_weight.to_numpy(),
            rcond=None,
        )
        residual = pd.Series(
            raw.to_numpy(dtype=float) - design @ coefficients,
            index=tickers,
        )
        exposures["NLSIZE"] = _standardize_style(
            residual,
            regression_weight,
            name="NLSIZE",
            winsorize_mad=config["size_winsorize_mad"],
        )

    return_based = {"BETA", "MOMENTUM", "RESVOL"}.intersection(requested)
    if return_based:
        history = returns.loc[returns.index <= exposure_date, tickers].tail(
            config["style_lookback_days"]
        )
        minimum = config["style_minimum_observations"]
        if len(history) < minimum:
            raise RiskModelError(
                f"only {len(history)} return rows exist for style exposures at {exposure_date}"
            )
        market_return = history.mean(axis=1, skipna=True)
        valid_counts = history.notna().sum()
        asset_means = history.mean(axis=0, skipna=True)
        centered_assets = history.subtract(asset_means, axis=1)
        centered_market = market_return - float(market_return.mean())
        covariance = centered_assets.mul(centered_market, axis=0).sum(axis=0)
        market_ss = history.notna().mul(centered_market.pow(2), axis=0).sum(axis=0)
        beta_values = covariance.divide(market_ss).where(
            valid_counts.ge(minimum) & market_ss.gt(0)
        )
        residuals = history.subtract(
            pd.DataFrame(
                np.outer(market_return.to_numpy(dtype=float), beta_values.to_numpy(dtype=float)),
                index=history.index,
                columns=history.columns,
            )
        )
        residual_volatility = residuals.std(axis=0, ddof=1).where(
            valid_counts.ge(minimum)
        )
        if "BETA" in requested:
            exposures["BETA"] = _standardize_style(
                beta_values.reindex(tickers),
                regression_weight,
                name="BETA",
                winsorize_mad=config["return_winsorize_mad"],
            )
        if "RESVOL" in requested:
            exposures["RESVOL"] = _standardize_style(
                residual_volatility.reindex(tickers),
                regression_weight,
                name="RESVOL",
                winsorize_mad=config["return_winsorize_mad"],
            )
        if "MOMENTUM" in requested:
            skip = config["momentum_skip_days"]
            momentum_history = history.iloc[:-skip] if skip else history
            valid_values = momentum_history.where(momentum_history.gt(-1.0))
            counts = valid_values.notna().sum()
            momentum = np.log1p(valid_values).sum(min_count=minimum)
            momentum = momentum.where(counts.ge(minimum))
            exposures["MOMENTUM"] = _standardize_style(
                momentum.reindex(tickers),
                regression_weight,
                name="MOMENTUM",
                winsorize_mad=config["return_winsorize_mad"],
            )
    return exposures.loc[:, requested], regression_weight


def _ewma_covariance(
    frame: pd.DataFrame,
    *,
    halflife: float,
    annualization: int,
    shrinkage: float,
    eigenvalue_floor: float,
) -> pd.DataFrame:
    if frame.isna().any().any():
        raise RiskModelError("factor returns contain missing values")
    count = len(frame)
    ages = np.arange(count - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / halflife)
    weights /= weights.sum()
    values = frame.to_numpy(dtype=float)
    mean = weights @ values
    centered = values - mean
    denominator = 1.0 - float(np.square(weights).sum())
    if denominator <= 0:
        raise RiskModelError("factor covariance has insufficient effective observations")
    covariance = (centered.T @ (centered * weights[:, None])) / denominator
    covariance *= annualization
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
        np.diag(covariance)
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    repaired = (eigenvectors * np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    return pd.DataFrame(repaired, index=frame.columns, columns=frame.columns)


def _estimate_specific_variance(
    residuals: pd.DataFrame,
    *,
    halflife: float,
    annualization: int,
    minimum_observations: int,
    shrinkage: float,
    floor: float,
) -> tuple[pd.Series, pd.Series]:
    ages = np.arange(len(residuals) - 1, -1, -1, dtype=float)
    base_weights = np.power(0.5, ages / halflife)
    estimates: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ticker in residuals.columns:
        values = residuals[ticker].to_numpy(dtype=float)
        valid = np.isfinite(values)
        count = int(valid.sum())
        counts[ticker] = count
        if count < minimum_observations:
            continue
        weights = base_weights[valid]
        weights /= weights.sum()
        estimate = float(weights @ np.square(values[valid])) * annualization
        if np.isfinite(estimate):
            estimates[ticker] = max(estimate, floor)
    specific = pd.Series(estimates, dtype=float, name="specific_var")
    observations = pd.Series(counts, dtype="int64", name="specific_observations")
    if specific.empty:
        raise RiskModelError("no assets have enough observations for specific risk")
    median = float(specific.median())
    specific = (1.0 - shrinkage) * specific + shrinkage * median
    return specific.clip(lower=floor), observations


def estimate_structural_risk_model(
    *,
    returns: pd.DataFrame,
    market_cap: pd.DataFrame,
    industry_history: pd.DataFrame | None,
    target_universe: pd.Index,
    requested_date: str,
    config: dict[str, Any],
) -> RiskModelResult:
    config = {**DEFAULT_RISK_CONFIG, **config}
    industry_mode = config["industry_mode"]
    if industry_mode == "required" and industry_history is None:
        raise RiskModelError("industry history is required by industry_mode=required")
    use_industry = industry_mode != "disabled" and industry_history is not None
    common_dates = returns.index.intersection(market_cap.index)
    common_dates = common_dates[common_dates <= requested_date].sort_values()
    required_rows = config["lookback_days"] + 1
    if len(common_dates) < config["minimum_regression_periods"] + 1:
        raise RiskModelError(
            f"only {len(common_dates)} aligned history dates are available through "
            f"{requested_date}"
        )
    window_dates = common_dates[-min(required_rows, len(common_dates)) :]
    return_dates = window_dates[1:]
    common_tickers = returns.columns.intersection(market_cap.columns)
    if len(common_tickers) < config["minimum_cross_section_assets"]:
        raise RiskModelError("returns and market cap have too few common assets")

    factor_rows: list[pd.Series] = []
    residual_rows: list[pd.Series] = []
    regression_assets: list[int] = []
    industry_codes: set[str] = set()
    prepared: list[
        tuple[str, pd.Series, pd.DataFrame, pd.Series, pd.Series | None]
    ] = []
    for position, return_date in enumerate(return_dates, start=1):
        exposure_date = window_dates[position - 1]
        industries = (
            _industry_asof(industry_history, exposure_date).reindex(common_tickers)
            if use_industry and industry_history is not None
            else None
        )
        daily_return = returns.loc[return_date, common_tickers]
        daily_cap = market_cap.loc[exposure_date, common_tickers]
        valid = (
            daily_return.notna()
            & np.isfinite(daily_return)
            & daily_cap.notna()
            & np.isfinite(daily_cap)
            & daily_cap.gt(0)
        )
        if industries is not None:
            valid &= industries.notna()
        if int(valid.sum()) < config["minimum_cross_section_assets"]:
            continue
        selected = common_tickers[valid.to_numpy()]
        response = _mad_clip(
            daily_return.reindex(selected), config["return_winsorize_mad"]
        )
        try:
            styles, regression_weight = _style_exposures_asof(
                returns,
                market_cap,
                exposure_date,
                selected,
                config,
            )
        except RiskModelError:
            continue
        industry = (
            industries.reindex(selected).astype(str) if industries is not None else None
        )
        prepared.append(
            (return_date, response, styles, regression_weight, industry)
        )
        if industry is not None:
            industry_codes.update(industry.unique().tolist())

    factors = [
        *( [] if use_industry else ["MARKET"] ),
        *config["style_factors"],
        *[f"INDUSTRY:{code}" for code in sorted(industry_codes)],
    ]
    for return_date, response, styles, regression_weight, industry in prepared:
        exposure = pd.DataFrame(0.0, index=response.index, columns=factors)
        if not use_industry:
            exposure["MARKET"] = 1.0
        exposure.loc[:, config["style_factors"]] = styles
        if industry is not None:
            for code in industry.unique():
                exposure.loc[industry.eq(code), f"INDUSTRY:{code}"] = 1.0
        root_weight = np.sqrt(regression_weight / float(regression_weight.mean()))
        weighted_x = exposure.to_numpy(dtype=float) * root_weight.to_numpy()[:, None]
        weighted_y = response.to_numpy(dtype=float) * root_weight.to_numpy()
        coefficients, _, rank, _ = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)
        if rank < len(factors):
            continue
        fitted = exposure.to_numpy(dtype=float) @ coefficients
        factor_rows.append(
            pd.Series(coefficients, index=factors, name=return_date, dtype=float)
        )
        residual_rows.append(
            pd.Series(
                response.to_numpy(dtype=float) - fitted,
                index=response.index,
                name=return_date,
                dtype=float,
            )
        )
        regression_assets.append(len(response))

    if len(factor_rows) < config["minimum_regression_periods"]:
        raise RiskModelError(
            f"only {len(factor_rows)} valid factor-return periods were estimated; "
            f"need {config['minimum_regression_periods']}"
        )
    factor_returns = pd.DataFrame(factor_rows).sort_index()
    residuals = pd.DataFrame(residual_rows).sort_index()
    factor_cov = _ewma_covariance(
        factor_returns,
        halflife=config["factor_covariance_halflife"],
        annualization=config["annualization_periods"],
        shrinkage=config["factor_covariance_diagonal_shrinkage"],
        eigenvalue_floor=config["factor_eigenvalue_floor"],
    )
    all_specific, observations = _estimate_specific_variance(
        residuals,
        halflife=config["specific_variance_halflife"],
        annualization=config["annualization_periods"],
        minimum_observations=config["minimum_specific_observations"],
        shrinkage=config["specific_variance_median_shrinkage"],
        floor=config["specific_variance_floor"],
    )

    missing_returns = target_universe.difference(returns.columns)
    missing_cap = target_universe.difference(market_cap.columns)
    if len(missing_returns) or len(missing_cap):
        missing = missing_returns.union(missing_cap)
        raise RiskModelError(f"target universe missing panel ticker(s): {list(missing[:10])}")
    target_industry = (
        _industry_asof(industry_history, requested_date).reindex(target_universe)
        if use_industry and industry_history is not None
        else None
    )
    target_cap = market_cap.loc[common_dates[-1]].reindex(target_universe)
    invalid_target = target_cap.isna() | ~np.isfinite(target_cap) | target_cap.le(0)
    if target_industry is not None:
        invalid_target |= target_industry.isna()
    if invalid_target.any():
        raise RiskModelError(
            "target universe lacks required industry or market cap: "
            f"{target_universe[invalid_target.to_numpy()].tolist()[:10]}"
        )
    target_styles, _ = _style_exposures_asof(
        returns,
        market_cap,
        common_dates[-1],
        target_universe,
        config,
    )
    exposures = pd.DataFrame(0.0, index=target_universe, columns=factors)
    if not use_industry:
        exposures["MARKET"] = 1.0
    exposures.loc[:, config["style_factors"]] = target_styles
    if target_industry is not None:
        for code in target_industry.astype(str).unique():
            factor_name = f"INDUSTRY:{code}"
            if factor_name not in exposures.columns:
                raise RiskModelError(f"target industry {code} is absent from factor history")
            exposures.loc[target_industry.astype(str).eq(code), factor_name] = 1.0

    specific = all_specific.reindex(target_universe)
    imputed = specific.isna()
    if target_industry is not None and industry_history is not None:
        for code in target_industry.astype(str).unique():
            members = target_industry.astype(str).eq(code)
            reference = all_specific.reindex(
                _industry_asof(industry_history, requested_date)
                .loc[lambda value: value.astype(str).eq(code)]
                .index
            ).dropna()
            fill = (
                float(reference.median())
                if not reference.empty
                else float(all_specific.median())
            )
            specific.loc[members & specific.isna()] = fill
    specific = specific.fillna(float(all_specific.median())).clip(
        lower=config["specific_variance_floor"]
    )
    imputation_fraction = float(imputed.mean())
    if imputation_fraction > config["max_specific_imputation_fraction"]:
        raise RiskModelError(
            f"specific-risk imputation fraction {imputation_fraction:.2%} exceeds "
            f"{config['max_specific_imputation_fraction']:.2%}"
        )

    x = exposures.to_numpy(dtype=float)
    covariance = x @ factor_cov.to_numpy(dtype=float) @ x.T
    covariance += np.diag(specific.to_numpy(dtype=float))
    covariance = 0.5 * (covariance + covariance.T)
    asset_cov = pd.DataFrame(
        covariance, index=target_universe, columns=target_universe
    )
    manifest = {
        "schema_version": 1,
        "status": "success",
        "model_type": "open_structural_multifactor",
        "is_proprietary_barra_model": False,
        "requested_date": requested_date,
        "history_start": str(factor_returns.index.min()),
        "history_end": str(factor_returns.index.max()),
        "regression_periods": int(len(factor_returns)),
        "factor_count": int(len(factors)),
        "factors": factors,
        "industry_mode_configured": industry_mode,
        "industry_history_used": use_industry,
        "target_asset_count": int(len(target_universe)),
        "regression_assets_min": int(min(regression_assets)),
        "regression_assets_median": float(np.median(regression_assets)),
        "regression_assets_max": int(max(regression_assets)),
        "specific_risk_imputed_count": int(imputed.sum()),
        "specific_risk_imputed_fraction": imputation_fraction,
        "specific_risk_observations_min_target": int(
            observations.reindex(target_universe, fill_value=0).min()
        ),
        "config": config,
        "timing_contract": (
            "return on date t is regressed on market cap, market-derived styles, "
            "and any enabled industry classification known on the preceding aligned "
            "trading date; no input date exceeds requested_date"
        ),
    }
    return RiskModelResult(
        asset_cov=asset_cov,
        factor_cov=factor_cov,
        specific_var=specific.rename("specific_var"),
        exposures=exposures,
        factor_returns=factor_returns,
        manifest=manifest,
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


def load_risk_model_context(
    *,
    config_path: str | Path,
    returns_file: str | Path,
    market_cap_file: str | Path,
    industry_file: str | Path | None,
) -> RiskModelContext:
    supplied: dict[str, str | Path] = {
        "config": config_path,
        "returns": returns_file,
        "market_cap": market_cap_file,
    }
    if industry_file is not None:
        supplied["industry"] = industry_file
    inputs = {
        name: {
            "path": str(Path(path).expanduser().resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in supplied.items()
    }
    return RiskModelContext(
        config=load_risk_config(config_path),
        returns=load_wide_panel(returns_file, "returns"),
        market_cap=load_wide_panel(market_cap_file, "market cap"),
        industry_history=(
            None if industry_file is None else load_industry_history(industry_file)
        ),
        inputs=inputs,
    )


def build_risk_model_from_context(
    *,
    context: RiskModelContext,
    target_universe: pd.Index,
    requested_date: object,
    output_dir: str | Path,
    manifest_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    date = normalize_date(requested_date)
    universe = pd.Index(
        [normalize_ticker(value) for value in target_universe], name="ticker"
    )
    if universe.empty or universe.has_duplicates:
        raise InputDataError("target universe must be non-empty and unique")
    result = estimate_structural_risk_model(
        returns=context.returns,
        market_cap=context.market_cap,
        industry_history=context.industry_history,
        target_universe=universe,
        requested_date=date,
        config=context.config,
    )
    manifest = {
        **result.manifest,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "inputs": context.inputs,
        "outputs": list(OUTPUT_FILES),
        **(manifest_metadata or {}),
    }
    destination, temporary = _prepare_output_directory(output_dir)
    try:
        result.asset_cov.to_parquet(temporary / "asset_cov.parquet")
        result.factor_cov.to_parquet(temporary / "factor_cov.parquet")
        result.specific_var.to_frame().to_parquet(temporary / "specific_var.parquet")
        result.exposures.to_parquet(temporary / "exposures.parquet")
        result.factor_returns.to_parquet(temporary / "factor_returns.parquet")
        (temporary / "risk_model_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "success",
        "date": date,
        "asset_count": int(len(result.asset_cov)),
        "factor_count": int(len(result.factor_cov)),
        "regression_periods": int(len(result.factor_returns)),
        "specific_risk_imputed_count": int(
            result.manifest["specific_risk_imputed_count"]
        ),
        "output_dir": str(destination),
    }


def build_risk_model_from_files(
    *,
    config_path: str | Path,
    returns_file: str | Path,
    market_cap_file: str | Path,
    industry_file: str | Path | None,
    universe_file: str | Path,
    requested_date: object,
    output_dir: str | Path,
) -> dict[str, Any]:
    date = normalize_date(requested_date)
    universe = _load_universe(universe_file, date)
    context = load_risk_model_context(
        config_path=config_path,
        returns_file=returns_file,
        market_cap_file=market_cap_file,
        industry_file=industry_file,
    )
    inputs = dict(context.inputs)
    inputs["universe"] = {
        "path": str(Path(universe_file).expanduser().resolve()),
        "sha256": sha256_file(universe_file),
    }
    context = RiskModelContext(
        config=context.config,
        returns=context.returns,
        market_cap=context.market_cap,
        industry_history=context.industry_history,
        inputs=inputs,
    )
    return build_risk_model_from_context(
        context=context,
        target_universe=universe,
        requested_date=date,
        output_dir=output_dir,
    )
