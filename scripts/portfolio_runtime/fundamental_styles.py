"""Point-in-time fundamental style exposures for the optional risk model.

The implementation is deliberately conservative: each as-of snapshot uses only
rows whose ``available_date`` is known by the requested date, and the output is
standardized by the risk-model caller.  These are Barra-style prototypes, not
commercial Barra definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .errors import InputDataError, RiskModelError
from .io import normalize_date, normalize_ticker


BASE_FIELDS = {
    "symbol",
    "available_date",
    "report_year",
    "report_quarter",
    "bs_total_assets",
    "bs_total_liab",
    "bs_total_hldr_eqy_exc_min_int",
    "is_total_revenue",
    "is_n_income_attr_p",
    "is_net_after_nr",
    "is_total_profit",
    "is_basic_eps",
    "bs_money_cap",
    "bs_total_cur_assets",
    "bs_total_cur_liab",
}

SUPPORTED_FUNDAMENTAL_STYLES = {
    "EARNYILD",
    "GROWTH",
    "BTOP",
    "LEVERAGE",
    "LIQUIDTY",
}


def _paths(root: str | Path) -> list[Path]:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise InputDataError(f"fundamental PIT path has no parquet files: {path}")
    return files


def load_fundamental_panel(root: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _paths(root):
        frame = pd.read_parquet(path)
        missing = sorted({"symbol", "available_date"} - set(frame.columns))
        if missing:
            raise InputDataError(
                f"fundamental PIT file missing column(s): {', '.join(missing)}: {path}"
            )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["ticker"] = result["symbol"].map(normalize_ticker)
    result["available_date"] = result["available_date"].map(normalize_date)
    result["report_year"] = pd.to_numeric(result.get("report_year"), errors="coerce")
    result["report_quarter"] = pd.to_numeric(
        result.get("report_quarter"), errors="coerce"
    )
    result = result.loc[
        result["ticker"].notna()
        & result["available_date"].notna()
        & result["report_year"].notna()
        & result["report_quarter"].notna()
    ].copy()
    result["period_key"] = (
        result["report_year"].astype(int) * 10
        + result["report_quarter"].astype(int)
    )
    for column in BASE_FIELDS:
        if column in result.columns and column not in {
            "symbol",
            "available_date",
            "report_year",
            "report_quarter",
        }:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    # Keep every available-date version. As-of selection below resolves revisions
    # without allowing a later revision to overwrite an earlier historical view.
    return result.sort_values(["available_date", "ticker", "period_key"]).reset_index(
        drop=True
    )


def _latest_rows(panel: pd.DataFrame, exposure_date: str, tickers: pd.Index) -> pd.DataFrame:
    available = panel.loc[
        panel["available_date"].le(exposure_date)
        & panel["ticker"].isin(tickers)
    ].copy()
    if available.empty:
        raise RiskModelError(
            f"fundamental PIT panel has no rows through {exposure_date}"
        )
    return (
        available.sort_values(
            ["ticker", "period_key", "available_date"], kind="stable"
        )
        .drop_duplicates("ticker", keep="last")
        .set_index("ticker")
        .reindex(tickers)
    )


def _same_period_prior(
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    exposure_date: str,
    column: str,
) -> pd.Series:
    if column not in panel.columns:
        return pd.Series(np.nan, index=latest.index, dtype=float)
    prior = panel.loc[
        panel["available_date"].le(exposure_date)
        & panel["ticker"].isin(latest.index)
    ].copy()
    prior["target_period"] = prior["period_key"] + 10
    wanted = pd.DataFrame(
        {
            "ticker": latest.index.astype(str),
            "target_period": latest["period_key"].to_numpy(dtype=int),
        }
    )
    prior = prior.merge(wanted, on=["ticker", "target_period"], how="inner")
    prior = prior.sort_values(["ticker", "available_date"], kind="stable")
    return prior.drop_duplicates("ticker", keep="last").set_index("ticker")[column].reindex(
        latest.index
    )


def _ttm_value(
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    exposure_date: str,
    column: str,
) -> pd.Series:
    """Convert cumulative income-statement values to trailing twelve months."""
    current = pd.to_numeric(latest.get(column), errors="coerce")
    if current is None:
        return pd.Series(np.nan, index=latest.index, dtype=float)
    # Keep names with no PIT filing missing; the caller applies the configured
    # coverage gate instead of coercing an absent period into an integer.
    valid_period = latest["report_year"].notna() & latest["report_quarter"].notna()
    result = pd.Series(np.nan, index=latest.index, dtype=float)
    if not valid_period.any():
        return result
    working = latest.loc[valid_period].copy()
    current_valid = current.loc[valid_period]
    annual = working.copy()
    annual["target_period"] = (annual["report_year"].astype(int) - 1) * 10 + 4
    same_period = working.copy()
    same_period["target_period"] = (
        (same_period["report_year"].astype(int) - 1) * 10
        + same_period["report_quarter"].astype(int)
    )
    known = panel.loc[
        panel["available_date"].le(exposure_date)
        & panel["ticker"].isin(working.index)
        & panel[column].notna()
    ].copy()
    known = known.sort_values(["ticker", "period_key", "available_date"], kind="stable")
    known = known.drop_duplicates(["ticker", "period_key"], keep="last")
    lookup = known.set_index(["ticker", "period_key"])[column]

    annual_index = pd.MultiIndex.from_arrays(
        [annual.index.astype(str), annual["target_period"].to_numpy(dtype=int)]
    )
    same_index = pd.MultiIndex.from_arrays(
        [same_period.index.astype(str), same_period["target_period"].to_numpy(dtype=int)]
    )
    prior_annual = pd.Series(lookup.reindex(annual_index).to_numpy(), index=working.index)
    prior_same = pd.Series(lookup.reindex(same_index).to_numpy(), index=working.index)
    quarter = working["report_quarter"].astype(int)
    ttm = current_valid.copy()
    non_annual = quarter.ne(4)
    # If the prior-year annual or same-quarter PIT row is unavailable, use only
    # the current cumulative filing and annualize it. This is a no-look-ahead
    # fallback and remains subject to the configured coverage gate.
    fallback = non_annual & (prior_annual.isna() | prior_same.isna()) & current_valid.notna()
    ttm.loc[fallback] = current_valid.loc[fallback] * (
        4.0 / quarter.loc[fallback].clip(lower=1)
    )
    ttm.loc[non_annual & ~fallback] = (
        prior_annual.loc[non_annual]
        + current_valid.loc[non_annual]
        - prior_same.loc[non_annual]
    )
    result.loc[working.index] = ttm
    return result


def build_fundamental_styles(
    panel: pd.DataFrame,
    *,
    exposure_date: str,
    tickers: pd.Index,
    market_cap: pd.Series,
    requested: list[str],
) -> pd.DataFrame:
    requested = [str(value).upper().strip() for value in requested]
    unknown = set(requested) - SUPPORTED_FUNDAMENTAL_STYLES
    if unknown:
        raise RiskModelError(f"unsupported fundamental style(s): {sorted(unknown)}")
    latest = _latest_rows(panel, exposure_date, tickers)
    cap = pd.to_numeric(market_cap.reindex(tickers), errors="coerce")
    output = pd.DataFrame(index=tickers)

    if "EARNYILD" in requested:
        earnings = _ttm_value(
            panel, latest, exposure_date, "is_n_income_attr_p"
        )
        if earnings.notna().sum() < 2:
            earnings = _ttm_value(panel, latest, exposure_date, "is_net_after_nr")
        output["EARNYILD"] = earnings / cap

    if "BTOP" in requested:
        equity = latest.get(
            "bs_total_hldr_eqy_exc_min_int",
            pd.Series(index=tickers, dtype=float),
        )
        output["BTOP"] = equity / cap

    if "LEVERAGE" in requested:
        liabilities = latest.get("bs_total_liab", pd.Series(index=tickers, dtype=float))
        assets = latest.get("bs_total_assets", pd.Series(index=tickers, dtype=float))
        output["LEVERAGE"] = liabilities / assets

    if "LIQUIDTY" in requested:
        cash = latest.get("bs_money_cap", pd.Series(index=tickers, dtype=float))
        current_liab = latest.get(
            "bs_total_cur_liab", pd.Series(index=tickers, dtype=float)
        )
        output["LIQUIDTY"] = cash / current_liab

    if "GROWTH" in requested:
        revenue = latest.get(
            "is_total_revenue", pd.Series(index=tickers, dtype=float)
        )
        prior_revenue = _same_period_prior(
            panel, latest, exposure_date, "is_total_revenue"
        )
        output["GROWTH"] = revenue / prior_revenue - 1.0

    return output.loc[:, requested]
