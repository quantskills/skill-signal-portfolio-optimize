from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .errors import InputDataError
from .io import normalize_date, normalize_ticker, read_table


RETURN_COLUMNS = {"date", "ticker", "return"}
TARGET_COLUMNS = {"date", "ticker", "portfolio", "target_weight"}


def load_asset_returns(path: str) -> pd.DataFrame:
    frame = read_table(path)
    missing = sorted(RETURN_COLUMNS - set(frame.columns))
    if missing:
        raise InputDataError("asset returns missing column(s): " + ", ".join(missing))
    result = frame.loc[:, ["date", "ticker", "return"]].copy()
    result["date"] = result["date"].map(normalize_date)
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result["return"] = pd.to_numeric(result["return"], errors="coerce")
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError("asset returns contain duplicate date-ticker rows")
    if result["return"].isna().any() or not np.isfinite(result["return"]).all():
        raise InputDataError("asset returns contain missing or non-finite values")
    if result["return"].lt(-1.0).any():
        raise InputDataError("asset returns must be at least -100%")
    return result.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def normalize_targets(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(TARGET_COLUMNS - set(frame.columns))
    if missing:
        raise InputDataError("target weights missing column(s): " + ", ".join(missing))
    result = frame.loc[:, ["date", "ticker", "portfolio", "target_weight"]].copy()
    result["date"] = result["date"].map(normalize_date)
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result["portfolio"] = result["portfolio"].astype("string").str.strip()
    result["target_weight"] = pd.to_numeric(result["target_weight"], errors="coerce")
    if result["portfolio"].eq("").any():
        raise InputDataError("target weights contain empty portfolio names")
    if result.duplicated(["date", "portfolio", "ticker"]).any():
        raise InputDataError("target weights contain duplicate keys")
    if result["target_weight"].isna().any() or not np.isfinite(
        result["target_weight"]
    ).all():
        raise InputDataError("target weights contain missing or non-finite values")
    if result["target_weight"].lt(-1.0e-12).any():
        raise InputDataError("target weights contain negative values")
    totals = result.groupby(["date", "portfolio"])["target_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1.0e-8):
        raise InputDataError("target weights must sum to one for every date and portfolio")
    return result.sort_values(["date", "portfolio", "ticker"], kind="stable")


def apply_returns(weights: pd.Series, returns: pd.Series, *, date: str) -> tuple[float, pd.Series]:
    aligned = returns.reindex(weights.index)
    held_missing = aligned.isna() & weights.abs().gt(1.0e-12)
    if held_missing.any():
        raise InputDataError(
            f"asset returns are missing held ticker(s) on {date}: "
            f"{held_missing.index[held_missing].tolist()[:10]}"
        )
    aligned = aligned.fillna(0.0)
    gross_return = float(weights @ aligned)
    post_value = weights * (1.0 + aligned)
    total = float(post_value.sum())
    if not np.isfinite(total) or total <= 0:
        raise InputDataError(f"portfolio value is non-positive after returns on {date}")
    return gross_return, (post_value / total).rename("weight")


def drift_weights(
    weights: pd.Series,
    returns_wide: pd.DataFrame,
    dates: list[str],
) -> pd.Series:
    result = weights.astype(float).copy()
    for date in dates:
        _, result = apply_returns(result, returns_wide.loc[date], date=date)
    return result


def backtest_targets(
    targets: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    initial_weights: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise InputDataError("transaction_cost_bps must be finite and non-negative")
    normalized = normalize_targets(targets)
    returns = asset_returns.copy()
    if not RETURN_COLUMNS.issubset(returns.columns):
        raise InputDataError("asset_returns must contain date, ticker, and return")
    returns["date"] = returns["date"].map(normalize_date)
    returns["ticker"] = returns["ticker"].map(normalize_ticker)
    if returns.duplicated(["date", "ticker"]).any():
        raise InputDataError("asset returns contain duplicate date-ticker rows")
    returns_wide = returns.pivot(index="date", columns="ticker", values="return").sort_index()
    calendar = returns_wide.index.astype(str).tolist()
    positions = {date: index for index, date in enumerate(calendar)}
    target_dates = sorted(normalized["date"].unique().tolist())
    effective_dates: dict[str, str] = {}
    for date in target_dates:
        if date not in positions:
            raise InputDataError(f"target date {date} is absent from asset returns")
        position = positions[date] + 1
        if position >= len(calendar):
            raise InputDataError(f"target date {date} has no following execution date")
        effective_dates[date] = calendar[position]

    portfolios = sorted(normalized["portfolio"].unique().tolist())
    rows: list[dict[str, Any]] = []
    for portfolio in portfolios:
        portfolio_targets = normalized.loc[normalized["portfolio"].eq(portfolio)]
        schedules: dict[str, pd.Series] = {}
        for date, group in portfolio_targets.groupby("date", sort=True):
            schedules[effective_dates[str(date)]] = group.set_index("ticker")[
                "target_weight"
            ].astype(float)
        first_effective = min(schedules)
        current = None
        if initial_weights is not None and portfolio in initial_weights:
            current = initial_weights[portfolio].astype(float).copy()
        nav = 1.0
        peak = 1.0
        for date in calendar[positions[first_effective] :]:
            turnover = 0.0
            if date in schedules:
                target = schedules[date]
                if current is None:
                    turnover = 0.0
                else:
                    universe = current.index.union(target.index)
                    turnover = 0.5 * float(
                        (
                            target.reindex(universe, fill_value=0.0)
                            - current.reindex(universe, fill_value=0.0)
                        ).abs().sum()
                    )
                current = target
            if current is None:
                continue
            gross_return, current = apply_returns(
                current, returns_wide.loc[date], date=date
            )
            cost = turnover * float(transaction_cost_bps) / 10000.0
            net_return = gross_return - cost
            nav *= 1.0 + net_return
            peak = max(peak, nav)
            rows.append(
                {
                    "date": date,
                    "portfolio": portfolio,
                    "gross_return": gross_return,
                    "transaction_cost": cost,
                    "net_return": net_return,
                    "turnover": turnover,
                    "nav": nav,
                    "drawdown": nav / peak - 1.0,
                }
            )
    if not rows:
        raise InputDataError("backtest produced no daily rows")
    return pd.DataFrame(rows).sort_values(["date", "portfolio"], kind="stable")


def summarize_backtest(daily: pd.DataFrame, periods_per_year: int = 252) -> dict[str, Any]:
    result: dict[str, Any] = {"periods_per_year": int(periods_per_year), "portfolios": {}}
    for portfolio, frame in daily.groupby("portfolio", sort=True):
        returns = frame["net_return"].to_numpy(dtype=float)
        count = len(returns)
        ending_nav = float(frame.iloc[-1]["nav"])
        annual_return = ending_nav ** (periods_per_year / count) - 1.0
        annual_volatility = (
            float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
            if count > 1
            else 0.0
        )
        sharpe = (
            float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(periods_per_year))
            if count > 1 and np.std(returns, ddof=1) > 0
            else None
        )
        result["portfolios"][str(portfolio)] = {
            "observations": count,
            "date_start": str(frame["date"].min()),
            "date_end": str(frame["date"].max()),
            "ending_nav": ending_nav,
            "annualized_return": float(annual_return),
            "annualized_volatility": annual_volatility,
            "sharpe": sharpe,
            "maximum_drawdown": float(frame["drawdown"].min()),
            "total_one_way_turnover": float(frame["turnover"].sum()),
            "total_transaction_cost": float(frame["transaction_cost"].sum()),
        }
    return result
