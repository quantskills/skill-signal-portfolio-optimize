"""Stockdemo-compatible order execution and accounting.

The optimizer produces target weights, while the original stockdemo backtest
simulates orders.  This module keeps that execution boundary separate from
the risk model and optimizer so the legacy native backtest remains reproducible.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .errors import InputDataError
from .io import normalize_date, normalize_ticker, read_table


@dataclass(frozen=True)
class StockDemoExecutionConfig:
    """Execution defaults copied from stockdemo/BackTest/config/backtestconfig.ini."""

    longx: int = 200
    stock_pool: str = "whole"
    trade_price_type: str = "twap"
    buy_sell_shift: int = 1
    transaction: float = 1.4
    keep: float = 0.8
    initial_cash: float = 100_000_000.0
    locked_limit: float = 0.095
    periods_per_year: int = 252

    @property
    def one_side_fee(self) -> float:
        # stockdemo uses transaction / 2 / 1000 for each side.
        return self.transaction / 2.0 / 1000.0

    def validate(self) -> None:
        if self.longx <= 0:
            raise InputDataError("longx must be positive")
        if self.stock_pool != "whole":
            raise InputDataError(
                "stockdemo_compat currently supports stock_pool=whole only; "
                "provide a dedicated pool adapter before using another pool"
            )
        if self.trade_price_type != "twap":
            raise InputDataError("stockdemo_compat currently requires trade_price_type=twap")
        if self.buy_sell_shift != 1:
            raise InputDataError("stockdemo_compat currently requires buy_sell_shift=1")
        if not 0.0 <= self.keep <= 1.0:
            raise InputDataError("keep must be between zero and one")
        if not np.isfinite(self.transaction) or self.transaction < 0:
            raise InputDataError("transaction must be finite and non-negative")
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise InputDataError("initial_cash must be positive and finite")
        if not 0.0 < self.locked_limit < 1.0:
            raise InputDataError("locked_limit must be between zero and one")


MARKET_REQUIRED = {
    "date",
    "ticker",
    "open",
    "close",
    "pre_close",
    "is_open",
    "is_st",
}

MARKET_INPUT_COLUMNS = {
    "date",
    "ticker",
    "symbol",
    "open",
    "openPrice",
    "close",
    "closePrice",
    "pre_close",
    "preClosePrice",
    "twap",
    "TWAP",
    "vwap",
    "VWAP",
    "is_open",
    "isOpen",
    "is_st",
    "isST",
    "ST",
    "adj_factor",
    "adjfactor",
    "accumAdjFactor",
}


def _read_many(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        files = sorted(source.glob("*.parquet")) + sorted(source.glob("*.csv"))
        if not files:
            raise InputDataError(f"no CSV or Parquet files found in {source}")
        return pd.concat([read_table(file) for file in files], ignore_index=True)
    return read_table(source)


def _read_market_slice(path: str | Path, start: str, end: str) -> pd.DataFrame:
    """Read only the requested dates and execution columns from Parquet."""

    source = Path(path).expanduser().resolve()
    files = (
        sorted(source.glob("*.parquet")) + sorted(source.glob("*.csv"))
        if source.is_dir()
        else [source]
    )
    if not files:
        raise InputDataError(f"no CSV or Parquet files found in {source}")
    frames: list[pd.DataFrame] = []
    for file in files:
        if file.suffix.lower() not in {".parquet", ".pq"}:
            frames.append(read_table(file))
            continue
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pq.read_schema(file)
            columns = [name for name in schema.names if name in MARKET_INPUT_COLUMNS]
            if "date" not in columns:
                raise InputDataError(f"stockdemo market missing column: date ({file})")
            date_type = schema.field("date").type
            if pa.types.is_integer(date_type):
                lower: object = int(start)
                upper: object = int(end)
            else:
                lower = start
                upper = end
            frames.append(
                pd.read_parquet(
                    file,
                    columns=columns,
                    filters=[("date", ">=", lower), ("date", "<=", upper)],
                )
            )
        except InputDataError:
            raise
        except Exception as exc:
            raise InputDataError(f"cannot read stockdemo market slice {file}: {exc}") from exc
    return pd.concat(frames, ignore_index=True)


def _coerce_bool(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        unknown = values.notna() & ~values.isin([0, 1])
        if unknown.any():
            raise InputDataError(f"{label} contains numeric values other than zero/one")
        return values.fillna(0).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    result = normalized.isin({"true", "1", "yes", "y", "t"})
    unknown = values.notna() & ~normalized.isin(
        {"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f", ""}
    )
    if unknown.any():
        raise InputDataError(f"{label} contains unrecognised boolean values")
    return result.astype(bool)


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def load_stockdemo_market(
    path: str | Path,
    *,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """Load and normalize the raw market table required by stockdemo execution."""

    start = normalize_date(start_date)
    end = normalize_date(end_date)
    frame = _read_market_slice(path, start, end).copy()
    # The unified long store includes both numeric ``ticker`` and exchange-
    # qualified ``symbol``. Frozen model signals use the qualified symbol.
    if "symbol" in frame.columns:
        frame["ticker"] = frame["symbol"].where(frame["symbol"].notna(), frame.get("ticker"))
    aliases = {
        "open": ("open", "openPrice"),
        "close": ("close", "closePrice"),
        "pre_close": ("pre_close", "preClosePrice"),
        # The unified stock-level market store calls the same execution price
        # ``vwap``; stockdemo's legacy files expose it as TWAP.
        "twap": ("twap", "TWAP", "vwap", "VWAP"),
        "is_open": ("is_open", "isOpen"),
        "is_st": ("is_st", "isST", "ST"),
        "adj_factor": ("adj_factor", "adjfactor", "accumAdjFactor"),
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        selected = _first_column(frame, candidates)
        if selected is not None:
            rename[selected] = canonical
    frame = frame.rename(columns=rename)
    missing = sorted(MARKET_REQUIRED - set(frame.columns))
    if missing:
        raise InputDataError(
            "stockdemo market missing column(s): " + ", ".join(missing)
        )
    frame["date"] = frame["date"].map(normalize_date)
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame = frame.loc[frame["date"].between(start, end)].copy()
    if frame.empty:
        raise InputDataError("stockdemo market has no rows in requested date range")
    if frame.duplicated(["date", "ticker"]).any():
        raise InputDataError("stockdemo market contains duplicate date-ticker rows")
    for column in ("open", "close", "pre_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "twap" not in frame:
        frame["twap"] = frame["open"]
    else:
        frame["twap"] = pd.to_numeric(frame["twap"], errors="coerce")
    frame["is_open"] = _coerce_bool(frame["is_open"], "is_open")
    frame["is_st"] = _coerce_bool(frame["is_st"], "is_st")
    adj_name = "adj_factor"
    if adj_name not in frame:
        frame[adj_name] = 1.0
    frame[adj_name] = pd.to_numeric(frame[adj_name], errors="coerce").fillna(1.0)
    if frame[adj_name].le(0).any() or not np.isfinite(frame[adj_name]).all():
        raise InputDataError("adj_factor must be positive and finite")
    # BackTest falls back to open when TWAP is unavailable and uses pre-close
    # for a zero/missing open when evaluating the opening limit state.
    frame["twap"] = frame["twap"].fillna(frame["open"])
    required_numeric = ["close", "pre_close"]
    if frame[required_numeric].isna().any().any() or not np.isfinite(
        frame[required_numeric]
    ).to_numpy().all():
        raise InputDataError("stockdemo market contains missing or non-finite prices")
    if frame["pre_close"].le(0).any():
        raise InputDataError("pre_close must be positive")
    # Match BackTest.load_auxilliary: adjusted prices are used for volume sizing,
    # transaction amounts and mark-to-market values.
    opening_price = frame["open"].replace(0.0, np.nan).fillna(frame["pre_close"])
    if frame["twap"].isna().any():
        raise InputDataError("stockdemo market TWAP cannot be recovered from open")
    if frame["twap"].le(0).any() or frame["close"].le(0).any():
        raise InputDataError("stockdemo market trade and close prices must be positive")
    frame["ideal_trade_price"] = opening_price / frame[adj_name]
    frame["trade_price"] = frame["twap"] / frame[adj_name]
    frame["balance_price"] = frame["close"] / frame[adj_name]
    frame["zt"] = opening_price / frame["pre_close"] > 1.0 + 0.095
    frame["dt"] = opening_price / frame["pre_close"] < 1.0 - 0.095
    frame["can_buy"] = frame["is_open"] & ~frame["zt"]
    frame["can_sell"] = frame["is_open"] & ~frame["dt"]
    return frame.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def load_stockdemo_signal(path: str | Path) -> pd.DataFrame:
    frame = _read_many(path).copy()
    signal_column = _first_column(frame, ("prediction", "mean", "signal", "proba"))
    if signal_column is None:
        raise InputDataError(
            "signal must contain one of prediction, mean, signal, or proba"
        )
    required = {"date", "ticker", signal_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputDataError("signal missing column(s): " + ", ".join(missing))
    result = frame[["date", "ticker", signal_column]].rename(
        columns={signal_column: "signal"}
    )
    result["date"] = result["date"].map(normalize_date)
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result["signal"] = pd.to_numeric(result["signal"], errors="coerce")
    if result["signal"].isna().any() or not np.isfinite(result["signal"]).all():
        raise InputDataError("signal contains missing or non-finite values")
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError("signal contains duplicate date-ticker rows")
    return result.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def load_target_weights(path: str | Path, portfolio: str) -> pd.DataFrame:
    frame = _read_many(path).copy()
    required = {"date", "ticker", "target_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputDataError("target weights missing column(s): " + ", ".join(missing))
    if "portfolio" in frame.columns:
        frame = frame.loc[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise InputDataError(f"target weights contain no rows for portfolio {portfolio!r}")
    result = frame[["date", "ticker", "target_weight"]].copy()
    result["date"] = result["date"].map(normalize_date)
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result["target_weight"] = pd.to_numeric(result["target_weight"], errors="coerce")
    if result["target_weight"].isna().any() or not np.isfinite(result["target_weight"]).all():
        raise InputDataError("target weights contain missing or non-finite values")
    if result["target_weight"].lt(-1.0e-12).any():
        raise InputDataError("target weights contain negative values")
    if result.duplicated(["date", "ticker"]).any():
        raise InputDataError("target weights contain duplicate date-ticker rows")
    totals = result.groupby("date")["target_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1.0e-7, rtol=0.0):
        raise InputDataError("target weights must sum to one for every date")
    return result.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def _effective_signal_date(signal_dates: list[str], execution_date: str) -> str | None:
    prior = [value for value in signal_dates if value < execution_date]
    return prior[-1] if prior else None


def _market_by_date(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {date: group.set_index("ticker") for date, group in market.groupby("date", sort=True)}


def _signal_map(signal: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        date: group.set_index("ticker")["signal"].astype(float)
        for date, group in signal.groupby("date", sort=True)
    }


def _target_map(targets: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        date: group.set_index("ticker")["target_weight"].astype(float)
        for date, group in targets.groupby("date", sort=True)
    }


def _lot(ticker: str) -> int:
    return 200 if ticker.zfill(6).startswith("688") else 100


def _equal_target(
    signal: pd.Series,
    day: pd.DataFrame,
    holdings: dict[str, float],
    *,
    longx: int,
    keep: float,
    first_day: bool,
) -> pd.Series:
    usable = signal.index.intersection(day.index)
    signal = signal.loc[usable].dropna()
    signal = signal.loc[
        day.loc[signal.index, "is_open"]
        & ~day.loc[signal.index, "zt"]
        & ~day.loc[signal.index, "dt"]
        & ~day.loc[signal.index, "is_st"]
    ]
    ranked = signal.sort_values(ascending=False, kind="mergesort")
    if first_day or not holdings:
        selected = ranked.head(longx).index.tolist()
        if not selected:
            raise InputDataError("stockdemo signal has no executable Top-N names")
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)

    current_pool = [ticker for ticker, volume in holdings.items() if volume > 0]
    current = day.reindex(current_pool)
    values = pd.to_numeric(current["ideal_trade_price"], errors="coerce") * pd.Series(
        holdings, dtype=float
    ).reindex(current_pool)
    values = values.fillna(0.0)
    total_value = float(values.sum())
    sellable = current.loc[
        (current["is_open"] == True) & (current["zt"] == False)
    ].index.tolist()
    drop_out = current.loc[
        current["is_st"].astype(bool) | ~current.index.isin(day.index)
    ].index.tolist()
    sellable = [ticker for ticker in sellable if ticker not in drop_out]
    unit = total_value * (1.0 - keep) - float(values.reindex(drop_out).sum())
    sell_tickers = list(drop_out)
    if unit > 0 and sellable:
        sell_order = pd.Series(
            {ticker: float(signal.get(ticker, -np.inf)) for ticker in sellable}
        ).sort_values(ascending=True, kind="mergesort")
        cumulative = 0.0
        for ticker in sell_order.index:
            if cumulative >= unit:
                break
            sell_tickers.append(ticker)
            cumulative += float(values.get(ticker, 0.0))
    survivors = [ticker for ticker in current_pool if ticker not in sell_tickers]
    buy_num = max(longx - len(survivors), 0)
    buy_tickers = [ticker for ticker in ranked.index if ticker not in current_pool][:buy_num]
    selected = survivors + buy_tickers
    if not selected:
        raise InputDataError("stockdemo keep logic produced an empty target")
    return pd.Series(1.0 / len(selected), index=selected, dtype=float)


def _apply_corporate_action(
    holdings: dict[str, float],
    holding_adj: dict[str, float],
    day: pd.DataFrame,
) -> None:
    for ticker in list(holdings):
        if ticker not in day.index:
            continue
        current_adj = float(day.loc[ticker, "adj_factor"])
        previous_adj = float(holding_adj.get(ticker, current_adj))
        if np.isfinite(current_adj) and previous_adj != current_adj:
            holdings[ticker] *= current_adj / previous_adj
            holding_adj[ticker] = current_adj


def _place_orders(
    *,
    day: pd.DataFrame,
    target: pd.Series,
    holdings: dict[str, float],
    holding_adj: dict[str, float],
    cash: float,
    fee_rate: float,
    initial_value: float | None,
) -> tuple[float, list[dict[str, Any]], float, float]:
    current_names = pd.Index(list(holdings), dtype="string")
    target = target[target.index.isin(day.index)].astype(float)
    target = target[target.gt(0)]
    prices = day["ideal_trade_price"]
    total_value = float(cash)
    for ticker, volume in holdings.items():
        if ticker in day.index:
            total_value += float(volume) * float(prices[ticker])
    if initial_value is not None:
        total_value = initial_value
    universe = pd.Index(sorted(set(holdings) | set(target.index)))
    target_weight = target.reindex(universe, fill_value=0.0)
    current_volume = pd.Series(holdings, dtype=float).reindex(universe, fill_value=0.0)
    ideal_price = prices.reindex(universe)
    desired = total_value * target_weight / ideal_price
    raw_trade = desired - current_volume
    orders: list[dict[str, Any]] = []
    for ticker in universe:
        price = float(day.loc[ticker, "trade_price"]) if ticker in day.index else math.nan
        if not np.isfinite(price) or not np.isfinite(float(ideal_price[ticker])):
            continue
        one_lot = _lot(str(ticker))
        raw = float(raw_trade[ticker])
        side = "buy" if raw > 0 else "sell"
        volume = round(abs(raw) / one_lot) * one_lot
        if float(target_weight[ticker]) == 0.0 and side == "sell":
            volume = float(current_volume[ticker])
        if side == "sell":
            volume = min(volume, float(current_volume[ticker]))
        if volume <= 0:
            continue
        row = {
            "ticker": str(ticker),
            "B/S": side,
            "volume": float(volume),
            "trade_price": price,
            "adj_factor": float(day.loc[ticker, "adj_factor"]),
        }
        row["amount"] = row["volume"] * row["trade_price"]
        row["transaction"] = row["amount"] * fee_rate
        row["can_buy"] = bool(day.loc[ticker, "can_buy"]) and not bool(day.loc[ticker, "is_st"])
        row["can_sell"] = bool(day.loc[ticker, "can_sell"])
        orders.append(row)

    executed: list[dict[str, Any]] = []
    turnover_amount = 0.0
    fees = 0.0
    # stockdemo sells before buys, and filters each side independently.
    for row in [item for item in orders if item["B/S"] == "sell" and item["can_sell"]]:
        volume = min(float(row["volume"]), float(holdings.get(row["ticker"], 0.0)))
        if volume <= 0:
            continue
        amount = volume * float(row["trade_price"])
        fee = amount * fee_rate
        holdings[row["ticker"]] = float(holdings.get(row["ticker"], 0.0)) - volume
        cash += amount - fee
        turnover_amount += amount
        fees += fee
        executed.append({**row, "volume": volume, "amount": amount, "transaction": fee})
    for row in [item for item in orders if item["B/S"] == "buy" and item["can_buy"]]:
        volume = float(row["volume"])
        price = float(row["trade_price"])
        while volume > 0 and volume * price * (1.0 + fee_rate) > cash + 1.0e-9:
            volume -= _lot(row["ticker"])
        if volume <= 0:
            continue
        amount = volume * price
        fee = amount * fee_rate
        cash -= amount + fee
        holdings[row["ticker"]] = float(holdings.get(row["ticker"], 0.0)) + volume
        holding_adj[row["ticker"]] = float(row["adj_factor"])
        turnover_amount += amount
        fees += fee
        executed.append({**row, "volume": volume, "amount": amount, "transaction": fee})
    for ticker in list(holdings):
        if holdings[ticker] <= 0:
            holdings.pop(ticker, None)
            holding_adj.pop(ticker, None)
    return cash, executed, turnover_amount, fees


def _stockdemo_metrics(stats: pd.DataFrame, periods_per_year: int) -> dict[str, float | int | str | None]:
    if stats.empty:
        raise InputDataError("stockdemo backtest produced no statistics")
    values = stats["unrealized_pnl"].astype(float)
    if len(values) < 2:
        return {
            "observations": int(len(values)),
            "date_start": str(stats["date"].iloc[0]),
            "date_end": str(stats["date"].iloc[-1]),
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "maximum_drawdown": 0.0,
        }
    days = max((pd.Timestamp(str(stats["date"].iloc[-1])) - pd.Timestamp(str(stats["date"].iloc[0]))).days, 1)
    years = days / 365.0
    total_return = float(values.iloc[-1] / values.iloc[0])
    annual_return = math.exp(math.log(total_return) / years) - 1.0 if total_return > 0 else -1.0
    simple_step = values.shift(1) / values
    annual_volatility = float(simple_step.std(ddof=1) * math.sqrt((len(values) - 1) / years))
    drawdown = 1.0 - values / values.rolling(10000, min_periods=1).max()
    return {
        "observations": int(len(values)),
        "date_start": str(stats["date"].iloc[0]),
        "date_end": str(stats["date"].iloc[-1]),
        "annualized_return": float(annual_return),
        "annualized_volatility": annual_volatility,
        "sharpe": float(annual_return / annual_volatility) if annual_volatility > 0 else None,
        "maximum_drawdown": float(drawdown.max()),
        "total_transaction_cost": float(stats["transaction_cost"].sum()),
        "average_turnover": float(stats["turnover"].mean()),
    }


def run_stockdemo_compat(
    *,
    market: pd.DataFrame,
    output_dir: str | Path,
    config: StockDemoExecutionConfig,
    signal: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    portfolio_name: str = "signal",
) -> dict[str, Any]:
    """Run signal or target-weight execution using stockdemo accounting rules."""

    config.validate()
    if (signal is None) == (targets is None):
        raise InputDataError("provide exactly one of signal or targets")
    required_market = MARKET_REQUIRED | {
        "adj_factor",
        "ideal_trade_price",
        "trade_price",
        "balance_price",
        "zt",
        "dt",
        "can_buy",
        "can_sell",
    }
    missing_market = sorted(required_market - set(market.columns))
    if missing_market:
        raise InputDataError(
            "market must be loaded with load_stockdemo_market; missing column(s): "
            + ", ".join(missing_market)
        )
    if signal is not None:
        signal = signal.copy()
        if not {"date", "ticker", "signal"}.issubset(signal.columns):
            raise InputDataError("signal DataFrame must contain date, ticker, signal")
        signal["date"] = signal["date"].map(normalize_date)
        signal["ticker"] = signal["ticker"].map(normalize_ticker)
    if targets is not None:
        targets = targets.copy()
        if not {"date", "ticker", "target_weight"}.issubset(targets.columns):
            raise InputDataError(
                "targets DataFrame must contain date, ticker, target_weight"
            )
        targets["date"] = targets["date"].map(normalize_date)
        targets["ticker"] = targets["ticker"].map(normalize_ticker)
    market = market.copy()
    by_date = _market_by_date(market)
    source_dates = sorted(
        (signal["date"] if signal is not None else targets["date"]).unique()
    )
    market_dates = sorted(by_date)
    market_positions = {date: index for index, date in enumerate(market_dates)}
    execution_map: dict[str, str] = {}
    for source_date in source_dates:
        if source_date not in market_positions:
            raise InputDataError(
                f"signal/target date {source_date} is absent from stockdemo market"
            )
        execution_position = market_positions[source_date] + config.buy_sell_shift
        if execution_position >= len(market_dates):
            continue
        execution_map[market_dates[execution_position]] = source_date
    execution_dates = sorted(execution_map)
    if not execution_dates:
        raise InputDataError("no market execution dates follow the supplied signal/target dates")
    if signal is not None:
        signal_by_date = _signal_map(signal)
    else:
        target_by_date = _target_map(targets)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    holdings: dict[str, float] = {}
    holding_adj: dict[str, float] = {}
    cash = float(config.initial_cash)
    rows: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    first = True
    for execution_date in execution_dates:
        signal_date = execution_map.get(execution_date)
        if signal_date is None:
            continue
        day = by_date[execution_date]
        _apply_corporate_action(holdings, holding_adj, day)
        if signal is not None:
            target = _equal_target(
                signal_by_date[signal_date],
                day,
                holdings,
                longx=config.longx,
                keep=config.keep,
                first_day=first,
            )
            initial_value = cash if first else None
        else:
            target = target_by_date[signal_date]
            initial_value = cash if first else None
        pre_value = cash + sum(
            float(volume) * float(day.loc[ticker, "ideal_trade_price"])
            for ticker, volume in holdings.items()
            if ticker in day.index
        )
        cash, day_orders, turnover_amount, fees = _place_orders(
            day=day,
            target=target,
            holdings=holdings,
            holding_adj=holding_adj,
            cash=cash,
            fee_rate=config.one_side_fee,
            initial_value=initial_value,
        )
        for order in day_orders:
            transactions.append({"date": execution_date, **order})
        market_value = sum(
            float(volume) * float(day.loc[ticker, "balance_price"])
            for ticker, volume in holdings.items()
            if ticker in day.index
        )
        nav_value = cash + market_value
        rows.append(
            {
                "date": execution_date,
                "cash": cash,
                "market_value": market_value,
                "unrealized_pnl": nav_value,
                "transaction_cost": fees,
                "turnover": turnover_amount / max(pre_value, 1.0),
                "holdings": len(holdings),
            }
        )
        for ticker, volume in holdings.items():
            if ticker in day.index:
                holding_rows.append(
                    {
                        "date": execution_date,
                        "ticker": ticker,
                        "volume": volume,
                        "holding_period": 0,
                        "price_current": float(day.loc[ticker, "balance_price"]),
                        "adj_factor": float(holding_adj.get(ticker, day.loc[ticker, "adj_factor"])),
                    }
                )
        first = False

    stats = pd.DataFrame(rows)
    stats["nav"] = stats["unrealized_pnl"] / config.initial_cash
    stats["daily_return"] = stats["nav"].pct_change().fillna(0.0)
    stats["drawdown"] = stats["nav"] / stats["nav"].cummax() - 1.0
    if benchmark is not None:
        benchmark_frame = benchmark.copy()
        if "date" not in benchmark_frame or not {"close", "benchmark"}.intersection(benchmark_frame.columns):
            raise InputDataError("benchmark must contain date and close or benchmark")
        benchmark_frame["date"] = benchmark_frame["date"].map(normalize_date)
        price_column = "close" if "close" in benchmark_frame else "benchmark"
        benchmark_frame[price_column] = pd.to_numeric(benchmark_frame[price_column], errors="coerce")
        benchmark_frame = benchmark_frame.sort_values("date")
        benchmark_frame["benchmark_nav"] = benchmark_frame[price_column] / benchmark_frame[price_column].iloc[0]
        stats = stats.merge(benchmark_frame[["date", "benchmark_nav"]], on="date", how="left")
        stats["active_nav"] = stats["nav"] / stats["benchmark_nav"]
    else:
        stats["benchmark_nav"] = np.nan
        stats["active_nav"] = np.nan
    stats.to_csv(output / "stats.csv", index=False)
    pd.DataFrame(transactions).to_csv(
        output / "transaction.csv",
        index=False,
        columns=["date", "ticker", "B/S", "volume", "trade_price", "adj_factor", "amount", "transaction"],
    )
    pd.DataFrame(holding_rows).to_csv(output / "holdings.csv", index=False)
    summary = {
        "status": "success",
        "portfolio": portfolio_name,
        "engine": "stockdemo_compat",
        "config": asdict(config),
        "metrics": _stockdemo_metrics(stats, config.periods_per_year),
        "output_dir": str(output),
        "transaction_count": int(len(transactions)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
