"""Render executed target weights with the original factor-backtest plotter."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Iterator

import numpy as np
import pandas as pd

from .errors import InputDataError


@dataclass(frozen=True)
class FactorBacktestReportConfig:
    factor_backtest_root: Path
    stats_file: Path
    holdings_file: Path
    transactions_file: Path
    signal_file: Path
    twap_file: Path
    output_dir: Path
    benchmark_name: str = "zz1000"
    benchmark_index_file: Path | None = None
    initial_cash: float | None = None


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _normalize_date(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("-", "", regex=False).str[:8]


def _load_original_plotter(root: Path):
    engine_dir = root / "scripts" / "backtest_engine"
    source = engine_dir / "BackTest.py"
    if not source.is_file():
        raise InputDataError(f"factor-backtest plotter is missing: {source}")

    module_name = "factor_backtest_original_plotter"
    with _working_directory(engine_dir):
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise InputDataError(f"cannot import factor-backtest plotter: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.TradingSystem


def _infer_initial_cash(stats: pd.DataFrame, configured: float | None) -> float:
    if configured is not None:
        if not np.isfinite(configured) or configured <= 0:
            raise InputDataError("initial_cash must be finite and positive")
        return float(configured)
    if {"unrealized_pnl", "nav"}.issubset(stats.columns):
        inferred = float(stats.iloc[0]["unrealized_pnl"] / stats.iloc[0]["nav"])
        if np.isfinite(inferred) and inferred > 0:
            return inferred
    raise InputDataError("cannot infer initial_cash; pass --initial-cash explicitly")


def _prepare_stats(
    stats_file: Path,
    initial_cash: float,
    benchmark_index_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not stats_file.is_file():
        raise InputDataError(f"stats file does not exist: {stats_file}")
    raw = pd.read_csv(stats_file)
    required = {"date", "unrealized_pnl"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise InputDataError("ledger stats missing column(s): " + ", ".join(missing))
    raw["date"] = _normalize_date(raw["date"])
    raw = raw.drop_duplicates("date", keep="last").sort_values("date")
    raw.index = pd.to_datetime(raw["date"], format="%Y%m%d")
    original_stats = raw[["unrealized_pnl"]].copy()
    if benchmark_index_file is not None:
        path = benchmark_index_file.expanduser().resolve()
        if not path.is_file():
            raise InputDataError(f"benchmark index file does not exist: {path}")
        index = pd.read_parquet(path)
        if not {"date", "close"}.issubset(index.columns):
            raise InputDataError("benchmark index requires date and close columns")
        index["date"] = _normalize_date(index["date"])
        index = index.drop_duplicates("date", keep="last").set_index("date")
        close = pd.to_numeric(index["close"], errors="coerce")
        close.index = pd.to_datetime(close.index, format="%Y%m%d")
        close = close.reindex(original_stats.index).ffill()
        if close.isna().any() or close.empty or float(close.iloc[0]) <= 0:
            raise InputDataError("benchmark index does not cover execution dates")
        benchmark = pd.DataFrame(
            {"benchmark": close / float(close.iloc[0]) * initial_cash},
            index=original_stats.index,
        )
    else:
        if "benchmark_nav" not in raw.columns:
            raise InputDataError("ledger stats requires benchmark_nav when benchmark index is omitted")
        values = pd.to_numeric(raw["benchmark_nav"], errors="coerce")
        if values.isna().all():
            raise InputDataError("benchmark_nav is empty; provide benchmark_index_file")
        benchmark = pd.DataFrame(
            {"benchmark": values.to_numpy() * initial_cash},
            index=original_stats.index,
        )
    return original_stats, benchmark


def _calculate_execution_ic(
    *,
    signal_file: Path,
    twap_file: Path,
    execution_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not signal_file.is_file():
        raise InputDataError(f"signal file does not exist: {signal_file}")
    if not twap_file.is_file():
        raise InputDataError(f"TWAP file does not exist: {twap_file}")
    signal = pd.read_parquet(signal_file, columns=["date", "ticker", "prediction"])
    signal["date"] = _normalize_date(signal["date"])
    signal["ticker"] = signal["ticker"].astype(str).str[:6]
    signal = signal.drop_duplicates(["date", "ticker"], keep="last")
    signal_wide = signal.pivot(index="date", columns="ticker", values="prediction").sort_index()

    prices = pd.read_parquet(twap_file)
    prices.index = prices.index.astype(str).str.replace("-", "", regex=False).str[:8]
    prices.columns = prices.columns.astype(str).str[:6]
    prices = prices.sort_index()
    common_dates = signal_wide.index.intersection(prices.index)
    common_tickers = signal_wide.columns.intersection(prices.columns)
    if len(common_dates) < 2 or len(common_tickers) < 2:
        raise InputDataError("signal and TWAP do not have enough common dates/tickers for IC")
    signal_wide = signal_wide.reindex(index=common_dates, columns=common_tickers)
    prices = prices.reindex(index=common_dates, columns=common_tickers)
    next_return = prices.shift(-1).div(prices).sub(1.0)
    signal_ic = signal_wide.rank(axis=1).corrwith(next_return.rank(axis=1), axis=1)

    execution_labels = execution_dates.strftime("%Y%m%d")
    execution_values: list[float] = []
    execution_signal: list[pd.Series] = []
    signal_dates = signal_ic.index.to_numpy()
    for execution_date in execution_labels:
        prior_dates = signal_dates[signal_dates < execution_date]
        if len(prior_dates):
            source_date = prior_dates[-1]
            execution_values.append(float(signal_ic.loc[source_date]))
            execution_signal.append(signal_wide.loc[source_date])
        else:
            execution_values.append(np.nan)
            execution_signal.append(pd.Series(np.nan, index=signal_wide.columns))
    return (
        pd.DataFrame({"1d": execution_values}, index=execution_dates),
        pd.DataFrame(execution_signal, index=execution_dates),
    )


def _compat_weekly_winrate(self, frame: pd.DataFrame):
    """Original method with pandas-3-compatible ISO-week extraction only."""
    temp = frame.copy()
    temp.index = pd.to_datetime(temp.index)
    temp["year"] = temp.index.year
    temp["month"] = temp.index.month
    temp["week"] = temp.index.isocalendar().week.to_numpy()
    temp["month"] = temp["year"].astype(str) + "-" + temp["month"].astype(str).str.zfill(2)
    temp["week"] = temp["year"].astype(str) + "-" + temp["week"].astype(str).str.zfill(2)
    week_series = temp.groupby("week")["hedged_unrealized_pnl"].apply(
        lambda values: (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
    )
    month_series = temp.groupby("month")["hedged_unrealized_pnl"].apply(
        lambda values: (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
    )
    week_winrate = len(week_series[week_series > 0]) / len(week_series)
    month_winrate = len(month_series[month_series > 0]) / len(month_series)
    plot = self._plot_module.plt
    plot.figure(figsize=(12, 4))
    plot.title(f"月超额收益 月胜率为：{month_winrate * 100:.2f}% 周胜率为：{week_winrate * 100:.2f}%")
    plot.bar(month_series.index, month_series)
    plot.xticks(rotation=45)
    plot.xticks(np.arange(0, len(month_series), step=max(int(len(month_series) / 12), 1)))
    plot.grid(True, linestyle="--")
    plot.savefig(Path(self.output_dir) / "winrate.png", bbox_inches="tight")
    return week_winrate, month_winrate, month_series


def render_factor_backtest_compatible_report(config: FactorBacktestReportConfig) -> dict[str, object]:
    """Render original factor-backtest PNL styling from executed target weights."""
    output_dir = config.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise InputDataError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    ledger = pd.read_csv(config.stats_file)
    initial_cash = _infer_initial_cash(ledger, config.initial_cash)
    stats, benchmark = _prepare_stats(
        config.stats_file, initial_cash, config.benchmark_index_file
    )
    ics, buy_data = _calculate_execution_ic(
        signal_file=config.signal_file,
        twap_file=config.twap_file,
        execution_dates=stats.index,
    )
    stats["IC"] = ics["1d"].reindex(stats.index)

    holdings = pd.read_csv(config.holdings_file)
    transactions = pd.read_csv(config.transactions_file)
    required_holdings = {"date", "ticker", "volume", "price_current"}
    required_transactions = {"date", "ticker", "B/S", "amount"}
    if missing := sorted(required_holdings - set(holdings.columns)):
        raise InputDataError("holdings file missing column(s): " + ", ".join(missing))
    if missing := sorted(required_transactions - set(transactions.columns)):
        raise InputDataError("transactions file missing column(s): " + ", ".join(missing))

    plotter = _load_original_plotter(config.factor_backtest_root.expanduser().resolve())
    trader = SimpleNamespace(
        stats=stats,
        holding_records=holdings,
        transaction_records=transactions,
        init_cash=initial_cash,
    )
    report = plotter.__new__(plotter)
    report.trader = trader
    report.output_dir = str(output_dir)
    report.benchmark = config.benchmark_name
    report.hedgesell = False
    report.hratio = 1.0
    report.hbili = 1.0
    report.init_cash = initial_cash
    report.savemode = 3
    report.date_list = stats.index.strftime("%Y%m%d").tolist()
    report.ICs = ics
    # These two matrices are only used by the original plot title to display
    # signal coverage; target-weight execution itself is already frozen.
    report.buy_data = buy_data
    report.mask_isopen = buy_data.notna().astype(float)
    report._plot_module = sys.modules[plotter.__module__]
    report.get_close_index = lambda *_args, **_kwargs: benchmark
    report.calc_weekly_winrate = MethodType(_compat_weekly_winrate, report)

    with _working_directory(config.factor_backtest_root.expanduser().resolve() / "scripts" / "backtest_engine"):
        plotter.plot(report)
    stats.to_csv(output_dir / "stats_factor_backtest_compatible.csv", index_label="date")
    ics.to_csv(output_dir / "ICs.csv", index_label="date")
    manifest = {
        "status": "success",
        "engine": "skill-factor-backtest TradingSystem.plot",
        "plot_compatibility": "original plotter with pandas ISO-week compatibility shim",
        "portfolio_nav_source": str(config.stats_file.resolve()),
        "benchmark_nav_source": "benchmark_nav column in executed target-weight ledger",
        "ic_source": {
            "signal_file": str(config.signal_file.resolve()),
            "twap_file": str(config.twap_file.resolve()),
            "definition": "rank IC of prior signal date versus next trading-day TWAP return",
        },
        "initial_cash": initial_cash,
        "benchmark_name": config.benchmark_name,
        "outputs": ["Pnl.png", "winrate.png", "stats_factor_backtest_compatible.csv", "ICs.csv"],
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
