# Stockdemo-compatible execution

v1.3.0 adds a separate order-level replay path. It does not change signal calibration, risk-factor construction, covariance estimation, optimizer constraints, or the legacy `native` backtest.

## Configuration

The defaults match `/home/wangxueliang/stockdemo/BackTest/config/backtestconfig.ini` for the `中证1000全市场选股` strategy:

| Parameter | Value |
| --- | --- |
| `longx` | `200` |
| `stock_pool` | `whole` |
| `trade_price_type` | `twap` |
| `buy_sell_shift` | `1` |
| `transaction` | `1.4` |
| `keep` | `0.8` |
| `initial_cash` | `100000000` |

`transaction=1.4` is converted to `transaction / 2 / 1000 = 0.0007` on each side, as in `stockdemo/BackTest/code/BackTest.py`.

## Inputs

The market table must contain:

```text
date | ticker | open | close | pre_close | is_open | is_st
```

An optional `twap` column is used for execution; `TWAP`, `vwap`, and `VWAP` are accepted aliases, and the price falls back to `open` when absent or null. When both a numeric `ticker` and exchange-qualified `symbol` are present, `symbol` is used to align with frozen model signals. Numeric `0/1` and boolean values are accepted for `is_open` and `is_st`. An optional `adj_factor` (or `accumAdjFactor`) is applied to opening sizing, TWAP execution, and close valuation. Opening price zero is replaced with `pre_close` for opening-limit checks. `zt` and `dt` use the original 9.5% threshold.

Parquet inputs are read with date and column projection so a short replay does not materialize the full stock-market history in memory.

Signals accept `prediction`, `mean`, `signal`, or `proba` as the score column. Target weights accept `date,ticker,target_weight` and optionally filter a `portfolio` column.

## Execution

For each signal/target date `t`, the next available market date is the execution date. Signal mode selects the highest-scoring executable names on the first day, then applies the original `keep` replacement budget. Target mode uses the supplied weights but shares the same order and accounting path.

Orders are rounded to 100 shares, or 200 shares for tickers beginning with `688`. Sells execute before buys. Buy orders require `is_open` and no opening limit-up; sells require `is_open` and no opening limit-down. ST names are excluded from signal selection. Cash, fees, holdings, adjusted prices, and mark-to-market values are recorded for every execution date.

Outputs are `stats.csv`, `transaction.csv`, `holdings.csv`, and `summary.json`. `summary.json` reports the stockdemo-style geometric annualized return, volatility, Sharpe ratio, maximum drawdown, average turnover, and total transaction cost.

## Parity boundary

Exact numerical parity requires the same frozen signal, market table, calendar, benchmark data, adjustment-factor convention, and date window as the original stockdemo run. The compatibility runner is an independent implementation with unit tests; parity should be established on a small fixture by comparing daily holdings, transactions, cash, and NAV against the original `TradingSystem` before interpreting a full-period result.
