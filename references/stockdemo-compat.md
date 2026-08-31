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

## Exact-window and rolling feedback

By default the standalone runner follows the original source-date window: it values every market date from the first execution through the final source date and does not execute the final source date on a date outside that window. Use `--include-final-next-execution` only when an extended research window is intentional. The implementation preserves the original sell-before-buy sequence, lot rounding, possible negative residual cash, buy-side turnover convention, and legacy volatility formula. These conventions are recorded as `metric_convention: stockdemo_legacy`; do not mix the resulting Sharpe ratio with metrics calculated under another convention.

`scripts/run_rolling_experiment.py --stockdemo-market-file ...` enables execution-aware feedback. The first optimization retains the configured initial-weight convention. Before each later optimization, the prior target executes on the next market date and the state advances through any intervening dates. Actual close holdings then replace theoretical drifted targets as `current_weight`. Because the optimizer remains a fully invested stock optimizer, actual stock market values are normalized to one; residual cash or leverage is written separately as `actual_cash_weight`. The final output adds `execution_feedback.parquet` and a strict replay under `stockdemo_compat/`.

When a non-tradable holding remains outside the candidate set, its frozen weight reduces the attainable candidate allocation. An exact configured candidate weight such as 100% is automatically reduced to `1 - frozen_outside_candidate_weight`; diagnostics retain both configured and effective bounds. This adjustment does not relax any tradable stock, style, industry, tracking-error, or turnover constraint.

## Parity boundary

Exact numerical parity requires the same frozen signal, market table, calendar, benchmark data, adjustment-factor convention, and date window as the original stockdemo run. The compatibility runner is an independent implementation with unit tests; parity should be established on a small fixture by comparing daily holdings, transactions, cash, and NAV against the original `TradingSystem` before interpreting a full-period result.
