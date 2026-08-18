# Rolling Backtest Contract

## Timing

- Form a target after observing the signal, benchmark, covariance, exposures, and tradability on trading date `t`.
- Apply that target before the asset return on the next available trading date.
- Never apply a target to the return on its signal date.
- Drift holdings after each asset return and pass the drifted optimized holdings into the next rebalance.

## Returns and costs

For pre-return weights `w` and simple asset returns `r`, calculate gross portfolio return as
`w'r`. Drift end-of-day weights in proportion to `w * (1+r)`.

Calculate one-way turnover at an effective rebalance as:

```text
0.5 * sum(abs(target_weight - drifted_current_weight))
```

Calculate proportional transaction cost as `turnover * bps / 10000` and subtract it from the
gross return on the effective date. Do not model orders, partial fills, or nonlinear market
impact in v0.6.

## Comparators

Apply identical dates, asset returns, execution timing, and transaction-cost assumptions to:

- `equal_weight_signal`
- `risk_optimized`
- `benchmark`

The benchmark is simulated from supplied rebalance weights. Disclose this method rather than
describing it as an official index return.

Compute daily active return as portfolio net return minus simulated benchmark net return.
Compute active NAV as portfolio NAV divided by benchmark NAV, geometrically annualized excess
return from that active NAV, realized tracking error as annualized active-return volatility,
and information ratio as annualized mean active return divided by active-return volatility.

## Failure rules

Fail when a target date lacks a following trading date, target weights do not sum to one, a
held asset lacks a return, or portfolio value becomes non-positive. Never fill held-asset
returns silently.

When the source return series ends permanently before the experiment end date, research may
use `scripts/prepare_terminal_returns.py` to create a separate derived return file. The adapter
adds a configured terminal return on the next market-calendar date, records every affected
ticker/date and both file hashes, and labels the value as an assumption rather than observed
market data. The source file remains unchanged. Use `-1` for a conservative total write-off;
report this assumption with performance results.
