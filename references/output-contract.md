# Output Contract

Successful runs write five files into a new or empty output directory.

## `target_weights.parquet`

Long-form rows for both portfolios:

```text
date | ticker | portfolio | target_weight | benchmark_weight |
current_weight | has_signal | raw_prediction | signal_score | expected_return | tradable
```

`portfolio` is either `equal_weight_signal` or `risk_optimized`.

## `constraint_diagnostics.json`

Contains solver status, iterations, objective value, hard-constraint tolerances, turnover, and per-industry/style benchmark exposure, portfolio exposure, active exposure, lower/upper bounds, slack, binding status, and violations for both portfolios. Raw and controllable stock-weight maxima are both recorded. A drifted non-tradable position outside a stock bound is held exactly and listed under `frozen_bound_exceptions`; the exception does not relax bounds for tradable assets.

Only `risk_optimized` must satisfy every configured optimizer constraint. Baseline violations are reported for comparison.

## `risk_summary.json`

Contains expected return, ex-ante absolute volatility, active volatility, and changes from the equal-weight signal portfolio.

## `signal_diagnostics.json`

Contains signal type, direction, requested date, observation count, raw/calibrated distribution summaries, winsorization count, and calibration settings.

## `run_manifest.json`

Contains schema and implementation versions, resolved configuration, input paths and SHA-256 fingerprints, output filenames, runtime timestamps, Python/package versions, requested date, and optimization, signal, and benchmark-only asset counts.

No single-date file represents orders or fills.

## Rolling outputs

`scripts/run_rolling_experiment.py` writes:

- `rebalance_weights.parquet`
- `daily_performance.parquet`
- `exposure_timeseries.parquet`
- `optimization_diagnostics.parquet`
- `portfolio_metrics.json`
- `rolling_manifest.json`

The daily table contains gross return, proportional transaction cost, net return, turnover,
NAV, and drawdown for equal-weight, optimized, and simulated benchmark portfolios.
`rolling_manifest.json` records a SHA-256 entry for every resolved covariance and, when
available, every resolved exposure file. It also records per-date risk source
(`static_reused`, `dynamic_built`, or `dynamic_reused`), dynamic-cache counts, and checkpoint
counts.

When `--checkpoint-root` is supplied, every successful rebalance is persisted under
`date=YYYYMMDD/run=<signature>/`. The signature covers implementation version, core input
hashes, drifted current weights, and resolved risk files. A rerun reconstructs drift in order
and reuses only an exact, complete checkpoint. Checkpoints and dynamic risk caches are outside
the final rolling output directory and survive an interrupted run.
