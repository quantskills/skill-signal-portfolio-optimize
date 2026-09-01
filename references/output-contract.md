# Output Contract

Successful runs write six files into a new or empty output directory.

## `target_weights.parquet`

Long-form rows for both portfolios:

```text
date | ticker | portfolio | target_weight | benchmark_weight |
current_weight | signal_available | is_candidate | has_signal | raw_prediction |
signal_score | expected_return | tradable
```

`portfolio` is either `equal_weight_signal` or `risk_optimized`. `signal_available` reports prediction coverage on the optimization universe; `is_candidate` distinguishes selectable names from benchmark- or current-only names; `has_signal` is retained as a compatibility alias for full-signal membership.

## `constraint_diagnostics.json`

Contains solver status, iterations, objective value, hard-constraint tolerances, turnover, candidate aggregate weight, and per-industry/style benchmark exposure, portfolio exposure, active exposure, lower/upper bounds, slack, binding status, and violations for both portfolios. `constraint_slacks` uses positive values for remaining capacity, zero for binding constraints, and negative values for violations. Raw and controllable stock-weight maxima are both recorded. Candidate diagnostics record configured and effective aggregate bounds; a frozen non-tradable holding outside the candidate set reduces only the effective attainable candidate maximum. A drifted non-tradable position outside a stock bound is held exactly and listed under `frozen_bound_exceptions`; the exception does not relax bounds for tradable assets.

Only `risk_optimized` must satisfy every configured optimizer constraint. Baseline violations are reported for comparison.

## `risk_summary.json`

Contains expected return, ex-ante absolute volatility, active volatility, and changes from the equal-weight signal portfolio.

## `signal_diagnostics.json`

Contains signal type, direction, requested date, full calibration, candidate, and optimization asset counts, optimization prediction coverage, candidate target weight, missing-prediction policy, allowed frozen-missing count, raw/calibrated distribution summaries, winsorization count, rank transform, rank power, and calibration settings.

## `optimization_summary.json`

Contains the primary and final signal utilities, signal utility floor and capture ratio, exact one-way turnover, estimated linear transaction cost, turnover saved versus the same-signal equal-weight baseline, risk form, selected backend, both stage statuses, hard-constraint slacks and bindings, available duals, and explicit reasons for unavailable dual or KKT fields.

## `run_manifest.json`

Contains schema and implementation versions, effective cost source, risk form, stable output hashes, resolved configuration, input paths and SHA-256 fingerprints, output filenames, runtime timestamps, Python/package versions, requested date, and optimization, full-signal, candidate, candidate-external, and benchmark-only asset counts.

No single-date file represents orders or fills.

## Rolling outputs

`scripts/run_rolling_experiment.py` writes:

- `rebalance_weights.parquet`
- `daily_performance.parquet`
- `exposure_timeseries.parquet`
- `optimization_diagnostics.parquet`
- `portfolio_metrics.json`
- `rolling_manifest.json`
- `optimization_summary.json`

The daily table contains gross return, proportional transaction cost, net return, turnover,
NAV, drawdown, benchmark net return, active return, active NAV, and active drawdown for
equal-weight, optimized, and simulated benchmark portfolios. `portfolio_metrics.json` adds
geometrically annualized excess return, realized tracking error, information ratio, ending
active NAV, and maximum active drawdown. `optimization_diagnostics.parquet` includes scalar
slacks for turnover, tracking error, stock limits, and candidate-weight bounds.
`optimization_summary.json` aggregates constraint binding counts and ratios, signal-capture minimum/mean/maximum, signal utility loss, turnover, estimated cost, runtime, backend, risk form, and cache reuse.

With `--stockdemo-market-file`, rolling output additionally contains `execution_feedback.parquet` and `stockdemo_compat/`. The feedback table records daily executed cash, cash weight, holdings, buy/sell amounts, costs, turnover, source target date, and the next rebalance that consumes the state. When explicit terminal write-off handling is enabled it also records `terminal_writeoff_count`, `terminal_writeoff_value`, and `terminal_writeoff_tickers`; with `carry_forward` it records `carried_forward_count`, `carried_forward_value`, and `carried_forward_tickers`. Rolling summaries aggregate the count and value fields. `optimization_diagnostics.parquet` identifies `current_state_source` and `actual_cash_weight`. The nested replay contains `stats.csv`, `transaction.csv`, `holdings.csv`, and `summary.json`.
`rolling_manifest.json` records a SHA-256 entry for every resolved covariance and, when
available, every resolved exposure file. It also records per-date risk source
(`static_reused`, `dynamic_built`, or `dynamic_reused`), dynamic-cache counts, and checkpoint
counts.

When `--checkpoint-root` is supplied, every successful rebalance is persisted under
`date=YYYYMMDD/run=<signature>/`. The signature covers implementation version, core input hashes including the optional candidate-universe hash, drifted current weights, and resolved risk files. A rerun reconstructs drift in order
and reuses only an exact, complete checkpoint. Checkpoints and dynamic risk caches are outside
the final rolling output directory and survive an interrupted run.

## Parameter sweep outputs

`scripts/run_parameter_sweep.py` writes resolved variant configs under `configs/`, complete
rolling artifacts under `runs/<variant>/`, and `sweep_summary.csv` plus
`sweep_summary.json` at the output root. Each summary row records success or the explicit
optimization error. With `--continue-on-error`, later variants continue after an infeasible
variant; no failed variant is converted to fallback weights.
