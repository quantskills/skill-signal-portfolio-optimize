# Validation Notes

## Status

- Catalog status: `active`
- Validation level: `runnable`
- Current implementation version: `1.1.0`; the comparison table below records the historical v1.0.0 experiment.
- Validation claim: commands, examples, declarations, and automated tests are runnable
- Excluded claim: this repository is not yet `verified` against a sealed holdout or independent reproduction package
- Governance note: final community listing and validation require QuantSkills maintainer review

## Common Evaluation Protocol

The version comparison uses one frozen Alpha191+LightGBM signal and a common portfolio backtest:

| Item | Value |
| --- | --- |
| Signal dates | 2023-01-04 through 2025-12-24 |
| Return dates | 2023-01-05 through 2026-01-23 |
| Rebalances | 37 |
| Return observations | 740 trading days |
| Candidate set | Deterministic tradable Top 200 per signal date |
| Optimization universe | Candidates, positive benchmark names, and positive current holdings |
| Execution timing | Target weights earn returns from the next trading day |
| Transaction cost | 7 bps one way |
| Initial holdings | Benchmark weights |
| Common baseline | Frozen Top 200 equal-weight signal portfolio |
| Simulated benchmark | Supplied historical rebalance weights |

The signal, candidate dates, benchmark, return path, and cost convention were held fixed for the reported version comparisons. The baseline Sharpe ratio is `1.092872044955838`.

## Version Evidence

| Version | Main change | Sharpe ratio | Difference from baseline |
| --- | --- | ---: | ---: |
| v0.8.0 | Calibrate ranks on the full market while freezing the Top 200 candidate set | 1.147055 | +0.054183 |
| v0.9.0 | Add configurable uniform, normal-score, and power rank transforms plus candidate-weight controls | 1.205389 | +0.112517 |
| v1.0.0 | Add lexicographic signal capture, exact linear transaction cost, and L2 stability | 1.214701 | +0.121829 |

The best v0.9.0 treatment was `normal_score_no_floor`. The best v1.0.0 treatment used `minimum_signal_capture: 0.9975`; its annualized return was 28.02%, annualized volatility 22.42%, absolute maximum drawdown -25.80%, active maximum drawdown -8.62%, total one-way turnover 11.08, and cumulative modeled cost 0.78%.

## Constraint Evidence

All four formal v1.0.0 variants completed all 37 rebalance dates without fallback. Their independently checked hard constraints passed. For the selected `capture_09975` treatment:

- minimum realized signal capture: 0.997499
- turnover-cap binding dates: 5
- tracking-error binding dates: 22
- SIZE lower-bound binding dates: 24
- industry constraints: disabled

The binding counts show that the optimizer is frequently shaped by risk and exposure limits. They do not prove that the selected limits are economically optimal.

## Interpretation Limits

This interval has been inspected repeatedly during v0.8.0, v0.9.0, and v1.0.0 development. Treat every metric above as development evidence, not as a sealed out-of-sample result.

The evidence covers one signal family, one market interval, one rebalance schedule, and one cost setting. It does not establish robustness across signals, candidate counts, market regimes, liquidity assumptions, or future periods. A high Sharpe ratio on this interval is not a performance guarantee.

Industry constraints were disabled because verified interval-valid point-in-time industry history was unavailable. Current industry labels were not used as artificial history. Fundamental value, quality, growth, and leverage styles are also outside the current six-factor model.

The backtest models close-to-close returns, next-trading-day target activation, and linear turnover cost. It does not model intraday execution, nonlinear market impact, borrow availability, or broker fills. Maximum drawdown in the version table is absolute portfolio drawdown; active drawdown is reported separately.

## Reproduction Requirements

A defensible reproduction must preserve:

- the frozen full-cross-section signal and candidate file
- benchmark, return, market-cap, and optional point-in-time industry inputs
- resolved YAML configuration
- solver backend and dependency versions
- input hashes and run manifest
- risk-model cache signatures
- next-day timing and transaction-cost convention
- constraint diagnostics for every rebalance date

The public examples demonstrate the interface but do not bundle proprietary or local research data. Keep data and formal experiment outputs outside the source repository.
