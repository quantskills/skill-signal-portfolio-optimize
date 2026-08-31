---
name: skill-signal-portfolio-optimize
description: Build an open Barra-style structural equity risk model and convert one frozen cross-sectional stock signal into benchmark-relative long-only target weights with auditable two-stage signal-capture, transaction-cost, exposure, turnover, tracking-error, and tradability controls. Use when Codex needs to optimize a LightGBM or single-factor stock signal, estimate or consume covariance, enforce market-cap, style, or point-in-time industry constraints, run a next-trading-day rolling portfolio study, or diagnose weights, risk, turnover, NAV, and drawdown results. Also use for Chinese requests such as 风险模型、协方差估计、信号组合优化、单因子权重优化、风险约束配权、行业风格中性、滚动优化、净值回撤、换手成本或优化前后回测对比.
quantSkills:
  schema_version: 2.1.0
  organization: quantskills
  organization_url: https://github.com/quantskills
  repository: skill-signal-portfolio-optimize
  repository_url: https://github.com/quantskills/skill-signal-portfolio-optimize
  project_type: skill
  license: GPL-3.0-only
  maintainer: X-Tech-group
  collection: portfolio-construction
  catalog:
    category: "05"
    subcategory: 05.portfolio-construction
  workflow:
    primary_stage: portfolio-construction
    workflow_stages:
      - data-ingestion
      - modeling
      - risk
      - portfolio-construction
      - backtesting
      - evaluation
      - reporting
  tags:
    - portfolio-optimization
    - signal-optimization
    - risk-model
    - benchmark-relative
    - factor-model
    - risk-control
    - backtesting
  platforms:
    - cursor
    - claude-code
    - codex
    - hermes
    - openclaw
  status: active
  requires: []
  validation_level: runnable
  maintainer_type: community
  summary_en: "Converts one stock signal into benchmark-relative target weights with auditable risk, exposure, turnover, and cost controls."
  summary_zh: "将单个股票信号转换为受基准相对风险、风格、行业、换手和成本约束的可审计组合权重。"
  interface:
    mode: natural-language
---

# Signal Portfolio Optimize

Turn one frozen stock-level signal into reviewable target weights. Treat the signal as an upstream alpha forecast and keep it separate from the risk factors used to constrain the portfolio.

## Scope

- Accept exactly one full-cross-section final signal column: `date | ticker | prediction`.
- Accept an optional `date | ticker` candidate universe; when omitted, all signal names are candidates. Optionally constrain their aggregate portfolio weight.
- Accept signals produced by LightGBM, a single factor, or another upstream model.
- Do not train models, select factors, or combine raw multi-factor columns.
- Accept either a supplied asset covariance or mutually exclusive `X`, `F`, and `D` factor-form risk inputs; otherwise estimate an open market/style/optional-industry structural model.
- Build the optimization universe from candidates, positive benchmark names, and positive current holdings; the full signal does not enlarge the risk matrix.
- Require SIZE control in schema versions 2 through 5 and allow target ranges for other style exposures.
- In schema versions 3 through 5, fail when an optimization asset lacks a prediction except for a positive, non-tradable frozen holding.
- Do not claim that the open model is MSCI Barra or a proprietary Barra product.
- Build an equal-weight signal baseline and a risk-optimized portfolio on the same date.
- Run rolling optimization with drifted current holdings and next-trading-day target execution.
- Reuse complete static risk files and dynamically rebuild exact risk coverage when carried holdings expand the optimization universe.
- Persist input-signed date checkpoints so interrupted rolling experiments can resume safely.
- v1.2 caches repeated rolling input tables in-process and partitions them by normalized date; this is a performance optimization only and does not change signal, risk, or execution semantics.
- v1.3 adds an isolated `stockdemo-compatible` order-level replay path. It does not change Barra calculations or the legacy `native` return backtest.
- v1.3.1 can explicitly feed Stockdemo-executed closing holdings into the next rolling optimization; omitting the market flag preserves theoretical-drift behavior.
- Fail closed on missing covariance coverage, invalid weights, infeasible constraints, or non-finite inputs.
- Produce target weights, `optimization_summary.json`, and machine-readable diagnostics; do not place orders.

## Workflow

1. Read [references/input-schema.md](references/input-schema.md) before preparing inputs.
2. Copy `examples/config.yaml` and set signal semantics and portfolio constraints.
3. When asset covariance is unavailable, read [references/risk-model.md](references/risk-model.md). Use `scripts/build_risk_model.py` for one date or `scripts/build_rolling_risk_models.py` for resumable date-partitioned caches, using only information available through each as-of date.
   For industry-constrained v1.1 runs, first materialize daily labels from interval history:

```bash
python scripts/prepare_industry_labels.py \
  --history-file /path/to/industry_membership_history.parquet \
  --universe-file /path/to/optimizer_universe.parquet \
  --output-file /path/to/industry_labels.parquet \
  --minimum-coverage 1.0
```

   The command fails closed by default when any requested date has incomplete coverage, overlapping
   intervals, or duplicate active classifications. For candidate-only gaps, run it on the
   candidate universe with --missing-policy exclude, --filtered-universe-file, and
   --exclusions-file; pass the filtered candidate file to the optimizer. Benchmark and existing
   holdings must still pass strict coverage. It never uses a current snapshot as historical data.
4. Run one-date optimization:

```bash
python scripts/run_single_date.py \
  --config /path/to/config.yaml \
  --signal-file /path/to/predictions.parquet \
  --candidate-file /path/to/candidates.parquet \
  --covariance-file /path/to/covariance.parquet \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --date 20230104 \
  --output-dir /path/to/output
```

5. Add `--current-weights-file` when turnover or frozen-position constraints apply.
6. Add `--sector-file`, `--exposure-file`, or `--tradability-file` only when their corresponding constraints are enabled.
7. Inspect `constraint_diagnostics.json` before consuming `target_weights.parquet`.

For a rolling experiment, read [references/backtest-contract.md](references/backtest-contract.md) and run:

```bash
python scripts/run_rolling_experiment.py \
  --config examples/config.yaml \
  --signal-file /path/to/predictions.parquet \
  --candidate-file /path/to/candidates.parquet \
  --covariance-root /path/to/risk_model \
  --exposure-root /path/to/risk_model \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --asset-returns-file /path/to/asset_returns.parquet \
  --transaction-cost-bps 7 \
  --output-dir /path/to/rolling_output
```

Add `--stockdemo-market-file /path/to/market.parquet` to this command only when actual next-day Stockdemo fills must become the next date current holdings. This mode writes `execution_feedback.parquet` and `stockdemo_compat/`; inspect `actual_cash_weight` because the optimizer normalizes the executed stock sleeve to one.

To compare against the original stockdemo execution semantics without changing the
optimizer, replay the frozen signal and/or the optimizer target weights with:

```bash
python scripts/run_stockdemo_compat_backtest.py \
  --market-file /path/to/stockdemo_market.parquet \
  --signal-file /path/to/frozen_signal.parquet \
  --target-weights-file /path/to/rebalance_weights.parquet \
  --benchmark-file /path/to/benchmark.parquet \
  --start-date 20230104 --end-date 20260121 \
  --portfolio both \
  --output-dir /path/to/stockdemo_compat_output
```

Use `--portfolio signal` for the Top200/`keep=0.8` baseline and `--portfolio target` for
an existing `risk_optimized` target-weight stream. The command uses TWAP (falling back to
open when unavailable), next-day
execution, 100/200-share lots, cash and fee accounting, ST/limit filters, and the
stockdemo-style metric calculation. It requires the raw market fields documented in
[references/stockdemo-compat.md](references/stockdemo-compat.md).

For rolling optimization whose current holdings can move outside a prebuilt static risk
universe, enable exact-universe dynamic risk and persistent checkpoints:

```bash
python scripts/run_rolling_experiment.py \
  --config examples/config.yaml \
  --signal-file /path/to/predictions.parquet \
  --candidate-file /path/to/candidates.parquet \
  --covariance-root /path/to/static_risk_model \
  --exposure-root /path/to/static_risk_model \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --asset-returns-file /path/to/asset_returns.parquet \
  --risk-model-config examples/risk-model-config.yaml \
  --risk-returns-file /path/to/risk_returns.parquet \
  --risk-market-cap-file /path/to/market_cap.parquet \
  --dynamic-risk-cache-root /path/to/risk_models_dynamic \
  --checkpoint-root /path/to/rolling_checkpoints \
  --output-dir /path/to/rolling_output
```

Pass `--risk-industry-file` when the risk-model config enables industry history. The four
required dynamic arguments must be supplied together. Static covariance and exposure files
are reused only when both cover the exact optimization universe assembled from candidates,
positive benchmark weights, and positive drifted current holdings. Missing coverage is built
under the dynamic cache without modifying the static cache.

Build the required rolling covariance cache first when it is not already available:

```bash
python scripts/build_rolling_risk_models.py \
  --config examples/risk-model-config.yaml \
  --returns-file /path/to/returns.parquet \
  --market-cap-file /path/to/market_cap.parquet \
  --universe-file /path/to/optimizer_universe.parquet \
  --start-date 20230104 \
  --end-date 20260121 \
  --rebalance-every 20 \
  --output-root /path/to/risk_model
```

The risk-model universe must include positive-weight benchmark constituents as well as signal
candidates. Rerunning the same command skips complete date caches only when recorded input hashes
still match. It fails on incomplete or stale caches instead of overwriting them.

Use `scripts/run_parameter_sweep.py` with `examples/parameter-sweep.yaml` to compare frozen
OOS variants. It writes each resolved config, one rolling output per variant, and CSV/JSON
summary tables. A rerun reuses a completed variant only when its config hash still matches.

```bash
python scripts/run_parameter_sweep.py \
  --base-config examples/alpha191-lgbm-oos-config.yaml \
  --matrix-file examples/parameter-sweep.yaml \
  --signal-file /path/to/signal_full.parquet \
  --candidate-file /path/to/signal_candidates.parquet \
  --covariance-root /path/to/risk_model \
  --exposure-root /path/to/risk_model \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --asset-returns-file /path/to/asset_returns.parquet \
  --output-root /path/to/sweep_output
```

Use `scripts/prepare_data_factor_store_inputs.py` to export portable return, market-cap,
benchmark, and tradability files from an authorized canonical data-factor store. Write those
files outside this repository.

Use `scripts/prepare_alpha191_lgbm_inputs.py` for a frozen Alpha191+LGBM experiment. It writes a full-cross-section calibration signal, deterministic tradable Top-N candidates on anchored rebalance dates, and a separate optimizer universe equal to those candidates union positive-weight benchmark constituents.

For an existing experiment that already contains a one-date candidate table and square
asset covariance, first run `scripts/prepare_existing_experiment_inputs.py`. This adapter
intersects the candidates with covariance coverage and creates the v0.1 input contract.
It deliberately uses an equal-weight benchmark over that eligible universe and records
`benchmark_is_market_index: false`; do not describe this mode as index-relative optimization.

Read [references/optimization.md](references/optimization.md) before changing objectives, scaling signals, or interpreting risk. Read [references/output-contract.md](references/output-contract.md) before consuming outputs programmatically.

## Risk Model

- Estimate the open structural factors MARKET, SIZE, BETA, MOMENTUM, RESVOL, and NLSIZE from point-in-time returns and market-cap inputs. These mean market, capitalization, market sensitivity, momentum, residual volatility, and nonlinear size.
- Add industry dummy factors only when interval-valid point-in-time classifications are available. Do not backfill a current industry snapshot into history.
- For v1.1 industry-constrained runs, use scripts/prepare_industry_labels.py to project the same interval history into the optimizer date | ticker | sector contract. Require complete coverage for the optimization universe. If only candidates are missing labels, use --missing-policy exclude to produce a date-filtered candidate file and an exclusion audit; benchmark and existing holdings remain strict.
- Residualize and standardize style exposures cross-sectionally, estimate factor returns by weighted regression, and combine factor covariance with specific variance.
- Use shrinkage, eigenvalue flooring, and positive-semidefinite repair where configured; disclose every repair and effective observation count.
- Keep factor calculation separate from portfolio constraints: a calculated factor is not automatically constrained. The example configuration requires SIZE control and leaves the other style ranges configurable.
- Accept a supplied asset covariance for simple integrations or explicit exposure, factor-covariance, and specific-variance files for factor-form optimization.

This is a transparent Barra-style implementation. It does not reproduce MSCI Barra's proprietary factor definitions, estimation universe, descriptors, or production covariance adjustments.

## Signal Rules

- Use `rank_score` for LightGBM predictions and most single-factor values that only carry ordering information.
- Use `rank_transform: uniform` for backward-compatible percentile spacing, `normal_score` for Gaussian rank spacing, or `power` with `rank_power` for an explicit tail-emphasis assumption.
- Use `expected_return` only when the input is already calibrated to the covariance horizon and units.
- Set `higher_is_better: false` for negatively oriented factors.
- Select and freeze the candidate set upstream; winsorize and standardize scores on the full OOS cross-section before optimizing that set.
- Keep OOS forecasts frozen. Do not tune signal scaling or constraints by repeatedly inspecting OOS returns.
- Require identical ticker identifiers across all inputs; never guess exchange suffix mappings.

## Optimization Rules

- Optimize active risk relative to the supplied benchmark.
- Use schema 5 `lexicographic_signal_cost` for rank signals: maximize active signal utility first, then retain `minimum_signal_capture` and minimize exact linear cost plus a strictly positive L2 stability term.
- Keep `score_max_te` available for schema 1 through 4 compatibility.
- Use `mean_variance` only when signal return units and covariance horizon are compatible.
- Require benchmark weights to sum to one within configured tolerance.
- Repair small covariance asymmetry and negative eigenvalues, and disclose the repair in diagnostics.
- Treat position, industry, market-cap, configurable style, turnover, tracking-error, and tradability limits as testable constraints.
- Treat `candidate_weight_range` as a hard aggregate-weight range over the supplied candidate universe, not as a filter on benchmark or carried names. When a non-tradable outside-candidate holding is frozen, reduce only the candidate range to its attainable maximum and disclose configured and effective bounds.
- Freeze a non-tradable asset at its current weight; fail if that current weight is unavailable. If market drift already pushed that frozen weight beyond a stock bound, preserve the executable freeze, disclose a frozen-bound exception, and keep the bound strict for every tradable asset.
- Under schema versions 3 through 5, never assign a neutral score to a tradable optimization asset with a missing prediction; only a positive non-tradable frozen holding is exempt.
- Never silently replace a failed optimization with equal weights.
- Select and record the schema 5 backend before solving: use CVXPY/Clarabel only when both are installed, otherwise use the SciPy two-stage backend. Never switch after a solve starts or describe cross-environment backend selection as bitwise deterministic.
- When CVXPY is unavailable, solve legacy `score_max_te` with auditable HiGHS ellipsoid cuts; independently recheck every hard constraint.

## Outputs

Write these stable artifacts for a successful run:

- `target_weights.parquet`
- `constraint_diagnostics.json`
- `risk_summary.json`
- `signal_diagnostics.json`
- `run_manifest.json`
- `optimization_summary.json`

Single-date output contains target weights, not executable orders. Rolling output contains a
research backtest, not broker fills. Keep production data and experiment outputs outside this
repository.

The stockdemo-compatible command additionally writes `stats.csv`, `transaction.csv`,
`holdings.csv`, and `summary.json` under the requested output directory. Treat numerical
parity as established only after comparing these files with the original stockdemo run on
a small fixture using the same input data.

## Validation Status And Limitations

- Catalog status is `active`; validation level is `runnable`. The checked-in examples, validator, and test suite exercise the documented commands. Final community listing and validation still require QuantSkills maintainer review.
- The current risk model can calculate six open market/style factors. v1.1 adds a strict interval-history adapter and industry-constrained configuration; industry controls still require complete point-in-time coverage and are not enabled in the backward-compatible default configuration.
- The reported 2023-2026 results are development-period evidence on one frozen Alpha191+LightGBM signal, not a sealed holdout and not proof of general performance.
- The rolling engine models next-trading-day target execution and configurable linear transaction costs. It does not model intraday fills, market impact, borrow, or live order routing.
- Solver results can differ slightly by backend and numerical library. Always use the recorded backend, resolved configuration, input hashes, and constraint diagnostics when reproducing a run.

See [references/validation-notes.md](references/validation-notes.md) for version-level evidence and interpretation limits.

## Project Boundary

Read [references/source-boundary.md](references/source-boundary.md) before adding data adapters or external code. Research use only. Do not present optimized or simulated results as investment advice or guaranteed performance.
