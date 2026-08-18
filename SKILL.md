---
name: skill-signal-portfolio-optimize
description: Estimate an open structural multifactor risk model or convert one final cross-sectional stock signal into benchmark-relative long-only target weights, then run an auditable next-trading-day rolling backtest. Use when Codex needs to build covariance from historical returns and market cap with required, optional, or disabled industry history; optimize a LightGBM or single-factor signal under position, SIZE, configurable style, industry, turnover, tracking-error, and tradability constraints; maximize a rank score under a risk budget; compare equal-weight, benchmark, and optimized portfolios; or diagnose risk-model, solver, exposure, turnover, return, NAV, and drawdown results. Also use for Chinese requests such as 风险模型、协方差估计、信号组合优化、单因子权重优化、风险约束配权、行业风格中性、滚动优化、净值回撤、换手成本或优化前后回测对比.
---

# Signal Portfolio Optimize

Turn one frozen stock-level signal into reviewable target weights. Treat the signal as an upstream return forecast and keep it separate from the risk factors used to constrain the portfolio.

## Scope

- Accept exactly one full-cross-section final signal column: `date | ticker | prediction`.
- Accept an optional `date | ticker` candidate universe; when omitted, all signal names are candidates. Optionally constrain their aggregate portfolio weight.
- Accept signals produced by LightGBM, a single factor, or another upstream model.
- Do not train models, select factors, or combine raw multi-factor columns.
- Accept a supplied asset covariance or estimate an open market/style/optional-industry structural model.
- Build the optimization universe from candidates, positive benchmark names, and positive current holdings; the full signal does not enlarge the risk matrix.
- Require SIZE control in schema versions 2 through 4 and allow target ranges for other style exposures.
- In schema versions 3 and 4, fail when an optimization asset lacks a prediction except for a positive, non-tradable frozen holding.
- Do not claim that the open model is MSCI Barra or a proprietary Barra product.
- Build an equal-weight signal baseline and a risk-optimized portfolio on the same date.
- Run rolling optimization with drifted current holdings and next-trading-day target execution.
- Reuse complete static risk files and dynamically rebuild exact risk coverage when carried holdings expand the optimization universe.
- Persist input-signed date checkpoints so interrupted rolling experiments can resume safely.
- Fail closed on missing covariance coverage, invalid weights, infeasible constraints, or non-finite inputs.
- Produce target weights and machine-readable diagnostics; do not place orders.

## Workflow

1. Read [references/input-schema.md](references/input-schema.md) before preparing inputs.
2. Copy `examples/config.yaml` and set signal semantics and portfolio constraints.
3. When asset covariance is unavailable, read [references/risk-model.md](references/risk-model.md). Use `scripts/build_risk_model.py` for one date or `scripts/build_rolling_risk_models.py` for resumable date-partitioned caches, using only information available through each as-of date.
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
  --transaction-cost-bps 10 \
  --output-dir /path/to/rolling_output
```

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
- Use `score_max_te` for rank signals and require an explicit tracking-error limit.
- Use `mean_variance` only when signal return units and covariance horizon are compatible.
- Require benchmark weights to sum to one within configured tolerance.
- Repair small covariance asymmetry and negative eigenvalues, and disclose the repair in diagnostics.
- Treat position, industry, market-cap, configurable style, turnover, tracking-error, and tradability limits as testable constraints.
- Treat `candidate_weight_range` as a hard aggregate-weight range over the supplied candidate universe, not as a filter on benchmark or carried names.
- Freeze a non-tradable asset at its current weight; fail if that current weight is unavailable. If market drift already pushed that frozen weight beyond a stock bound, preserve the executable freeze, disclose a frozen-bound exception, and keep the bound strict for every tradable asset.
- Under schema version 3, never assign a neutral score to a tradable optimization asset with a missing prediction; only a positive non-tradable frozen holding is exempt.
- Never silently replace a failed optimization with equal weights.
- When CVXPY is unavailable, solve `score_max_te` with auditable HiGHS ellipsoid cuts; independently recheck every hard constraint.

## Outputs

Write exactly these stable artifacts for a successful run:

- `target_weights.parquet`
- `constraint_diagnostics.json`
- `risk_summary.json`
- `signal_diagnostics.json`
- `run_manifest.json`

Single-date output contains target weights, not executable orders. Rolling output contains a
research backtest, not broker fills. Keep production data and experiment outputs outside this
repository.

## Project Boundary

Read [references/source-boundary.md](references/source-boundary.md) before adding data adapters or external code. Research use only. Do not present optimized or simulated results as investment advice or guaranteed performance.
