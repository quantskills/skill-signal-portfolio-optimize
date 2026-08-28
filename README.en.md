# Signal Portfolio Optimize

[中文](README.md) | [Skill workflow](SKILL.md)

Convert one frozen cross-sectional stock signal into benchmark-relative long-only target weights with open Barra-style risk modeling, portfolio constraints, rolling backtests, and auditable diagnostics.

| Item | Current state |
| --- | --- |
| Catalog status | `active` |
| Validation level | `runnable` |
| Implementation version | `1.3.0` |
| Python | CI uses 3.12 |
| License | GPL-3.0-only |

Here, `active` is the project lifecycle status and `runnable` is the repository-declared, locally checked L2 target. Final community listing and validation still require QuantSkills maintainer review.

## What It Solves

An upstream model ranks stocks but does not determine how much capital to assign to each name. This Skill treats a LightGBM prediction or single-factor score as alpha, separates it from the risk model, and produces constrained target weights:

```text
frozen signal -> full-cross-section calibration -> candidates -> risk model
              -> two-stage optimization -> target weights
              -> next-day rolling backtest -> risk and constraint diagnostics
```

It does not train prediction models, select alpha factors, place orders, or reproduce MSCI Barra's proprietary model.

## Capabilities

- Consume one final `date | ticker | prediction` signal from LightGBM, a single factor, or another upstream model.
- Consume an asset covariance matrix or factor-form exposure `X`, factor covariance `F`, and specific risk `D`.
- Estimate an open structural risk model from historical returns and market capitalization.
- Constrain stock weights, active weights, SIZE and other style exposures, industry exposures, turnover, tracking error, and frozen non-tradable holdings.
- Use the v1.0.0 lexicographic objective: maximize active signal utility, preserve a minimum signal-capture ratio, then reduce linear transaction cost and weight instability.
- v1.1.0 adds a strict point-in-time interval adapter that checks daily industry coverage before hard industry constraints are enabled.
- Run next-trading-day rolling backtests with dynamic risk caches, resumable checkpoints, and parameter sweeps.
- v1.2.0 caches repeated rolling input tables in-process and slices them by date without changing optimization semantics.
- v1.3.0 adds an isolated `stockdemo-compatible` execution engine for TWAP, next-day execution, `keep=0.8`, lots, cash, ST/limit filters, and stockdemo metric semantics; the legacy `native` backtest remains unchanged.
- Emit target weights, risk summaries, constraint diagnostics, signal diagnostics, and hash-backed run manifests.

## Risk Factors

| Code | Meaning | Current support |
| --- | --- | --- |
| MARKET | Broad market | Calculated |
| SIZE | Market capitalization | Calculated; constrained by the example config |
| BETA | Market sensitivity | Calculated; optional constraint |
| MOMENTUM | Price momentum | Calculated; optional constraint |
| RESVOL | Residual volatility | Calculated; optional constraint |
| NLSIZE | Nonlinear size | Calculated; optional constraint |
| INDUSTRY:* | Industry dummies | Calculated only with interval-valid point-in-time history |

Calculation does not automatically enable a constraint. Configure target exposures and tolerances explicitly. v1.1.0 provides [industry portfolio](examples/v1.1-industry-portfolio-config.yaml) and [industry risk-model](examples/v1.1-industry-risk-model-config.yaml) examples. Industry labels are strict by default; for candidate-only gaps, use --missing-policy exclude to generate a filtered candidate file and exclusion audit. Benchmark and existing holdings still require labels; never backfill current classifications into historical dates.

## Installation

```bash
git clone git@github.com:quantskills/skill-signal-portfolio-optimize.git
cd skill-signal-portfolio-optimize
python -m pip install -r requirements.txt
```

## Minimum Inputs

A one-date asset-covariance run requires:

- signal: `date | ticker | prediction`
- benchmark: `date | ticker | benchmark_weight`
- covariance: a square matrix indexed and columned by ticker
- configuration: start from [examples/config.yaml](examples/config.yaml)

A `date | ticker` candidate file is optional; all signal names become candidates when it is omitted. Turnover, industry, style, and tradability constraints require current-weight, industry, exposure, or tradability files respectively. See [references/input-schema.md](references/input-schema.md) for complete schemas and validation rules.

## Single-Date Quick Start

```bash
python scripts/run_single_date.py \
  --config examples/config.yaml \
  --signal-file /path/to/predictions.parquet \
  --candidate-file /path/to/candidates.parquet \
  --covariance-file /path/to/covariance.parquet \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --date 20230104 \
  --output-dir outputs/date=20230104
```

Successful stdout resembles:

```json
{
  "asset_count": 197,
  "date": "20230104",
  "optimized_active_volatility": 0.05552,
  "optimized_expected_return": 0.07836,
  "solver_iterations": 17,
  "status": "success"
}
```

Stable outputs include:

- `target_weights.parquet`
- `constraint_diagnostics.json`
- `risk_summary.json`
- `signal_diagnostics.json`
- `run_manifest.json`
- `optimization_summary.json`

These are research target weights, not executable orders. Read [references/output-contract.md](references/output-contract.md) before consuming them programmatically.

## Rolling Experiment

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
  --output-dir outputs/rolling
```

See [references/risk-model.md](references/risk-model.md) and [references/backtest-contract.md](references/backtest-contract.md) for factor-form risk, dynamic rebuilding, cache signatures, and resumable checkpoints.

## Stockdemo-compatible backtest

The optimizer emits target weights, while the original stockdemo backtest generates orders from holdings, cash, lots, and tradability. v1.3.0 keeps that execution layer separate from Barra logic:

```bash
python scripts/run_stockdemo_compat_backtest.py \
  --market-file /path/to/stockdemo_market.parquet \
  --signal-file /path/to/frozen_signal.parquet \
  --target-weights-file /path/to/rebalance_weights.parquet \
  --benchmark-file /path/to/benchmark.parquet \
  --start-date 20230104 --end-date 20260121 \
  --portfolio both \
  --output-dir outputs/stockdemo_compat
```

Use `--portfolio signal` for the Top200/`keep` baseline, `--portfolio target` for Barra target weights, or `both` for separate `signal_baseline/` and `risk_optimized/` outputs. The market table needs `date,ticker,open,close,pre_close,is_open,is_st`; a missing `twap` falls back to open as in stockdemo, and `adj_factor` is optional. See [references/stockdemo-compat.md](references/stockdemo-compat.md) for the complete contract.

## Validation

```bash
python scripts/validate_skill.py .
python -m compileall -q scripts
python -m pytest -q
```

The common comparison below uses the same previously inspected Alpha191+LightGBM development interval. The baseline is the frozen Top 200 equal-weight signal portfolio, not the market benchmark:

| Portfolio | Sharpe ratio |
| --- | ---: |
| Top 200 equal-weight baseline | 1.0929 |
| v0.8.0 full-market signal calibration | 1.1471 |
| v0.9.0 best normal-score transform | 1.2054 |
| v1.0.0 best two-stage signal-capture variant | 1.2147 |

See [references/validation-notes.md](references/validation-notes.md) for the protocol, constraint evidence, and interpretation limits. The interval has been inspected repeatedly, so this is development evidence rather than sealed holdout validation and does not establish performance across other signals or regimes.

## Boundaries

- Quantitative research and portfolio-construction experiments only.
- No real-time market data, broker integration, orders, market-impact simulation, or production risk controls.
- Do not present optimized or simulated results as investment advice or guaranteed performance.
- Keep external data and experiment outputs outside the repository; see [references/source-boundary.md](references/source-boundary.md).

## License

[GPL-3.0-only](LICENSE)
