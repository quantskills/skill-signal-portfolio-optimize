# Risk Model Contract

The optional structural risk stage estimates an open, auditable multifactor model. It is
Barra-style in architecture but is not MSCI Barra data, software, or a proprietary Barra model.

## Inputs

- Returns: wide CSV or Parquet with dates on rows and ticker columns.
- Market cap: the same wide layout and ticker convention as returns.
- Optional industry history: long-form `stock_symbol | l1_code | in_date | out_date`.
- Universe: long-form `date | ticker`; a signal file can serve as the universe input.
- Config: copy `examples/risk-model-config.yaml`.

For BETA, MOMENTUM, and RESVOL, provide enough return history before the first modeled date.
With the example lookbacks and regression minimum, use at least two calendar years of warm-up
data and keep the full timing history in the manifest.

For each return on date `t`, the estimator uses market cap and industry from the preceding
aligned trading date. It ignores dates after the requested as-of date.

## Model

The open model supports:

- `MARKET` when industry history is disabled;
- `SIZE`, `BETA`, `MOMENTUM`, `RESVOL`, and `NLSIZE` market-derived styles;
- one-hot level-1 industry exposures when interval history is supplied;
- daily weighted least-squares factor returns;
- EWMA factor covariance with configurable diagonal shrinkage;
- EWMA specific variance with configurable median shrinkage;
- industry-median specific-risk imputation when an asset lacks enough residual history.

It assembles annualized asset covariance as:

```text
Sigma = X F X' + diag(specific_variance)
```

## Run

```bash
python scripts/build_risk_model.py \
  --config examples/risk-model-config.yaml \
  --returns-file /path/to/returns.parquet \
  --market-cap-file /path/to/market_cap.parquet \
  --industry-file /path/to/industry_membership_history.parquet \
  --universe-file /path/to/signal.parquet \
  --date 20230104 \
  --output-dir /path/to/risk_model/date=20230104
```

Set `industry_mode` to `required`, `optional`, or `disabled`. Do not pass a static current
industry snapshot through historical dates. Pass `asset_cov.parquet` to the optimizer and
`exposures.parquet` as its style-exposure input.

For a rolling experiment, build all selected dates with a resumable cache:

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

Each selected date is written atomically under `date=YYYYMMDD`. A rerun skips only complete
date directories whose manifest input hashes match the current inputs. An incomplete cache or
changed input fails closed so the operator can preserve, inspect, and explicitly replace it.
`rolling_risk_manifest.json` is updated after every date for `nohup` progress inspection.
For benchmark-relative optimization, the universe file must contain signal candidates union
all positive-weight benchmark constituents on every modeled date.

## Dynamic rolling coverage

The rolling optimizer can repair static-universe gaps caused by carried positions. Supply
`--risk-model-config`, `--risk-returns-file`, `--risk-market-cap-file`, and
`--dynamic-risk-cache-root` together. `--risk-industry-file` remains optional according to
`industry_mode`.

For each date, the runtime builds the same universe used by the optimizer: signal names,
positive benchmark names, and positive drifted current holdings. It first checks whether both
the resolved static covariance and exposure files cover that universe. If either file is
missing or incomplete, it estimates the risk model from the already-loaded return and
market-cap panels and writes atomically to:

```text
risk_models_dynamic/date=YYYYMMDD/universe=<fingerprint>/
```

The fingerprint includes the date, sorted universe, resolved risk config, and SHA-256 hashes
of the risk return, market-cap, and optional industry inputs. A universe, config, or source
change therefore creates a different cache directory. Complete matching dynamic caches are
reused; incomplete or stale exact-path caches fail closed. Static caches are never overwritten.

Every positive current holding must have return-panel coverage and a finite positive as-of
market cap. When tradability is supplied, every optimization name must also have an explicit
record; a non-tradable holding remains frozen at its current weight.

## Outputs

- `asset_cov.parquet`
- `factor_cov.parquet`
- `specific_var.parquet`
- `exposures.parquet`
- `factor_returns.parquet`
- `risk_model_manifest.json`

The manifest discloses model type, timing contract, lookback, factor list, whether industry
history was used, regression coverage, and specific-risk imputation. Fundamental styles are
not yet estimated by the open model.
