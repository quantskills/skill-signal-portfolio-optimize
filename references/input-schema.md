# Input Schema

## Identifier and date rules

- Use one ticker representation consistently across every file.
- Preserve exchange suffixes when present, for example `000001.SZ`.
- Numeric tickers such as `1` are accepted, but are not automatically mapped to suffixed identifiers.
- Represent dates as `YYYYMMDD`, ISO date strings, or timestamps that resolve to one trading date.
- Duplicate rows for the requested date are errors.

## Full signal

Required long-form CSV or Parquet:

```text
date | ticker | prediction
```

`prediction` must be finite. It represents one final alpha signal, not a raw multi-factor matrix. Winsorization and standardization use every row on the requested date before the optimization universe is formed.

## Candidate universe

Optional long-form CSV or Parquet:

```text
date | ticker
```

Candidates must be a non-empty subset of the full signal on each requested date. When no candidate file is supplied, every full-signal ticker is a candidate for backward compatibility. The optimization universe is the union of candidates, positive benchmark weights, and positive current holdings; full-signal names outside that union are used only for calibration. `constraints.candidate_weight_range` optionally places an absolute lower and/or upper bound on the sum of candidate target weights. If a positive non-tradable current holding lies outside the candidate set, the runtime reduces an otherwise unattainable candidate minimum/maximum to `1 - frozen_outside_candidate_weight` and reports both configured and effective ranges; no other constraint is relaxed.

## Risk input modes

Set `covariance.risk_form` to exactly one of `asset_covariance` or `factor_model`. Do not supply dense and factor-form files together. The risk form and every resolved input hash enter run manifests and rolling checkpoint signatures.

## Asset covariance

Required square CSV or Parquet with identical ticker labels on rows and columns. For CSV, the first column contains row tickers:

```text
ticker    | 000001.SZ | 000002.SZ
000001.SZ | 0.0800    | 0.0120
000002.SZ | 0.0120    | 0.1200
```

The matrix may contain extra names but must cover every optimization-universe ticker, be finite, and use annualized return covariance when `covariance.annualized: true`.

## Factor-form risk

For `factor_model`, supply three aligned files:

- exposures: `ticker` index or column plus every factor in `factor_covariance`;
- factor covariance: square factor-by-factor `factor_cov.parquet`;
- specific variance: `ticker | specific_var`, finite and non-negative.

The files may contain extra assets but must cover the complete candidate, positive benchmark, and drifted current-holding union. Rolling roots resolve `date=YYYYMMDD/exposures.parquet`, `factor_cov.parquet`, and `specific_var.parquet`. The optimizer evaluates `X F X' + D` directly without assembling an asset covariance for the solve.

## Benchmark weights

Required long-form CSV or Parquet:

```text
date | ticker | benchmark_weight
```

Weights must be non-negative and sum to one within `constraints.weight_sum_tolerance`. Positive-weight benchmark constituents automatically join the optimization universe. In schema versions 3 through 5 they must have a full-signal prediction unless they are positive, non-tradable frozen current holdings; schema versions 1 and 2 retain neutral-fill compatibility.

## Current weights

Optional unless turnover is constrained or any asset is non-tradable:

```text
date | ticker | current_weight
```

Current weights must be long-only and sum to one. Positive current holdings automatically join the optimization universe. Missing names are treated as zero after the union is built.

## Industry labels

Required when `constraints.industry_active_range` or legacy `sector_active_limit` is set:

```text
date? | ticker | sector
```

The `date` column is optional. When present, only the requested date is used. Every optimization-universe ticker must have one non-empty industry label.

For interval history in the form `stock_symbol | l1_code | in_date | out_date`, use
`scripts/prepare_industry_labels.py` to produce daily labels. The adapter treats both interval
endpoints as inclusive, rejects overlapping intervals and duplicate active labels, and records a
coverage table and input-hash manifest. It defaults to strict `--missing-policy error`. For
candidate-only gaps, `--missing-policy exclude` writes a filtered date/ticker candidate file and
an exclusion audit; pass that filtered file to the optimizer. Do not pass an unfiltered candidate
file with partial labels to a hard industry constraint run.

```bash
python scripts/prepare_industry_labels.py \
  --history-file /path/to/industry_membership_history.parquet \
  --universe-file /path/to/optimizer_universe.parquet \
  --output-file /path/to/industry_labels.parquet
```

## Style exposures

Required when `constraints.style_active_ranges` or legacy `factor_active_limit` is set:

```text
date? | ticker | SIZE | BETA | MOMENTUM | ...
```

Configured style columns must be numeric and finite for every optimization-universe ticker. Extra style columns are ignored. Columns prefixed with `INDUSTRY:` are ignored as style constraints; provide industry labels separately.

## Tradability

Optional long-form CSV or Parquet:

```text
date? | ticker | tradable
```

Accepted true values are `true`, `1`, `yes`, and `y`; false values are `false`, `0`, `no`, and `n`. A false value freezes the asset at its current weight.

## Terminal event manifest (optional)

Only for confirmed terminal holdings that disappear from the execution market, supply the manifest produced alongside derived rolling returns:

```json
{
  "terminal_events": [
    {"date": "20230710", "ticker": "300392.SZ", "return": -1.0}
  ]
}
```

For legacy StockDemo parity, `carry_forward` records missing held tickers and values them at their last valid close without treating the gap as a return or making them tradable.

The event date is the first missing execution date. Events are normalized to `YYYYMMDD` and exchange-qualified tickers; any return other than exactly `-1.0` is rejected. This manifest does not fill market rows or alter raw returns. It is accepted only with the explicit Stockdemo policy `terminal_writeoff`; under `error` ordinary missing held tickers remain fatal, while the default `carry_forward` does not infer a terminal event.

## Rolling asset returns

Required by the rolling runner:

```text
date | ticker | return
```

Use simple close-to-close returns. Rows may be sparse for assets not held, but every non-zero
holding must have a finite return on each simulated date.

When `--covariance-root` is a directory, store each matrix at one of:

```text
date=YYYYMMDD/asset_cov.parquet
date=YYYYMMDD/covariance.parquet
YYYYMMDD.parquet
```

When `--exposure-root` is supplied, store the corresponding style exposures at one of:

```text
date=YYYYMMDD/exposures.parquet
date=YYYYMMDD/style_exposures.parquet
YYYYMMDD.parquet
```

Use the same date-partitioned risk-model root for `--covariance-root` and
`--exposure-root` when it contains both `asset_cov.parquet` and `exposures.parquet`.

`--covariance-root` may be omitted only when all required dynamic-risk inputs are supplied.
Dynamic risk returns and market cap use the wide panel format documented in
[risk-model.md](risk-model.md). Large panels are loaded once per rolling process, not once per
rebalance date.

## Frozen Alpha191+LGBM adapter

`scripts/prepare_alpha191_lgbm_inputs.py` writes:

- `signal_full.parquet`: all selected-date `date | ticker | prediction` rows used for calibration;
- `signal_candidates.parquet`: deterministic tradable Top-N `date | ticker` rows;
- `signal.parquet`: legacy Top-N candidate signal retained for compatibility;
- `optimizer_universe.parquet`: Top-N candidates union positive-weight benchmark constituents;
- `benchmark_weights.parquet`: benchmark rows on selected rebalance dates;
- `rebalance_dates.parquet`: selected dates and per-date coverage counts;
- `input_manifest.json`: source hashes, ranking direction, selection interval, and counts.

Use `optimizer_universe.parquet` to build covariance. Pass `signal_full.parquet` as `--signal-file` and `signal_candidates.parquet` as `--candidate-file`. When the adapter has already selected rebalance dates, pass `--rebalance-every 1` downstream.

Rounded benchmark weights are accepted only when their raw daily sum is within the configured normalization tolerance, then divided by that daily sum. The manifest records the raw range, maximum deviation, tolerance, and normalization rule.

The adapter requires a long-form `date | ticker | tradable` eligibility file. It ranks only tradable signal rows. Benchmark constituents need a market record but may be non-tradable; constituents without records are excluded only below an explicit aggregate-weight tolerance, then the retained benchmark is renormalized and the exclusion is disclosed in the manifest.
