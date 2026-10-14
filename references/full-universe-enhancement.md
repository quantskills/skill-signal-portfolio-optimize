# Full-Universe Industry and SIZE Enhancement

The runtime treats every full-signal ticker as a candidate when `--candidate-file` is omitted. This permits benchmark-relative, full-market enhancement without changing the solver or treating an external candidate list as implicit.

`examples/full-universe-industry-size.yaml` is a conservative starting point:

- every tradable, scored full-signal name may be selected;
- maximum stock and active weight are `0.2%`;
- one-way turnover is capped at `20%`;
- tracking error is capped at `6%`;
- industry active exposure is capped at `+/-1%`;
- only SIZE is constrained among the style factors, at `0 +/- 0.10`.

The preset is intentionally separate from the Top-N anchor presets. It changes the investable candidate set, so its result must be compared with an equally full-market signal baseline rather than a Top50/Top200 portfolio. It is an experiment preset, not a claim that these values improve every signal.

Run it by omitting `--candidate-file`:

```bash
python scripts/run_rolling_experiment.py \
  --config examples/full-universe-industry-size.yaml \
  --signal-file /path/to/full_signal.parquet \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --asset-returns-file /path/to/asset_returns.parquet \
  --tradability-file /path/to/tradability.parquet \
  --risk-model-config /path/to/risk_model.yaml \
  --risk-returns-file /path/to/risk_returns.parquet \
  --risk-market-cap-file /path/to/market_cap.parquet \
  --output-dir /path/to/output
```

Use `scripts/render_exposure_report.py` after the run to confirm that the industry and SIZE bands actually held.
