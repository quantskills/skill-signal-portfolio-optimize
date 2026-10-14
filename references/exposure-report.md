# Exposure Report

`exposure_timeseries.parquet` is written by a completed rolling experiment. It records, for each rebalance date, the portfolio, benchmark and active exposure for every configured style and industry constraint.

Render the report without re-running optimization or backtesting:

```bash
python scripts/render_exposure_report.py \
  --exposure-file /path/to/rolling-output/exposure_timeseries.parquet \
  --output-dir /path/to/rolling-output/exposure_report \
  --industry-frequency monthly
```

The report contains:

- `style_active_exposure.png`: active style exposures with configured lower and upper bounds.
- `industry_active_exposure.png`: active industry-exposure heatmap. Daily rows are reduced to the last rebalance in each requested week or month when requested.
- `exposure_summary.csv`: mean, maximum absolute, 95th-percentile absolute exposure, binding count, violation count and minimum available slack.
- `report_manifest.json`: source hash, date range and plotting choices.

An active exposure is `portfolio exposure - benchmark exposure`. It is an exposure diagnostic, not a factor return attribution or a risk-contribution chart. The command only reads a completed rolling output and cannot change historical weights, optimization decisions or backtest performance.
