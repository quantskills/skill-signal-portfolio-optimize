# Recommended Presets

The package keeps the historical `DEFAULT_CONFIG` fallback for backward compatibility. For reproducible research runs, pass one of the versioned YAML presets below explicitly.

| Preset | Intended use | Key behavior |
| --- | --- | --- |
| `defaults/default_lexicographic_v1_8.yaml` | General recommended optimizer | Clarabel lexicographic signal/cost objective, factor risk, TE 6%, turnover 12%, 4% stock and active-weight caps, industry +/-2%, and SIZE/BETA exposure controls. |
| `defaults/259_flexible_benchmark_enhancement_keep07.yaml` | Reproduce the 259-factor flexible benchmark-enhancement study | 259 LGBM signal, factor risk, TE 6%, turnover 12%, industry +/-2%, and the flexible candidate + benchmark + current-holding optimization domain. Use `keep=0.7` in the StockDemo execution arguments. |
| `defaults/259_strict_topn_p005.yaml` | Reproduce the 259 strict TopN boundary-selection study | Schema 10 risk-aware selection, exact equal-weight TopN, boundary risk penalty `0.005`, and a wider candidate pool. This is research-specific, not a universal default. |

## Reproducible Execution

The YAML file controls optimizer behavior. Execution behavior must also be recorded explicitly:

```bash
--initial-position-policy benchmark \
--rebalance-every 1 \
--risk-refresh-frequency monthly \
--transaction-cost-bps 7 \
--stockdemo-transaction 1.4 \
--stockdemo-keep 0.7 \
--stockdemo-turnover-mode flex \
--stockdemo-missing-target-policy cash \
--stockdemo-missing-held-policy carry_forward
```

For a five-day study, change only `--rebalance-every 5` and keep the other execution settings unchanged. Every formal run should retain the generated `rolling_manifest.json`, which records input paths, configuration fingerprints, dates, risk refresh frequency, and execution settings.

The general preset is the recommended starting point. The two 259 presets are explicit research reproductions and should not replace the general default for unrelated signals.
