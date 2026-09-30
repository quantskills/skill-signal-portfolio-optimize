from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from .errors import OptimizationError
from .risk import PortfolioRisk, as_portfolio_risk


@dataclass(frozen=True)
class SelectionResult:
    weights: pd.Series
    diagnostics: dict[str, Any]


def _rank_zscore(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="average", pct=True)
    centered = ranked - float(ranked.mean())
    scale = float(centered.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise OptimizationError("selection input has no cross-sectional variation")
    return centered / scale


def _ordered_tickers(values: pd.Series, *, ascending: bool = False) -> pd.Index:
    frame = pd.DataFrame(
        {"value": values.astype(float), "_ticker": values.index.astype(str)},
        index=values.index,
    )
    frame = frame.sort_values(
        ["value", "_ticker"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    return pd.Index(frame.index, name="ticker")


def _ticker_hash(tickers: pd.Index) -> str:
    payload = "\n".join(sorted(tickers.astype(str).tolist())).encode("utf-8")
    return sha256(payload).hexdigest()


def select_risk_aware_equal_weight(
    *,
    signal_score: pd.Series,
    risk_input: pd.DataFrame | PortfolioRisk,
    anchor_weights: pd.Series,
    candidate_mask: pd.Series,
    config: dict[str, Any],
) -> SelectionResult:
    index = signal_score.index
    score = pd.to_numeric(signal_score.reindex(index), errors="coerce")
    anchor = pd.to_numeric(anchor_weights.reindex(index), errors="coerce")
    candidates = candidate_mask.reindex(index)
    if score.isna().any() or not np.isfinite(score).all():
        raise OptimizationError("risk-aware selection signal is missing or non-finite")
    if anchor.isna().any() or not np.isfinite(anchor).all():
        raise OptimizationError("risk-aware selection anchor is missing or non-finite")
    if candidates.isna().any():
        raise OptimizationError("risk-aware selection candidate mask is incomplete")
    candidates = candidates.astype(bool)

    pool_size = int(config["pool_size"])
    protected_top_n = int(config["protected_top_n"])
    target_holdings = int(config["target_holdings"])
    risk_penalty = float(config["risk_penalty"])
    candidate_scores = score.loc[candidates]
    if len(candidate_scores) < target_holdings:
        raise OptimizationError(
            "risk-aware selection has fewer candidates than target holdings"
        )

    anchor_support = pd.Index(anchor[anchor > 0.0].index, name="ticker")
    if len(anchor_support) != target_holdings:
        raise OptimizationError(
            "risk-aware selection anchor holding count must equal target_holdings"
        )
    missing_anchor = anchor_support.difference(candidate_scores.index)
    if len(missing_anchor):
        raise OptimizationError(
            "risk-aware selection anchor must be contained in the candidate pool"
        )

    ranked_candidates = _ordered_tickers(candidate_scores)
    ranked_pool = ranked_candidates[: min(pool_size, len(ranked_candidates))]
    protected = _ordered_tickers(score.loc[anchor_support])[:protected_top_n]
    replaceable = anchor_support.difference(protected, sort=False)
    alternatives = ranked_pool.difference(anchor_support, sort=False)
    boundary = replaceable.append(alternatives).drop_duplicates()
    slots = target_holdings - protected_top_n
    if len(boundary) < slots:
        raise OptimizationError("risk-aware selection boundary is too small")

    risk = as_portfolio_risk(risk_input)
    anchor_values = anchor.to_numpy(dtype=float)
    covariance_times_anchor = 0.5 * risk.gradient(anchor_values)
    marginal_addition = pd.Series(
        covariance_times_anchor + 0.5 * risk.diagonal() / target_holdings,
        index=index,
        name="marginal_addition_risk",
    )
    if risk_penalty == 0.0:
        selected_boundary = replaceable
    else:
        comparison = pd.DataFrame(
            {
                "signal": score.loc[boundary],
                "risk": marginal_addition.loc[boundary],
            }
        )
        comparison["signal_z"] = _rank_zscore(comparison["signal"])
        comparison["risk_z"] = _rank_zscore(comparison["risk"])
        comparison["adjusted_score"] = (
            comparison["signal_z"] - risk_penalty * comparison["risk_z"]
        )
        ranked_boundary = comparison.assign(
            _ticker=comparison.index.astype(str)
        ).sort_values(
            ["adjusted_score", "signal", "_ticker"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selected_boundary = pd.Index(ranked_boundary.index[:slots], name="ticker")
    selected = protected.append(selected_boundary).drop_duplicates()
    if len(selected) != target_holdings:
        raise OptimizationError("risk-aware selection did not produce exact holdings")

    weights = pd.Series(0.0, index=index, name="target_weight")
    weights.loc[selected] = 1.0 / target_holdings
    overlap = selected.intersection(anchor_support)
    added = selected.difference(anchor_support)
    removed = anchor_support.difference(selected)
    diagnostics = {
        "selection_mode": "risk_aware_boundary",
        "selection_pool_size": pool_size,
        "selection_effective_pool_size": int(len(ranked_pool)),
        "selection_protected_top_n": protected_top_n,
        "selection_target_holdings": target_holdings,
        "selection_risk_penalty": risk_penalty,
        "selection_overlap_count": int(len(overlap)),
        "selection_overlap_ratio": float(len(overlap) / target_holdings),
        "selection_replacement_count": int(len(added)),
        "selection_signal_score_change": float(score @ (weights - anchor)),
        "selection_added_tickers": added.astype(str).tolist(),
        "selection_removed_tickers": removed.astype(str).tolist(),
        "selection_ticker_sha256": _ticker_hash(selected),
    }
    return SelectionResult(weights=weights, diagnostics=diagnostics)
