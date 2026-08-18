from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .errors import InputDataError


def _summary(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(values.median()),
    }


def _mad_winsorize(values: pd.Series, multiple: float) -> tuple[pd.Series, int]:
    if multiple <= 0:
        return values.copy(), 0
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad <= 0:
        return values.copy(), 0
    lower = median - multiple * 1.4826 * mad
    upper = median + multiple * 1.4826 * mad
    clipped = values.clip(lower=lower, upper=upper)
    return clipped, int((clipped != values).sum())


def calibrate_signal(
    prediction: pd.Series, config: dict[str, Any], requested_date: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(prediction) < 2:
        raise InputDataError("signal requires at least two assets")
    direction = 1.0 if config["higher_is_better"] else -1.0
    oriented = prediction.astype(float) * direction
    winsorized, changed = _mad_winsorize(oriented, config["winsorize_mad"])

    if config["type"] == "rank_score":
        score = winsorized.rank(method="average", pct=True)
        score = score - float(score.mean())
        standard_deviation = float(score.std(ddof=0))
        if standard_deviation <= 0:
            raise InputDataError("rank signal has no cross-sectional variation")
        signal_score = score
        if config["zscore"]:
            signal_score = score / standard_deviation
        expected_return = signal_score * config["annualized_alpha_scale"]
    else:
        signal_score = winsorized.copy()
        if config["zscore"]:
            standard_deviation = float(signal_score.std(ddof=0))
            if standard_deviation <= 0:
                raise InputDataError("expected_return signal has no cross-sectional variation")
            signal_score = (signal_score - float(signal_score.mean())) / standard_deviation
        expected_return = signal_score.copy()

    frame = pd.DataFrame(
        {
            "raw_prediction": prediction,
            "signal_score": signal_score,
            "expected_return": expected_return,
        }
    )
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise InputDataError("calibrated signal contains non-finite values")

    diagnostics: dict[str, Any] = {
        "date": requested_date,
        "type": config["type"],
        "higher_is_better": config["higher_is_better"],
        "observation_count": int(len(frame)),
        "winsorize_mad": float(config["winsorize_mad"]),
        "winsorized_observation_count": changed,
        "zscore": bool(config["zscore"]),
        "annualized_alpha_scale": float(config["annualized_alpha_scale"]),
        "raw_prediction": _summary(prediction),
        "signal_score": _summary(frame["signal_score"]),
        "expected_return": _summary(frame["expected_return"]),
    }
    return frame, diagnostics
