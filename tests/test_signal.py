from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.config import DEFAULT_CONFIG, validate_config  # noqa: E402
from portfolio_runtime.errors import ConfigError  # noqa: E402
from portfolio_runtime.signal import calibrate_signal  # noqa: E402


def _signal_config(transform: str, power: float = 1.0) -> dict[str, object]:
    config = deepcopy(DEFAULT_CONFIG["signal"])
    config.update({"rank_transform": transform, "rank_power": power})
    return config


def test_uniform_rank_transform_preserves_v080_percentile_scores() -> None:
    prediction = pd.Series([4.0, 1.0, 3.0, 2.0], index=list("ABCD"))
    config = _signal_config("uniform")
    config["zscore"] = False

    frame, diagnostics = calibrate_signal(prediction, config, "20230104")

    expected = prediction.rank(method="average", pct=True)
    expected -= expected.mean()
    assert np.allclose(frame["signal_score"], expected)
    assert diagnostics["rank_transform"] == "uniform"


def test_normal_score_transform_is_finite_symmetric_and_standardized() -> None:
    prediction = pd.Series(np.arange(1.0, 8.0), index=list("ABCDEFG"))

    frame, _ = calibrate_signal(
        prediction, _signal_config("normal_score"), "20230104"
    )

    score = frame["signal_score"].to_numpy(dtype=float)
    assert np.isfinite(score).all()
    assert np.all(np.diff(score) > 0)
    assert np.allclose(score, -score[::-1])
    assert np.isclose(np.std(score, ddof=0), 1.0)


def test_power_transform_changes_rank_spacing_without_changing_order() -> None:
    prediction = pd.Series(np.arange(1.0, 8.0), index=list("ABCDEFG"))
    uniform, _ = calibrate_signal(prediction, _signal_config("uniform"), "20230104")
    powered, diagnostics = calibrate_signal(
        prediction, _signal_config("power", 2.0), "20230104"
    )

    assert powered["signal_score"].rank().equals(uniform["signal_score"].rank())
    assert not np.allclose(powered["signal_score"], uniform["signal_score"])
    assert diagnostics["rank_power"] == 2.0


def test_expected_return_rejects_rank_transform_settings() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["signal"].update(
        {
            "type": "expected_return",
            "zscore": False,
            "rank_transform": "normal_score",
        }
    )
    with pytest.raises(ConfigError, match="apply only to rank_score"):
        validate_config(config)
