"""Generic execution of frozen long-only target weights.

The module deliberately owns no data paths.  Callers provide market data,
targets, optional benchmark data, and all execution parameters explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import InputDataError
from .stockdemo_compat import (
    StockDemoExecutionConfig,
    load_target_weights,
    run_stockdemo_compat,
)


@dataclass(frozen=True)
class TargetWeightExecutionConfig:
    """Market execution settings for a supplied date-ticker target-weight table."""

    transaction: float = 1.4
    initial_cash: float = 100_000_000.0
    locked_limit: float = 0.095
    missing_target_policy: str = "cash"
    missing_held_policy: str = "carry_forward"
    include_final_next_execution: bool = False

    def as_runtime_config(self) -> StockDemoExecutionConfig:
        return StockDemoExecutionConfig(
            transaction=self.transaction,
            initial_cash=self.initial_cash,
            locked_limit=self.locked_limit,
            # Target weights are already formed upstream; selection is not rerun.
            exact_window=not self.include_final_next_execution,
            missing_target_policy=self.missing_target_policy,
            missing_held_policy=self.missing_held_policy,
        )


def run_target_weight_execution(
    *,
    market: pd.DataFrame,
    target_weights: pd.DataFrame,
    output_dir: str | Path,
    config: TargetWeightExecutionConfig,
    benchmark: pd.DataFrame | None = None,
    terminal_events: dict[str, Any] | None = None,
    portfolio_name: str = "target_weight",
) -> dict[str, Any]:
    """Replay frozen weights with next-day TWAP, locks, cash, and carry rules."""

    required = {"date", "ticker", "target_weight"}
    missing = required - set(target_weights.columns)
    if missing:
        raise InputDataError(
            "target weights missing column(s): " + ", ".join(sorted(missing))
        )
    return run_stockdemo_compat(
        market=market,
        targets=target_weights.loc[:, ["date", "ticker", "target_weight"]].copy(),
        benchmark=benchmark,
        output_dir=output_dir,
        config=config.as_runtime_config(),
        portfolio_name=portfolio_name,
        terminal_events=terminal_events,
    )


def load_frozen_target_weights(path: str | Path, portfolio: str | None = None) -> pd.DataFrame:
    """Load and validate a frozen target table without assuming its producer."""

    return load_target_weights(path, portfolio)
