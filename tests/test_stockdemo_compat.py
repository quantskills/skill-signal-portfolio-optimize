from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from portfolio_runtime.config import DEFAULT_CONFIG
from portfolio_runtime.errors import InputDataError
from portfolio_runtime.stockdemo_compat import (
    _equal_target,
    _place_orders,
    StockDemoExecutionConfig,
    StockDemoPortfolioState,
    advance_stockdemo_state,
    load_stockdemo_market,
    load_terminal_events,
    run_stockdemo_compat,
)


def test_defaults_match_ba875_single_factor_execution() -> None:
    config = StockDemoExecutionConfig()

    assert config.longx == 200
    assert config.stock_pool == "whole"
    assert config.trade_price_type == "twap"
    assert config.buy_sell_shift == 1
    assert config.transaction == pytest.approx(1.4)
    assert config.keep == pytest.approx(0.7)
    assert config.turnover_mode == "flex"
    assert config.missing_held_policy == "carry_forward"
    assert config.initial_cash == pytest.approx(100_000_000.0)
    assert DEFAULT_CONFIG["baseline"]["top_n"] == 200
    assert DEFAULT_CONFIG["cost_model"]["linear_cost_bps"] == pytest.approx(7.0)


def _market() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in (20230102, 20230103, 20230104, 20230105):
        for ticker in ("000001.SZ", "000002.SZ", "688001.SH"):
            price = {"000001.SZ": 100.0, "000002.SZ": 110.0, "688001.SH": 120.0}[ticker]
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "close": price * (1.0 + (0.01 if date == 20230103 else 0.0)),
                    "pre_close": price,
                    "twap": price,
                    "is_open": True,
                    "is_st": False,
                    "adj_factor": 1.0,
                }
            )
    for row in rows:
        if row["date"] == 20230103 and row["ticker"] == "000001.SZ":
            row["is_st"] = True
        if row["date"] == 20230104 and row["ticker"] == "000002.SZ":
            row["open"] = 120.0
            row["pre_close"] = 100.0
            row["twap"] = 120.0
    return pd.DataFrame(rows)


def test_stockdemo_market_normalizes_aliases_and_limit_state(tmp_path: Path) -> None:
    source = tmp_path / "market.parquet"
    frame = _market().rename(
        columns={
            "pre_close": "preClosePrice",
            "is_open": "isOpen",
            "is_st": "isST",
            "adj_factor": "accumAdjFactor",
        }
    )
    frame.to_parquet(source, index=False)
    loaded = load_stockdemo_market(source, start_date=20230102, end_date=20230105)
    assert loaded["date"].tolist()[:3] == ["20230102"] * 3
    locked = loaded.loc[(loaded["date"] == "20230104") & (loaded["ticker"] == "000002.SZ")].iloc[0]
    assert bool(locked["zt"])
    assert not bool(locked["can_buy"])


def test_stockdemo_market_accepts_unified_long_store_schema(tmp_path: Path) -> None:
    source = tmp_path / "market.parquet"
    frame = _market().rename(
        columns={"twap": "vwap", "adj_factor": "adjfactor"}
    )
    frame["symbol"] = frame["ticker"]
    frame["ticker"] = frame["symbol"].str[:6].astype(int)
    frame["is_open"] = frame["is_open"].astype(float)
    frame["is_st"] = frame["is_st"].astype(float)
    frame.to_parquet(source, index=False)

    loaded = load_stockdemo_market(source, start_date=20230102, end_date=20230105)

    assert set(loaded["ticker"]) == {"000001.SZ", "000002.SZ", "688001.SH"}
    row = loaded.loc[
        (loaded["date"] == "20230102") & (loaded["ticker"] == "000001.SZ")
    ].iloc[0]
    assert row["trade_price"] == row["twap"] / row["adj_factor"]


def test_signal_execution_uses_next_date_and_st_filter(tmp_path: Path) -> None:
    signal = pd.DataFrame(
        {
            "date": [20230102] * 3 + [20230103] * 3 + [20230104] * 3,
            "ticker": ["000001.SZ", "000002.SZ", "688001.SH"] * 3,
            "signal": [3.0, 2.0, 1.0] * 3,
        }
    )
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230105
    )
    summary = run_stockdemo_compat(
        market=market,
        signal=signal,
        output_dir=tmp_path / "output",
        config=StockDemoExecutionConfig(longx=2, initial_cash=1_000_000.0),
    )
    stats = pd.read_csv(tmp_path / "output" / "stats.csv")
    assert stats["date"].tolist() == [20230103, 20230104]
    holdings = pd.read_csv(tmp_path / "output" / "holdings.csv")
    first_day = holdings.loc[holdings["date"] == 20230103, "ticker"].tolist()
    assert "000001.SZ" not in first_day
    assert summary["engine"] == "stockdemo_compat"
    assert summary["metrics"]["metric_convention"] == "stockdemo_legacy"
    payload = json.loads((tmp_path / "output" / "summary.json").read_text())
    assert np.isfinite(payload["metrics"]["annualized_return"])


def test_target_execution_reuses_stockdemo_order_accounting(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230105
    )
    targets = pd.DataFrame(
        {
            "date": [20230102] * 3 + [20230103] * 3 + [20230104] * 3,
            "ticker": ["000001.SZ", "000002.SZ", "688001.SH"] * 3,
            "target_weight": [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5],
        }
    )
    summary = run_stockdemo_compat(
        market=market,
        targets=targets,
        output_dir=tmp_path / "target_output",
        config=StockDemoExecutionConfig(longx=2, initial_cash=1_000_000.0),
        portfolio_name="risk_optimized",
    )
    stats = pd.read_csv(tmp_path / "target_output" / "stats.csv")
    assert stats["date"].tolist() == [20230103, 20230104]
    transactions = pd.read_csv(tmp_path / "target_output" / "transaction.csv")
    assert set(transactions["B/S"]) == {"buy", "sell"}
    assert summary["portfolio"] == "risk_optimized"


def test_keep_budget_does_not_oversell_first_holding(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230102
    ).set_index("ticker")
    holdings = {"000001.SZ": 100.0, "000002.SZ": 100.0, "688001.SH": 200.0}
    signal = pd.Series(
        {"000001.SZ": 3.0, "000002.SZ": 2.0, "688001.SH": 1.0}
    )

    target = _equal_target(
        signal, market, holdings, longx=3, keep=0.8, first_day=False
    )

    assert set(target.index) == set(holdings)
    assert np.isclose(target.sum(), 1.0)


def test_order_accounting_preserves_stockdemo_negative_cash_behavior(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    day = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230102
    ).set_index("ticker")
    day["trade_price"] = day["ideal_trade_price"] * 2.0
    holdings: dict[str, float] = {}
    holding_adj: dict[str, float] = {}

    cash, orders, _, _ = _place_orders(
        day=day,
        target=pd.Series({"000001.SZ": 1.0}),
        holdings=holdings,
        holding_adj=holding_adj,
        cash=10_000.0,
        fee_rate=0.0007,
        initial_value=10_000.0,
    )

    assert cash < 0.0
    assert orders[0]["volume"] == 100.0


def test_execution_state_returns_actual_normalized_holdings(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    day = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230102
    ).set_index("ticker")
    state = StockDemoPortfolioState.initial(1_000_000.0)

    snapshot, transactions, holdings, weights = advance_stockdemo_state(
        state=state,
        date="20230102",
        day=day,
        fee_rate=0.0007,
        target=pd.Series({"000001.SZ": 0.50, "000002.SZ": 0.50}),
    )

    assert snapshot["holdings"] == 2
    assert len(transactions) == 2
    assert len(holdings) == 2
    assert weights.sum() == pytest.approx(1.0)
    assert set(weights.index) == {"000001.SZ", "000002.SZ"}
    assert snapshot["cash_weight"] != 0.0


def test_positive_target_missing_market_fails_loudly(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    day = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230102
    ).set_index("ticker")

    with pytest.raises(InputDataError, match="missing positive target"):
        _place_orders(
            day=day,
            target=pd.Series({"999999.SZ": 1.0}),
            holdings={},
            holding_adj={},
            cash=10_000.0,
            fee_rate=0.0007,
            initial_value=10_000.0,
        )


def test_exact_window_values_non_rebalance_market_dates(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230105
    )
    signal = pd.DataFrame(
        {
            "date": [20230102] * 3 + [20230104] * 3,
            "ticker": ["000001.SZ", "000002.SZ", "688001.SH"] * 2,
            "signal": [3.0, 2.0, 1.0] * 2,
        }
    )

    run_stockdemo_compat(
        market=market,
        signal=signal,
        output_dir=tmp_path / "sparse_output",
        config=StockDemoExecutionConfig(longx=2, initial_cash=1_000_000.0),
    )

    stats = pd.read_csv(tmp_path / "sparse_output" / "stats.csv")
    transactions = pd.read_csv(tmp_path / "sparse_output" / "transaction.csv")
    assert stats["date"].tolist() == [20230103, 20230104]
    assert transactions["date"].unique().tolist() == [20230103]


def test_missing_positive_target_can_remain_in_cash(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(market_path, start_date=20230103, end_date=20230103)
    day = market.set_index("ticker")
    state = StockDemoPortfolioState.initial(1_000_000.0)
    target = pd.Series({"000001.SZ": 0.5, "000099.SZ": 0.5})

    snapshot, transactions, _, _ = advance_stockdemo_state(
        state=state,
        date="20230103",
        day=day,
        fee_rate=0.0007,
        target=target,
        missing_target_policy="cash",
    )

    assert snapshot["missing_target_count"] == 1
    assert snapshot["missing_target_weight"] == pytest.approx(0.5)
    assert snapshot["missing_target_tickers"] == "000099.SZ"
    assert all(row["ticker"] != "000099.SZ" for row in transactions)
    assert snapshot["cash"] == pytest.approx(499_650.0)



def test_carry_forward_preserves_missing_holding_and_valuation(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230104
    )
    state = StockDemoPortfolioState.initial(1_000_000.0)
    first_day = market.loc[market["date"].eq("20230102")].set_index("ticker")
    first_snapshot, _, _, _ = advance_stockdemo_state(
        state=state,
        date="20230102",
        day=first_day,
        fee_rate=0.0007,
        target=pd.Series({"000001.SZ": 1.0}),
    )
    missing_day = market.loc[
        market["date"].eq("20230103") & market["ticker"].ne("000001.SZ")
    ].set_index("ticker")

    snapshot, transactions, holding_rows, weights = advance_stockdemo_state(
        state=state,
        date="20230103",
        day=missing_day,
        fee_rate=0.0007,
        missing_held_policy="carry_forward",
    )

    assert snapshot["carried_forward_count"] == 1
    assert snapshot["carried_forward_tickers"] == "000001.SZ"
    assert snapshot["carried_forward_value"] > 0
    assert snapshot["unrealized_pnl"] == pytest.approx(first_snapshot["unrealized_pnl"])
    assert transactions == []
    assert state.holdings["000001.SZ"] > 0
    assert holding_rows[0]["ticker"] == "000001.SZ"
    assert weights.index.tolist() == ["000001.SZ"]


def test_terminal_manifest_loads_and_normalizes(tmp_path: Path) -> None:
    manifest = tmp_path / "terminal_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "terminal_events": [
                    {"date": "2023-01-04", "ticker": "000001.SZ", "return": -1.0},
                    {"date": 20230104, "ticker": "000002.SZ", "return": -1},
                ]
            }
        ),
        encoding="utf-8",
    )
    events = load_terminal_events(manifest)
    assert events == {"20230104": frozenset({"000001.SZ", "000002.SZ"})}


def test_terminal_writeoff_removes_only_declared_missing_holding(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230104
    )
    state = StockDemoPortfolioState.initial(1_000_000.0)
    advance_stockdemo_state(
        state=state,
        date="20230102",
        day=market.loc[market["date"].eq("20230102")].set_index("ticker"),
        fee_rate=0.0007,
        target=pd.Series({"000001.SZ": 0.5, "000002.SZ": 0.5}),
    )
    advance_stockdemo_state(
        state=state,
        date="20230103",
        day=market.loc[market["date"].eq("20230103")].set_index("ticker"),
        fee_rate=0.0007,
    )
    day = market.loc[
        market["date"].eq("20230104") & market["ticker"].ne("000001.SZ")
    ].set_index("ticker")
    snapshot, _, holding_rows, weights = advance_stockdemo_state(
        state=state,
        date="20230104",
        day=day,
        fee_rate=0.0007,
        missing_held_policy="terminal_writeoff",
        terminal_events={"20230104": {"000001.SZ"}},
    )
    assert snapshot["terminal_writeoff_count"] == 1
    assert snapshot["terminal_writeoff_tickers"] == "000001.SZ"
    assert snapshot["terminal_writeoff_value"] > 0
    assert "000001.SZ" not in state.holdings
    assert "000001.SZ" not in {row["ticker"] for row in holding_rows}
    assert set(weights.index) == {"000002.SZ"}


def test_undeclared_missing_holding_still_fails(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    market = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230104
    )
    state = StockDemoPortfolioState.initial(1_000_000.0)
    advance_stockdemo_state(
        state=state,
        date="20230102",
        day=market.loc[market["date"].eq("20230102")].set_index("ticker"),
        fee_rate=0.0007,
        target=pd.Series({"000001.SZ": 0.5, "000002.SZ": 0.5}),
    )
    with pytest.raises(InputDataError, match="missing held ticker"):
        advance_stockdemo_state(
            state=state,
            date="20230103",
            day=market.loc[
                market["date"].eq("20230103") & market["ticker"].ne("000001.SZ")
            ].set_index("ticker"),
            missing_held_policy="error",
            fee_rate=0.0007,
        )




def test_stockdemo_replay_accepts_manifest_terminal_event(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    source = _market().loc[
        ~((_market()["date"] == 20230104) & (_market()["ticker"] == "000001.SZ"))
    ].copy()
    source.to_parquet(market_path, index=False)
    market = load_stockdemo_market(market_path, start_date=20230102, end_date=20230104)
    targets = pd.DataFrame(
        {
            "date": [20230102, 20230103],
            "ticker": ["000001.SZ", "000001.SZ"],
            "target_weight": [1.0, 1.0],
        }
    )
    summary = run_stockdemo_compat(
        market=market,
        targets=targets,
        output_dir=tmp_path / "output",
        config=StockDemoExecutionConfig(
            longx=1,
            initial_cash=1_000_000.0,
            exact_window=False,
            missing_target_policy="cash",
            missing_held_policy="terminal_writeoff",
        ),
        terminal_events={"20230104": {"000001.SZ"}},
    )
    stats = pd.read_csv(tmp_path / "output" / "stats.csv")
    terminal = stats.loc[stats["date"].eq(20230104)].iloc[0]
    assert terminal["terminal_writeoff_count"] == 1
    assert terminal["terminal_writeoff_tickers"] == "000001.SZ"
    assert summary["metrics"]["observations"] == 2



def test_flex_turnover_mode_matches_factor_backtest_rule() -> None:
    tickers = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    day = pd.DataFrame(
        {
            "ideal_trade_price": [0.05, 0.03, 0.02, 0.01],
            "is_open": [True] * 4,
            "zt": [False] * 4,
            "dt": [False] * 4,
            "is_st": [False] * 4,
        },
        index=tickers,
    )
    holdings = {ticker: 100.0 for ticker in tickers[:3]}
    signal = pd.Series({ticker: score for ticker, score in zip(tickers, [4.0, 3.0, 2.0, 1.0])})

    flex = _equal_target(
        signal, day, holdings, longx=3, keep=0.7, first_day=False, turnover_mode="flex"
    )
    normal = _equal_target(
        signal, day, holdings, longx=3, keep=0.7, first_day=False, turnover_mode="normal"
    )

    assert set(flex.index) == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert set(normal.index) == {"000001.SZ", "000002.SZ", "000004.SZ"}


def test_missing_signal_is_sorted_last_for_keep_budget() -> None:
    tickers = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    day = pd.DataFrame(
        {
            "ideal_trade_price": [1.0] * 4,
            "is_open": [True] * 4,
            "zt": [False] * 4,
            "dt": [False] * 4,
            "is_st": [False] * 4,
        },
        index=tickers,
    )
    holdings = {ticker: 100.0 for ticker in tickers[:3]}
    # 000001 is absent from the executable signal universe, as it would be
    # when the stock is locked down on the execution date.
    signal = pd.Series({"000002.SZ": 1.0, "000003.SZ": 2.0, "000004.SZ": 3.0})

    target = _equal_target(
        signal, day, holdings, longx=3, keep=0.2, first_day=False, turnover_mode="flex"
    )

    assert "000001.SZ" in target.index
    assert "000004.SZ" in target.index


def test_legacy_wide_twap_overlay_overrides_market_alias(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    _market().to_parquet(market_path, index=False)
    twap_path = tmp_path / "trade_price.parquet"
    pd.DataFrame(
        {
            "000001": [101.0],
            "000002": [111.0],
            "688001": [121.0],
        },
        index=pd.Index([20230102], name=None),
    ).to_parquet(twap_path)

    loaded = load_stockdemo_market(
        market_path, start_date=20230102, end_date=20230102, twap_file=twap_path
    )
    row = loaded.loc[loaded["ticker"].eq("000001.SZ")].iloc[0]
    assert row["twap"] == pytest.approx(101.0)
    assert row["trade_price"] == pytest.approx(101.0)
