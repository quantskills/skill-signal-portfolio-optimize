from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_alpha191_lgbm_inputs import OUTPUT_FILES, prepare_inputs  # noqa: E402


def test_prepare_alpha191_lgbm_inputs_selects_dates_and_unions_benchmark(
    tmp_path: Path,
) -> None:
    signal_file = tmp_path / "merged_signal.csv"
    benchmark_file = tmp_path / "benchmark.parquet"
    eligibility_file = tmp_path / "tradability.parquet"
    tickers = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    pd.DataFrame(
        [
            {"date": date, "symbol": ticker, "signal": score}
            for date in [20230102, 20230103, 20230104]
            for ticker, score in zip(tickers, [0.1, 0.4, 0.3, 0.2], strict=True)
        ]
    ).to_csv(signal_file, index=False)
    pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "benchmark_weight": weight,
            }
            for date in [20230102, 20230103, 20230104]
            for ticker, weight in [("000001.SZ", 0.60003), ("000004.SZ", 0.4)]
        ]
    ).to_parquet(benchmark_file, index=False)
    pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "tradable": ticker != "000004.SZ"}
            for date in [20230102, 20230103, 20230104]
            for ticker in tickers
        ]
    ).to_parquet(eligibility_file, index=False)
    output = tmp_path / "output"

    result = prepare_inputs(
        signal_file=signal_file,
        benchmark_file=benchmark_file,
        eligibility_file=eligibility_file,
        start_date=20230102,
        end_date=20230104,
        rebalance_every=2,
        candidate_count=2,
        output_dir=output,
    )

    assert result["status"] == "success"
    assert result["rebalance_count"] == 2
    assert {path.name for path in output.iterdir()} == set(OUTPUT_FILES)
    signal = pd.read_parquet(output / "signal.parquet")
    assert signal.groupby("date").size().to_dict() == {20230102: 2, 20230104: 2}
    assert set(signal["ticker"]) == {"000002.SZ", "000003.SZ"}
    universe = pd.read_parquet(output / "optimizer_universe.parquet")
    assert universe.groupby("date").size().to_dict() == {20230102: 4, 20230104: 4}
    dates = pd.read_parquet(output / "rebalance_dates.parquet")
    assert dates["date"].tolist() == [20230102, 20230104]
    assert dates["universe_count"].tolist() == [4, 4]
    benchmark = pd.read_parquet(output / "benchmark_weights.parquet")
    assert benchmark.groupby("date")["benchmark_weight"].sum().round(12).eq(1.0).all()


def test_prepare_alpha191_lgbm_inputs_excludes_small_missing_benchmark_weight(
    tmp_path: Path,
) -> None:
    signal_file = tmp_path / "signal.csv"
    benchmark_file = tmp_path / "benchmark.parquet"
    eligibility_file = tmp_path / "eligibility.parquet"
    pd.DataFrame(
        [
            {"date": 20230102, "symbol": "000001.SZ", "signal": 0.1},
            {"date": 20230102, "symbol": "000002.SZ", "signal": 0.2},
        ]
    ).to_csv(signal_file, index=False)
    pd.DataFrame(
        [
            {"date": 20230102, "ticker": "000001.SZ", "benchmark_weight": 0.999},
            {"date": 20230102, "ticker": "000099.SZ", "benchmark_weight": 0.001},
        ]
    ).to_parquet(benchmark_file, index=False)
    pd.DataFrame(
        [
            {"date": 20230102, "ticker": "000001.SZ", "tradable": True},
            {"date": 20230102, "ticker": "000002.SZ", "tradable": True},
        ]
    ).to_parquet(eligibility_file, index=False)

    prepare_inputs(
        signal_file=signal_file,
        benchmark_file=benchmark_file,
        eligibility_file=eligibility_file,
        start_date=20230102,
        end_date=20230102,
        rebalance_every=1,
        candidate_count=1,
        benchmark_exclusion_tolerance=0.002,
        output_dir=tmp_path / "output",
    )

    benchmark = pd.read_parquet(tmp_path / "output" / "benchmark_weights.parquet")
    assert benchmark["ticker"].tolist() == ["000001.SZ"]
    assert benchmark["benchmark_weight"].tolist() == [1.0]
    manifest = json.loads(
        (tmp_path / "output" / "input_manifest.json").read_text(encoding="utf-8")
    )
    summary = manifest["benchmark_quality"]
    assert summary["excluded_without_market_record_count"] == 1
    assert summary["excluded_without_market_record_weight_max"] == 0.001
