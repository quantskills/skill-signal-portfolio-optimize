from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.errors import InputDataError
from prepare_existing_experiment_inputs import OUTPUT_FILES, prepare_inputs


class PrepareExistingInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidates = self.root / "candidates.parquet"
        pd.DataFrame(
            {
                "date": [20230104] * 4,
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "alpha_score": [1.5, 0.5, -0.5, -1.5],
            }
        ).to_parquet(self.candidates, index=False)
        self.covariance = self.root / "asset_cov.parquet"
        tickers = ["000001.SZ", "000002.SZ", "000003.SZ"]
        pd.DataFrame(
            np.diag([0.1, 0.2, 0.3]), index=tickers, columns=tickers
        ).to_parquet(self.covariance)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_writes_contract_and_discloses_equal_weight_benchmark(self) -> None:
        result = prepare_inputs(
            candidate_file=self.candidates,
            covariance_file=self.covariance,
            requested_date=20230104,
            output_dir=self.root / "inputs",
        )
        output = Path(result["output_dir"])
        self.assertEqual(set(OUTPUT_FILES), {path.name for path in output.iterdir()})
        signal = pd.read_parquet(output / "signal.parquet")
        benchmark = pd.read_parquet(output / "benchmark_weights.parquet")
        covariance = pd.read_parquet(output / "covariance.parquet")
        self.assertEqual(signal.columns.tolist(), ["date", "ticker", "prediction"])
        self.assertEqual(len(signal), 3)
        self.assertAlmostEqual(float(benchmark["benchmark_weight"].sum()), 1.0)
        self.assertEqual(covariance.shape, (3, 3))
        manifest = json.loads(
            (output / "input_manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["benchmark_is_market_index"])
        self.assertEqual(manifest["excluded_without_covariance_tickers"], ["000004.SZ"])

    def test_rejects_covariance_asset_outside_candidates(self) -> None:
        covariance = pd.read_parquet(self.covariance)
        covariance.loc["999999.SZ"] = 0.0
        covariance["999999.SZ"] = 0.0
        covariance.loc["999999.SZ", "999999.SZ"] = 0.1
        covariance.to_parquet(self.covariance)
        with self.assertRaisesRegex(InputDataError, "outside candidate universe"):
            prepare_inputs(
                candidate_file=self.candidates,
                covariance_file=self.covariance,
                requested_date=20230104,
                output_dir=self.root / "inputs",
            )


if __name__ == "__main__":
    unittest.main()
