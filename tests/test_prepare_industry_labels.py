from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from portfolio_runtime.errors import InputDataError
from prepare_industry_labels import prepare_industry_labels


class PrepareIndustryLabelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.history = self.root / "history.parquet"
        pd.DataFrame(
            {
                "stock_symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "l1_code": ["A", "B", "A"],
                "in_date": [20200101, 20230105, 20200101],
                "out_date": [20230104, pd.NA, pd.NA],
            }
        ).to_parquet(self.history, index=False)
        self.universe = self.root / "universe.parquet"
        pd.DataFrame(
            {
                "date": [20230104, 20230105, 20230105],
                "ticker": ["000001.SZ", "000001.SZ", "000002.SZ"],
            }
        ).to_parquet(self.universe, index=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_interval_end_is_inclusive_and_manifest_is_written(self) -> None:
        output = self.root / "labels.parquet"
        payload = prepare_industry_labels(
            history_file=self.history,
            universe_file=self.universe,
            output_file=output,
        )
        labels = pd.read_parquet(output).sort_values(["date", "ticker"])
        self.assertEqual(labels["sector"].tolist(), ["A", "B", "A"])
        self.assertEqual(payload["minimum_observed_coverage"], 1.0)
        manifest = json.loads((self.root / "labels.manifest.json").read_text())
        self.assertEqual(manifest["date_count"], 2)

    def test_missing_label_fails_closed_by_default(self) -> None:
        universe = self.root / "missing.parquet"
        pd.DataFrame(
            {"date": [20230104, 20230104], "ticker": ["000001.SZ", "000003.SZ"]}
        ).to_parquet(universe, index=False)
        with self.assertRaisesRegex(InputDataError, "below minimum_coverage"):
            prepare_industry_labels(
                history_file=self.history,
                universe_file=universe,
                output_file=self.root / "missing_labels.parquet",
            )

    def test_missing_candidate_can_be_excluded_with_audit_outputs(self) -> None:
        universe = self.root / "missing_exclude.parquet"
        pd.DataFrame(
            {
                "date": [20230104, 20230104, 20230105],
                "ticker": ["000001.SZ", "000003.SZ", "000002.SZ"],
            }
        ).to_parquet(universe, index=False)
        output = self.root / "filtered_labels.parquet"
        payload = prepare_industry_labels(
            history_file=self.history,
            universe_file=universe,
            output_file=output,
            missing_policy="exclude",
        )

        labels = pd.read_parquet(output).sort_values(["date", "ticker"])
        self.assertEqual(
            labels[["date", "ticker", "sector"]].values.tolist(),
            [
                ["20230104", "000001.SZ", "A"],
                ["20230105", "000002.SZ", "A"],
            ],
        )
        filtered = pd.read_parquet(
            self.root / "filtered_labels.filtered_universe.parquet"
        )
        self.assertEqual(filtered["ticker"].tolist(), ["000001.SZ", "000002.SZ"])
        exclusions = pd.read_parquet(
            self.root / "filtered_labels.exclusions.parquet"
        )
        self.assertEqual(
            exclusions.to_dict("records"),
            [
                {
                    "date": "20230104",
                    "ticker": "000003.SZ",
                    "reason": "missing_industry_asof",
                }
            ],
        )
        self.assertEqual(payload["excluded_row_count"], 1)
        self.assertEqual(payload["missing_policy"], "exclude")

    def test_exclude_requires_remaining_candidate_count(self) -> None:
        universe = self.root / "all_missing.parquet"
        pd.DataFrame(
            {"date": [20230104], "ticker": ["000003.SZ"]}
        ).to_parquet(universe, index=False)
        with self.assertRaisesRegex(
            InputDataError, "no industry-valid candidates remain"
        ):
            prepare_industry_labels(
                history_file=self.history,
                universe_file=universe,
                output_file=self.root / "all_missing_labels.parquet",
                missing_policy="exclude",
            )

    def test_overlapping_history_fails_closed(self) -> None:
        overlapping = self.root / "overlap.parquet"
        pd.DataFrame(
            {
                "stock_symbol": ["000001.SZ", "000001.SZ"],
                "l1_code": ["A", "B"],
                "in_date": [20200101, 20230104],
                "out_date": [20230105, pd.NA],
            }
        ).to_parquet(overlapping, index=False)
        with self.assertRaisesRegex(InputDataError, "overlapping intervals"):
            prepare_industry_labels(
                history_file=overlapping,
                universe_file=self.universe,
                output_file=self.root / "overlap_labels.parquet",
            )


if __name__ == "__main__":
    unittest.main()
