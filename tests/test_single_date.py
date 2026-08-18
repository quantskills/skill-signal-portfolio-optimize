from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime import __version__
from portfolio_runtime.errors import ConfigError, InputDataError, OptimizationError
from portfolio_runtime.pipeline import OUTPUT_FILES, run_single_date


class SingleDatePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.date = 20230104
        self.tickers = [f"{index:06d}.SZ" for index in range(1, 7)]

        self.signal = self.root / "signal.csv"
        pd.DataFrame(
            {
                "date": self.date,
                "ticker": self.tickers,
                "prediction": [0.8, 0.6, 0.4, 0.2, -0.1, -0.4],
            }
        ).to_csv(self.signal, index=False)

        self.benchmark = self.root / "benchmark.csv"
        pd.DataFrame(
            {
                "date": self.date,
                "ticker": self.tickers,
                "benchmark_weight": [1.0 / 6.0] * 6,
            }
        ).to_csv(self.benchmark, index=False)

        self.current = self.root / "current.csv"
        pd.DataFrame(
            {
                "date": self.date,
                "ticker": self.tickers,
                "current_weight": [1.0 / 6.0] * 6,
            }
        ).to_csv(self.current, index=False)

        self.sectors = self.root / "sectors.csv"
        pd.DataFrame(
            {
                "ticker": self.tickers,
                "sector": ["A", "A", "B", "B", "C", "C"],
            }
        ).to_csv(self.sectors, index=False)

        self.exposures = self.root / "exposures.csv"
        pd.DataFrame(
            {
                "ticker": self.tickers,
                "SIZE": [-1.2, -0.7, -0.2, 0.2, 0.7, 1.2],
                "BETA": [0.4, -0.3, 0.2, -0.1, 0.1, -0.3],
            }
        ).to_csv(self.exposures, index=False)

        self.tradability = self.root / "tradability.csv"
        pd.DataFrame(
            {
                "date": self.date,
                "ticker": self.tickers,
                "tradable": [False, True, True, True, True, True],
            }
        ).to_csv(self.tradability, index=False)

        diagonal = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13])
        covariance = np.diag(diagonal)
        covariance += 0.01 * np.ones((6, 6))
        self.covariance = self.root / "covariance.csv"
        pd.DataFrame(
            covariance, index=self.tickers, columns=self.tickers
        ).to_csv(self.covariance, index_label="ticker")

        self.config = self.root / "config.yaml"
        self.write_config()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, **overrides: object) -> None:
        config = {
            "schema_version": 1,
            "signal": {
                "type": "rank_score",
                "higher_is_better": True,
                "winsorize_mad": 5.0,
                "zscore": True,
                "annualized_alpha_scale": 0.04,
            },
            "covariance": {
                "annualized": True,
                "periods_per_year": 252,
                "eigenvalue_floor": 1.0e-10,
                "symmetry_tolerance": 1.0e-10,
            },
            "optimizer": {
                "risk_aversion": 5.0,
                "turnover_penalty": 0.001,
                "smoothing_epsilon": 1.0e-8,
                "max_iterations": 2000,
                "ftol": 1.0e-10,
            },
            "constraints": {
                "max_weight": 0.30,
                "max_active_weight": 0.20,
                "max_turnover": 0.25,
                "max_tracking_error": 0.15,
                "sector_active_limit": 0.15,
                "factor_active_limit": {"SIZE": 0.30, "BETA": 0.20},
                "weight_sum_tolerance": 1.0e-8,
                "constraint_tolerance": 1.0e-6,
            },
            "baseline": {"top_n": 3},
        }
        for dotted_key, value in overrides.items():
            section, key = dotted_key.split("__", 1)
            config[section][key] = value
        self.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def run_pipeline(
        self,
        output_name: str = "output",
        candidate_file: Path | None = None,
    ) -> dict[str, object]:
        return run_single_date(
            config_path=self.config,
            signal_file=self.signal,
            candidate_file=candidate_file,
            covariance_file=self.covariance,
            benchmark_file=self.benchmark,
            current_weights_file=self.current,
            sector_file=self.sectors,
            exposure_file=self.exposures,
            tradability_file=self.tradability,
            requested_date=self.date,
            output_dir=self.root / output_name,
        )

    def write_v2_config(
        self,
        *,
        styles: dict[str, dict[str, object]],
        industry: dict[str, object] | None = None,
    ) -> None:
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["schema_version"] = 2
        constraints = config["constraints"]
        constraints["sector_active_limit"] = None
        constraints["factor_active_limit"] = None
        constraints["max_turnover"] = None
        constraints["max_tracking_error"] = None
        constraints["industry_active_range"] = industry
        constraints["style_active_ranges"] = styles
        self.config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def write_v3_config(self) -> None:
        self.write_v2_config(
            styles={"SIZE": {"target_active": 0.0, "tolerance": 0.30}}
        )
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["schema_version"] = 3
        config["signal"]["missing_prediction_policy"] = "error_except_frozen"
        self.config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def write_v4_config(
        self, candidate_weight_range: dict[str, float | None]
    ) -> None:
        self.write_v3_config()
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["schema_version"] = 4
        config["signal"]["rank_transform"] = "normal_score"
        config["signal"]["rank_power"] = 1.0
        config["constraints"]["candidate_weight_range"] = candidate_weight_range
        self.config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def write_candidates(
        self, tickers: list[str], name: str = "candidates.csv"
    ) -> Path:
        path = self.root / name
        pd.DataFrame({"date": self.date, "ticker": tickers}).to_csv(path, index=False)
        return path

    def test_end_to_end_writes_stable_outputs_and_satisfies_constraints(self) -> None:
        result = self.run_pipeline()
        output = Path(result["output_dir"])
        self.assertEqual(set(OUTPUT_FILES), {path.name for path in output.iterdir()})

        weights = pd.read_parquet(output / "target_weights.parquet")
        optimized = weights.loc[weights["portfolio"] == "risk_optimized"].set_index("ticker")
        baseline = weights.loc[
            weights["portfolio"] == "equal_weight_signal"
        ].set_index("ticker")
        self.assertAlmostEqual(float(optimized["target_weight"].sum()), 1.0, places=7)
        self.assertAlmostEqual(
            float(optimized.loc[self.tickers[0], "target_weight"]), 1.0 / 6.0, places=7
        )
        self.assertEqual(int((baseline["target_weight"] > 0).sum()), 3)

        constraints = json.loads(
            (output / "constraint_diagnostics.json").read_text(encoding="utf-8")
        )
        self.assertTrue(constraints["risk_optimized"]["constraints"]["passed"])
        self.assertFalse(constraints["equal_weight_signal"]["constraints"]["passed"])
        self.assertLessEqual(
            constraints["risk_optimized"]["constraints"]["one_way_turnover"], 0.250001
        )

        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["schema_version"], 4)
        self.assertEqual(manifest["implementation_version"], __version__)
        self.assertEqual(manifest["asset_count"], 6)
        self.assertEqual(set(manifest["outputs"]), set(OUTPUT_FILES))

    def test_indexed_parquet_exposures_are_supported(self) -> None:
        indexed = pd.read_csv(self.exposures).set_index("ticker")
        self.exposures = self.root / "exposures.parquet"
        indexed.to_parquet(self.exposures)
        result = self.run_pipeline("indexed-exposure-output")
        self.assertEqual(result["status"], "success")

    def test_v2_controls_size_and_targets_positive_beta_exposure(self) -> None:
        exposures = pd.read_csv(self.exposures)
        exposures["UNUSED"] = np.linspace(-1.0, 1.0, len(exposures))
        exposures.to_csv(self.exposures, index=False)
        self.write_v2_config(
            styles={
                "SIZE": {"target_active": 0.0, "tolerance": 0.05},
                "BETA": {"target_active": 0.10, "tolerance": 0.02},
                "UNUSED": {"enabled": False},
            },
            industry={
                "default": {"target_active": 0.0, "tolerance": 0.20},
                "overrides": {
                    "A": {"min_active": -0.10, "max_active": 0.10}
                },
            },
        )
        tradability = pd.read_csv(self.tradability)
        tradability["tradable"] = True
        tradability.to_csv(self.tradability, index=False)
        result = self.run_pipeline("v2-output")
        diagnostics = json.loads(
            (Path(result["output_dir"]) / "constraint_diagnostics.json").read_text(
                encoding="utf-8"
            )
        )["risk_optimized"]["constraints"]
        self.assertTrue(diagnostics["passed"])
        self.assertLessEqual(abs(diagnostics["style_exposures"]["SIZE"]["active_exposure"]), 0.050001)
        beta = diagnostics["style_exposures"]["BETA"]
        self.assertGreaterEqual(beta["active_exposure"], 0.079999)
        self.assertLessEqual(beta["active_exposure"], 0.120001)
        self.assertNotIn("UNUSED", diagnostics["style_exposures"])
        self.assertIn("A", diagnostics["industry_exposures"])

    def test_v2_rejects_missing_enabled_style_column(self) -> None:
        self.write_v2_config(
            styles={
                "SIZE": {"target_active": 0.0, "tolerance": 0.05},
                "MOMENTUM": {"target_active": 0.2, "tolerance": 0.1},
            }
        )
        with self.assertRaisesRegex(InputDataError, "MOMENTUM"):
            self.run_pipeline()

    def test_v2_requires_enabled_size_constraint(self) -> None:
        self.write_v2_config(
            styles={"BETA": {"target_active": 0.0, "tolerance": 0.1}}
        )
        with self.assertRaisesRegex(ConfigError, "SIZE must be enabled"):
            self.run_pipeline()

    def test_v3_requires_strict_missing_prediction_policy(self) -> None:
        self.write_v2_config(
            styles={"SIZE": {"target_active": 0.0, "tolerance": 0.30}}
        )
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["schema_version"] = 3
        self.config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ConfigError, "error_except_frozen"):
            self.run_pipeline("invalid-v3-output")

    def test_benchmark_only_asset_joins_optimization_universe(self) -> None:
        extra_ticker = "000007.SZ"
        signal = pd.read_csv(self.signal)
        signal = signal.loc[signal["ticker"] != self.tickers[-1]]
        signal.to_csv(self.signal, index=False)
        benchmark = pd.read_csv(self.benchmark)
        benchmark["benchmark_weight"] = 0.8 / len(benchmark)
        benchmark = pd.concat(
            [
                benchmark,
                pd.DataFrame(
                    {
                        "date": [self.date],
                        "ticker": [extra_ticker],
                        "benchmark_weight": [0.2],
                    }
                ),
            ],
            ignore_index=True,
        )
        benchmark.to_csv(self.benchmark, index=False)
        current = pd.read_csv(self.current)
        current["current_weight"] = 0.8 / len(current)
        current = pd.concat(
            [
                current,
                pd.DataFrame(
                    {
                        "date": [self.date],
                        "ticker": [extra_ticker],
                        "current_weight": [0.2],
                    }
                ),
            ],
            ignore_index=True,
        )
        current.to_csv(self.current, index=False)
        sectors = pd.read_csv(self.sectors)
        sectors = pd.concat(
            [sectors, pd.DataFrame({"ticker": [extra_ticker], "sector": ["C"]})],
            ignore_index=True,
        )
        sectors.to_csv(self.sectors, index=False)
        exposures = pd.read_csv(self.exposures)
        exposures = pd.concat(
            [
                exposures,
                pd.DataFrame({"ticker": [extra_ticker], "SIZE": [0.0], "BETA": [0.0]}),
            ],
            ignore_index=True,
        )
        exposures.to_csv(self.exposures, index=False)
        tradability = pd.read_csv(self.tradability)
        tradability["tradable"] = True
        tradability = pd.concat(
            [
                tradability,
                pd.DataFrame(
                    {"date": [self.date], "ticker": [extra_ticker], "tradable": [True]}
                ),
            ],
            ignore_index=True,
        )
        tradability.to_csv(self.tradability, index=False)
        covariance = pd.read_csv(self.covariance, index_col=0)
        covariance[extra_ticker] = 0.0
        covariance.loc[extra_ticker] = 0.0
        covariance.loc[extra_ticker, extra_ticker] = 0.10
        covariance.to_csv(self.covariance, index_label="ticker")

        result = self.run_pipeline("union-output")
        weights = pd.read_parquet(
            Path(result["output_dir"]) / "target_weights.parquet"
        )
        row = weights.loc[
            (weights["portfolio"] == "risk_optimized")
            & (weights["ticker"] == extra_ticker)
        ].iloc[0]
        self.assertFalse(bool(row["has_signal"]))
        self.assertTrue(pd.isna(row["raw_prediction"]))
        self.assertEqual(float(row["expected_return"]), 0.0)
        manifest = json.loads(
            (Path(result["output_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(manifest["benchmark_only_asset_count"], 1)

    def test_full_signal_calibration_is_separate_from_candidate_universe(self) -> None:
        candidates = self.write_candidates(self.tickers[:2])

        result = self.run_pipeline("candidate-output", candidates)

        output = Path(result["output_dir"])
        weights = pd.read_parquet(output / "target_weights.parquet")
        optimized = weights.loc[
            weights["portfolio"].eq("risk_optimized")
        ].set_index("ticker")
        baseline = weights.loc[
            weights["portfolio"].eq("equal_weight_signal")
        ].set_index("ticker")
        diagnostics = json.loads(
            (output / "signal_diagnostics.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (output / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(diagnostics["calibration_asset_count"], 6)
        self.assertEqual(diagnostics["candidate_asset_count"], 2)
        self.assertEqual(diagnostics["optimization_asset_count"], 6)
        self.assertEqual(
            float(diagnostics["optimization_prediction_coverage"]), 1.0
        )
        self.assertFalse(bool(optimized.loc[self.tickers[-1], "is_candidate"]))
        self.assertTrue(bool(optimized.loc[self.tickers[-1], "signal_available"]))
        self.assertLess(float(optimized.loc[self.tickers[-1], "signal_score"]), 0.0)
        self.assertEqual(
            set(baseline.index[baseline["target_weight"].gt(0.0)]),
            set(self.tickers[:2]),
        )
        self.assertEqual(
            diagnostics["portfolio_candidate_weight"]["equal_weight_signal"], 1.0
        )
        self.assertEqual(manifest["candidate_asset_count"], 2)
        self.assertEqual(manifest["benchmark_or_current_only_asset_count"], 4)

    def test_candidate_missing_from_full_signal_is_rejected(self) -> None:
        candidates = self.write_candidates([*self.tickers[:2], "000099.SZ"])
        with self.assertRaisesRegex(InputDataError, "candidate universe contains"):
            self.run_pipeline("missing-candidate-output", candidates)

    def test_v4_enforces_candidate_aggregate_weight_and_reports_slack(self) -> None:
        self.write_v4_config({"min_weight": 0.55, "max_weight": None})
        tradability = pd.read_csv(self.tradability)
        tradability["tradable"] = True
        tradability.to_csv(self.tradability, index=False)
        candidates = self.write_candidates(self.tickers[:2])

        result = self.run_pipeline("candidate-floor-output", candidates)

        constraints = json.loads(
            (
                Path(result["output_dir"]) / "constraint_diagnostics.json"
            ).read_text(encoding="utf-8")
        )["risk_optimized"]["constraints"]
        assert constraints["candidate_weight"] >= 0.55 - 1.0e-6
        assert constraints["candidate_weight_range"]["passed"]
        assert constraints["constraint_slacks"]["candidate_weight_lower"] >= -1.0e-6

    def test_v4_fails_when_candidate_weight_floor_is_infeasible(self) -> None:
        self.write_v4_config({"min_weight": 0.70, "max_weight": None})
        tradability = pd.read_csv(self.tradability)
        tradability["tradable"] = True
        tradability.to_csv(self.tradability, index=False)
        candidates = self.write_candidates(self.tickers[:2])

        with self.assertRaisesRegex(OptimizationError, "infeasible"):
            self.run_pipeline("candidate-infeasible-output", candidates)

    def test_v3_rejects_missing_prediction_for_tradable_asset(self) -> None:
        self.write_v3_config()
        signal = pd.read_csv(self.signal)
        signal = signal.loc[signal["ticker"].ne(self.tickers[-1])]
        signal.to_csv(self.signal, index=False)
        candidates = self.write_candidates(self.tickers[:2])

        with self.assertRaisesRegex(InputDataError, "missing full-universe prediction"):
            self.run_pipeline("strict-missing-output", candidates)

    def test_v3_allows_missing_prediction_only_for_frozen_positive_holding(self) -> None:
        self.write_v3_config()
        signal = pd.read_csv(self.signal)
        signal = signal.loc[signal["ticker"].ne(self.tickers[0])]
        signal.to_csv(self.signal, index=False)
        candidates = self.write_candidates(self.tickers[1:])

        result = self.run_pipeline("strict-frozen-output", candidates)

        output = Path(result["output_dir"])
        weights = pd.read_parquet(output / "target_weights.parquet")
        frozen = weights.loc[
            weights["portfolio"].eq("risk_optimized")
            & weights["ticker"].eq(self.tickers[0])
        ].iloc[0]
        diagnostics = json.loads(
            (output / "signal_diagnostics.json").read_text(encoding="utf-8")
        )
        self.assertFalse(bool(frozen["signal_available"]))
        self.assertAlmostEqual(float(frozen["target_weight"]), 1.0 / 6.0)
        self.assertEqual(float(frozen["signal_score"]), 0.0)
        self.assertEqual(diagnostics["allowed_frozen_missing_prediction_count"], 1)

    def test_signal_rejects_additional_factor_columns(self) -> None:
        frame = pd.read_csv(self.signal)
        frame["alpha002"] = np.arange(len(frame))
        frame.to_csv(self.signal, index=False)
        with self.assertRaisesRegex(InputDataError, "one prediction column"):
            self.run_pipeline()

    def test_covariance_rejects_material_asymmetry(self) -> None:
        covariance = pd.read_csv(self.covariance, index_col=0)
        covariance.iloc[0, 1] += 0.02
        covariance.to_csv(self.covariance, index_label="ticker")
        with self.assertRaisesRegex(InputDataError, "asymmetry"):
            self.run_pipeline()

    def test_infeasible_position_cap_fails_before_writing_outputs(self) -> None:
        self.write_config(constraints__max_weight=0.10)
        tradability = pd.read_csv(self.tradability)
        tradability["tradable"] = True
        tradability.to_csv(self.tradability, index=False)
        with self.assertRaisesRegex(OptimizationError, "cannot support full investment"):
            self.run_pipeline()
        self.assertFalse((self.root / "output").exists())

    def test_expected_return_requires_preserved_units(self) -> None:
        self.write_config(signal__type="expected_return", signal__zscore=True)
        with self.assertRaisesRegex(ConfigError, "must be false"):
            self.run_pipeline()

    def test_optional_diagnostic_inputs_do_not_enable_constraints(self) -> None:
        self.write_config(
            constraints__sector_active_limit=None,
            constraints__factor_active_limit=None,
        )
        result = self.run_pipeline()
        self.assertEqual(result["status"], "success")

    def test_frozen_weight_breach_is_preserved_and_disclosed(self) -> None:
        current = pd.read_csv(self.current)
        current.loc[0, "current_weight"] = 0.35
        current.loc[1:, "current_weight"] = 0.65 / 5.0
        current.to_csv(self.current, index=False)

        result = self.run_pipeline()

        diagnostics = json.loads(
            (Path(result["output_dir"]) / "constraint_diagnostics.json").read_text(
                encoding="utf-8"
            )
        )["risk_optimized"]["constraints"]
        weights = pd.read_parquet(Path(result["output_dir"]) / "target_weights.parquet")
        frozen = weights.loc[
            weights["ticker"].eq("000001.SZ")
            & weights["portfolio"].eq("risk_optimized"),
            "target_weight",
        ].iloc[0]
        self.assertAlmostEqual(float(frozen), 0.35)
        self.assertTrue(diagnostics["passed"])
        self.assertEqual(
            diagnostics["frozen_bound_exceptions"][0]["ticker"], "000001.SZ"
        )
        self.assertLessEqual(diagnostics["maximum_controllable_weight"], 0.30 + 1e-6)

    def test_public_cli_runs_end_to_end(self) -> None:
        output = self.root / "cli-output"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_single_date.py"),
            "--config",
            str(self.config),
            "--signal-file",
            str(self.signal),
            "--covariance-file",
            str(self.covariance),
            "--benchmark-file",
            str(self.benchmark),
            "--current-weights-file",
            str(self.current),
            "--sector-file",
            str(self.sectors),
            "--exposure-file",
            str(self.exposures),
            "--tradability-file",
            str(self.tradability),
            "--date",
            str(self.date),
            "--output-dir",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        envelope = json.loads(completed.stdout)
        self.assertEqual(envelope["status"], "success")
        self.assertEqual(set(OUTPUT_FILES), {path.name for path in output.iterdir()})

    def test_constant_rank_signal_is_rejected(self) -> None:
        signal = pd.read_csv(self.signal)
        signal["prediction"] = 1.0
        signal.to_csv(self.signal, index=False)
        with self.assertRaisesRegex(InputDataError, "no cross-sectional variation"):
            self.run_pipeline()



if __name__ == "__main__":
    unittest.main()
