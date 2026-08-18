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

from portfolio_runtime.risk_model import (
    OUTPUT_FILES,
    build_risk_model_from_files,
    estimate_structural_risk_model,
)


class StructuralRiskModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(17)
        self.dates = pd.bdate_range("2022-01-03", periods=100).strftime("%Y%m%d")
        self.tickers = pd.Index(
            [f"{index:06d}.SZ" for index in range(1, 13)], name="ticker"
        )
        industries = np.array(["A"] * 6 + ["B"] * 6)
        base_cap = np.exp(np.linspace(18.0, 22.0, len(self.tickers)))
        cap_values = np.vstack(
            [base_cap * np.exp(0.001 * step) for step in range(len(self.dates))]
        )
        market_cap = pd.DataFrame(
            cap_values, index=self.dates, columns=self.tickers
        )
        returns = pd.DataFrame(np.nan, index=self.dates, columns=self.tickers)
        for position in range(1, len(self.dates)):
            log_cap = np.log(market_cap.iloc[position - 1])
            size = (log_cap - log_cap.mean()) / log_cap.std(ddof=0)
            size_return = rng.normal(0.0002, 0.003)
            industry_return = {"A": rng.normal(0.0001, 0.004), "B": rng.normal(-0.0001, 0.004)}
            specific = rng.normal(0.0, 0.008, len(self.tickers))
            returns.iloc[position] = [
                size_return * size.iloc[index] + industry_return[industries[index]] + specific[index]
                for index in range(len(self.tickers))
            ]
        self.returns = returns
        self.market_cap = market_cap
        self.industry = pd.DataFrame(
            {
                "ticker": self.tickers,
                "industry": industries,
                "in_date": "20200101",
                "out_date_normalized": "99991231",
            }
        )
        self.universe = self.tickers[:8]
        self.config = {
            "schema_version": 1,
            "lookback_days": 80,
            "minimum_regression_periods": 60,
            "minimum_cross_section_assets": 10,
            "minimum_specific_observations": 30,
            "return_winsorize_mad": 8.0,
            "size_winsorize_mad": 5.0,
            "regression_weight_power": 0.5,
            "factor_covariance_halflife": 30.0,
            "specific_variance_halflife": 20.0,
            "factor_covariance_diagonal_shrinkage": 0.1,
            "specific_variance_median_shrinkage": 0.1,
            "annualization_periods": 252,
            "factor_eigenvalue_floor": 1.0e-8,
            "specific_variance_floor": 1.0e-6,
            "max_specific_imputation_fraction": 0.25,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def estimate(self, returns: pd.DataFrame | None = None):
        return estimate_structural_risk_model(
            returns=self.returns if returns is None else returns,
            market_cap=self.market_cap,
            industry_history=self.industry,
            target_universe=self.universe,
            requested_date=self.dates[-1],
            config=self.config,
        )

    def test_covariance_is_aligned_symmetric_and_positive_definite(self) -> None:
        result = self.estimate()
        self.assertEqual(result.asset_cov.shape, (8, 8))
        self.assertEqual(result.asset_cov.index.tolist(), self.universe.tolist())
        np.testing.assert_allclose(result.asset_cov, result.asset_cov.T, atol=1e-12)
        self.assertGreater(float(np.linalg.eigvalsh(result.asset_cov).min()), 0.0)
        self.assertTrue((result.specific_var > 0).all())
        self.assertEqual(result.factor_cov.index.tolist(), ["SIZE", "INDUSTRY:A", "INDUSTRY:B"])

    def test_future_return_rows_do_not_change_historical_asof_result(self) -> None:
        first = self.estimate()
        extended = self.returns.copy()
        extended.loc["20261231"] = np.linspace(-0.5, 0.5, len(self.tickers))
        second = self.estimate(extended)
        np.testing.assert_allclose(first.asset_cov, second.asset_cov, atol=1e-12)
        np.testing.assert_allclose(first.factor_cov, second.factor_cov, atol=1e-12)

    def test_market_style_model_runs_without_industry_history(self) -> None:
        config = {
            **self.config,
            "industry_mode": "disabled",
            "style_factors": ["SIZE", "BETA", "MOMENTUM", "RESVOL", "NLSIZE"],
            "style_lookback_days": 50,
            "style_minimum_observations": 20,
            "momentum_skip_days": 5,
            "minimum_regression_periods": 45,
        }

        result = estimate_structural_risk_model(
            returns=self.returns,
            market_cap=self.market_cap,
            industry_history=None,
            target_universe=self.universe,
            requested_date=self.dates[-1],
            config=config,
        )

        self.assertEqual(
            result.factor_cov.index.tolist(),
            ["MARKET", "SIZE", "BETA", "MOMENTUM", "RESVOL", "NLSIZE"],
        )
        self.assertFalse(result.manifest["industry_history_used"])
        self.assertGreater(float(np.linalg.eigvalsh(result.asset_cov).min()), 0.0)

    def test_public_cli_writes_stable_outputs(self) -> None:
        returns_path = self.root / "returns.parquet"
        cap_path = self.root / "market_cap.parquet"
        industry_path = self.root / "industry.parquet"
        universe_path = self.root / "universe.parquet"
        config_path = self.root / "risk.yaml"
        self.returns.to_parquet(returns_path)
        self.market_cap.to_parquet(cap_path)
        pd.DataFrame(
            {
                "stock_symbol": self.tickers,
                "l1_code": ["A"] * 6 + ["B"] * 6,
                "in_date": 20200101,
                "out_date": pd.NA,
            }
        ).to_parquet(industry_path, index=False)
        pd.DataFrame(
            {"date": self.dates[-1], "ticker": self.universe}
        ).to_parquet(universe_path, index=False)
        config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        output = self.root / "output"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_risk_model.py"),
                "--config",
                str(config_path),
                "--returns-file",
                str(returns_path),
                "--market-cap-file",
                str(cap_path),
                "--industry-file",
                str(industry_path),
                "--universe-file",
                str(universe_path),
                "--date",
                self.dates[-1],
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        envelope = json.loads(completed.stdout)
        self.assertEqual(envelope["status"], "success")
        self.assertEqual(set(OUTPUT_FILES), {path.name for path in output.iterdir()})
        manifest = json.loads(
            (output / "risk_model_manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["is_proprietary_barra_model"])
        self.assertIn("preceding aligned trading date", manifest["timing_contract"])

    def test_rolling_builder_resumes_completed_date_caches(self) -> None:
        returns_path = self.root / "batch_returns.parquet"
        cap_path = self.root / "batch_market_cap.parquet"
        industry_path = self.root / "batch_industry.parquet"
        universe_path = self.root / "batch_universe.parquet"
        config_path = self.root / "batch_risk.yaml"
        self.returns.to_parquet(returns_path)
        self.market_cap.to_parquet(cap_path)
        pd.DataFrame(
            {
                "stock_symbol": self.tickers,
                "l1_code": ["A"] * 6 + ["B"] * 6,
                "in_date": 20200101,
                "out_date": pd.NA,
            }
        ).to_parquet(industry_path, index=False)
        selected_dates = [self.dates[-2], self.dates[-1]]
        pd.DataFrame(
            [
                {"date": date, "ticker": ticker}
                for date in selected_dates
                for ticker in self.universe
            ]
        ).to_parquet(universe_path, index=False)
        config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        output = self.root / "batch_output"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_rolling_risk_models.py"),
            "--config",
            str(config_path),
            "--returns-file",
            str(returns_path),
            "--market-cap-file",
            str(cap_path),
            "--industry-file",
            str(industry_path),
            "--universe-file",
            str(universe_path),
            "--output-root",
            str(output),
        ]

        first = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        first_result = json.loads(first.stdout)
        self.assertEqual(first_result["built_count"], 2)
        self.assertEqual(first_result["skipped_count"], 0)
        for date in selected_dates:
            self.assertEqual(
                {path.name for path in (output / f"date={date}").iterdir()},
                set(OUTPUT_FILES),
            )

        second = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        second_result = json.loads(second.stdout)
        self.assertEqual(second_result["built_count"], 0)
        self.assertEqual(second_result["skipped_count"], 2)
        root_manifest = json.loads(
            (output / "rolling_risk_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(root_manifest["status"], "success")
        self.assertEqual(root_manifest["skipped_dates"], selected_dates)

        without_industry = command.copy()
        industry_position = without_industry.index("--industry-file")
        del without_industry[industry_position : industry_position + 2]
        stale = subprocess.run(
            without_industry, cwd=ROOT, text=True, capture_output=True, check=False
        )
        self.assertEqual(stale.returncode, 2)
        self.assertIn("stale risk-model cache", stale.stderr)


if __name__ == "__main__":
    unittest.main()
