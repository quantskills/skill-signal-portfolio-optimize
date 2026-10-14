from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from portfolio_runtime.exposure_report import ExposureReportConfig, render_exposure_report


def test_render_exposure_report(tmp_path: Path) -> None:
    exposure_file = tmp_path / "exposure_timeseries.parquet"
    frame = pd.DataFrame(
        [
            {"date": "20250102", "exposure_type": "style", "name": "SIZE", "benchmark_exposure": 0.0, "portfolio_exposure": 0.04, "active_exposure": 0.04, "lower_bound": -0.1, "upper_bound": 0.1, "lower_slack": 0.14, "upper_slack": 0.06, "binding": False, "passed": True},
            {"date": "20250103", "exposure_type": "style", "name": "SIZE", "benchmark_exposure": 0.0, "portfolio_exposure": -0.02, "active_exposure": -0.02, "lower_bound": -0.1, "upper_bound": 0.1, "lower_slack": 0.08, "upper_slack": 0.12, "binding": False, "passed": True},
            {"date": "20250102", "exposure_type": "industry", "name": "Banks", "benchmark_exposure": 0.10, "portfolio_exposure": 0.11, "active_exposure": 0.01, "lower_bound": -0.02, "upper_bound": 0.02, "lower_slack": 0.03, "upper_slack": 0.01, "binding": False, "passed": True},
            {"date": "20250103", "exposure_type": "industry", "name": "Banks", "benchmark_exposure": 0.10, "portfolio_exposure": 0.08, "active_exposure": -0.02, "lower_bound": -0.02, "upper_bound": 0.02, "lower_slack": 0.0, "upper_slack": 0.04, "binding": True, "passed": True},
        ]
    )
    frame.to_parquet(exposure_file, index=False)
    output = tmp_path / "report"
    manifest = render_exposure_report(ExposureReportConfig(exposure_file=exposure_file, output_dir=output))
    assert manifest["status"] == "success"
    assert manifest["style_factor_count"] == 1
    assert manifest["industry_factor_count"] == 1
    assert (output / "style_active_exposure.png").is_file()
    assert (output / "industry_active_exposure.png").is_file()
    assert (output / "Style_radar.png").is_file()
    assert (output / "Industry_radar.png").is_file()
    exposure_table = pd.read_csv(output / "Explo.csv")
    assert {"port", "benchmark", "date"}.issubset(exposure_table.columns)
    summary = pd.read_csv(output / "exposure_summary.csv")
    assert set(summary["name"]) == {"SIZE", "Banks"}
    assert int(summary.loc[summary["name"] == "Banks", "binding_count"].iloc[0]) == 1


def test_render_exposure_report_filters_named_portfolio(tmp_path: Path) -> None:
    exposure_file = tmp_path / "exposure_timeseries.parquet"
    frame = pd.DataFrame(
        [
            {"date": "20250102", "portfolio": "equal_weight_signal", "exposure_type": "style", "name": "SIZE", "benchmark_exposure": 0.0, "portfolio_exposure": 0.20, "active_exposure": 0.20},
            {"date": "20250102", "portfolio": "risk_optimized", "exposure_type": "style", "name": "SIZE", "benchmark_exposure": 0.0, "portfolio_exposure": 0.05, "active_exposure": 0.05},
        ]
    )
    frame.to_parquet(exposure_file, index=False)
    output = tmp_path / "report"
    manifest = render_exposure_report(
        ExposureReportConfig(
            exposure_file=exposure_file,
            output_dir=output,
            portfolio="equal_weight_signal",
        )
    )
    assert manifest["portfolio"] == "equal_weight_signal"
    table = pd.read_csv(output / "Explo.csv")
    assert table["port"].tolist() == [0.20]
