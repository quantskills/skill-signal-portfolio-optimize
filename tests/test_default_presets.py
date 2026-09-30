from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from portfolio_runtime.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_lexicographic_default_preset_matches_locked_parameters() -> None:
    config = load_config(ROOT / "defaults" / "default_lexicographic_v1_8.yaml")
    assert config["schema_version"] == 5
    assert config["optimizer"]["objective_mode"] == "lexicographic_signal_cost"
    assert config["optimizer"]["solver_backend"] == "clarabel_socp"
    assert config["constraints"]["max_tracking_error"] == 0.06
    assert config["constraints"]["max_turnover"] == 0.12
    assert config["constraints"]["max_weight"] == 0.04
    assert config["constraints"]["max_active_weight"] == 0.04


def test_259_flexible_preset_is_factor_model_and_industry_controlled() -> None:
    config = load_config(
        ROOT / "defaults" / "259_flexible_benchmark_enhancement_keep07.yaml"
    )
    assert config["covariance"]["risk_form"] == "factor_model"
    assert config["optimizer"]["objective_mode"] == "lexicographic_signal_cost"
    assert config["constraints"]["max_tracking_error"] == 0.06
    assert config["constraints"]["max_turnover"] == 0.12
    assert config["constraints"]["industry_active_range"] is not None


def test_259_strict_p005_preset_has_exact_selection_contract() -> None:
    config = load_config(ROOT / "defaults" / "259_strict_topn_p005.yaml")
    assert config["schema_version"] == 10
    assert config["optimizer"]["objective_mode"] == "risk_aware_selection"
    assert config["selection"]["risk_penalty"] == 0.005
    assert config["constraints"]["candidate_weight_range"] == {
        "min_weight": 1.0,
        "max_weight": 1.0,
    }
