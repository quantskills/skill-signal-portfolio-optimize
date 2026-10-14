from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from portfolio_runtime.config import validate_config


def test_full_universe_industry_size_preset_is_valid() -> None:
    source = ROOT / "examples" / "full-universe-industry-size.yaml"
    config = validate_config(yaml.safe_load(source.read_text()))
    assert config["baseline"]["top_n"] == 500
    assert config["constraints"]["max_weight"] == 0.002
    assert config["constraints"]["industry_active_range"]["default"]["lower_active"] == -0.01
    assert config["constraints"]["style_active_ranges"]["SIZE"]["upper_active"] == 0.10
    assert config["selection"]["mode"] == "none"
