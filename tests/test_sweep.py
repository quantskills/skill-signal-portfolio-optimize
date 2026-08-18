from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from portfolio_runtime.config import DEFAULT_CONFIG, load_config  # noqa: E402
from portfolio_runtime.errors import ConfigError  # noqa: E402
from portfolio_runtime.sweep import (  # noqa: E402
    apply_dotted_overrides,
    load_sweep_variants,
)


def test_example_sweep_matrix_is_valid() -> None:
    variants = load_sweep_variants(ROOT / "examples" / "parameter-sweep.yaml")
    assert len(variants) == 5
    assert len({variant["name"] for variant in variants}) == 5


def test_dotted_overrides_validate_resolved_config() -> None:
    base = deepcopy(DEFAULT_CONFIG)
    resolved = apply_dotted_overrides(
        base,
        {
            "signal.rank_transform": "normal_score",
            "constraints.candidate_weight_range": {
                "min_weight": 0.3,
                "max_weight": None,
            },
        },
    )
    assert resolved["signal"]["rank_transform"] == "normal_score"
    assert resolved["constraints"]["candidate_weight_range"] == {
        "min_weight": 0.3,
        "max_weight": 1.0,
    }


def test_resolved_sweep_config_can_be_serialized_and_reloaded(tmp_path: Path) -> None:
    base = load_config(ROOT / "examples" / "alpha191-lgbm-oos-config.yaml")
    resolved = apply_dotted_overrides(
        base,
        {
            "signal.rank_transform": "uniform",
            "constraints.candidate_weight_range": None,
        },
    )
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    reloaded = load_config(path)

    assert reloaded == resolved


def test_sweep_rejects_unknown_override_path() -> None:
    with pytest.raises(ConfigError, match="unknown configuration override path"):
        apply_dotted_overrides(deepcopy(DEFAULT_CONFIG), {"signal.unknown": 1})


def test_sweep_rejects_duplicate_variant_names(tmp_path: Path) -> None:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "variants": [
                    {"name": "same", "overrides": {}},
                    {"name": "same", "overrides": {}},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_sweep_variants(path)
