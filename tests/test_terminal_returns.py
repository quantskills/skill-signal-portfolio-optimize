from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_terminal_returns import prepare_terminal_returns  # noqa: E402


def test_terminal_returns_are_explicit_atomic_and_reusable(tmp_path: Path) -> None:
    source = tmp_path / "returns.parquet"
    output = tmp_path / "derived" / "returns.parquet"
    manifest = tmp_path / "derived" / "manifest.json"
    pd.DataFrame(
        [
            {"date": 20230102, "ticker": "A", "return": 0.01},
            {"date": 20230102, "ticker": "B", "return": 0.02},
            {"date": 20230103, "ticker": "A", "return": 0.03},
            {"date": 20230103, "ticker": "B", "return": 0.04},
            {"date": 20230104, "ticker": "A", "return": 0.05},
        ]
    ).to_parquet(source, index=False)

    first = prepare_terminal_returns(
        input_file=source,
        output_file=output,
        manifest_file=manifest,
        end_date=20230104,
    )
    second = prepare_terminal_returns(
        input_file=source,
        output_file=output,
        manifest_file=manifest,
        end_date=20230104,
    )

    derived = pd.read_parquet(output)
    terminal = derived.loc[(derived["date"] == "20230104") & (derived["ticker"] == "B")]
    assert terminal["return"].tolist() == [-1.0]
    assert first["terminal_event_count"] == 1
    assert second["cache_status"] == "reused"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "not an observed return" in payload["assumption"]


def test_terminal_returns_allow_no_events(tmp_path: Path) -> None:
    source = tmp_path / "returns.parquet"
    output = tmp_path / "derived" / "returns.parquet"
    manifest = tmp_path / "derived" / "manifest.json"
    pd.DataFrame(
        [
            {"date": 20230102, "ticker": "A", "return": 0.01},
            {"date": 20230103, "ticker": "A", "return": 0.02},
        ]
    ).to_parquet(source, index=False)

    result = prepare_terminal_returns(
        input_file=source,
        output_file=output,
        manifest_file=manifest,
        end_date=20230103,
    )

    assert result["terminal_event_count"] == 0
    assert len(pd.read_parquet(output)) == 2
