from __future__ import annotations

import json
from pathlib import Path

from attacks.a3_card_testing.presentation import build_presentation


def test_presentation_builds_four_offline_svg_charts_and_data(tmp_path):
    summary = Path(__file__).parents[1] / "results" / "summary.json"
    result = build_presentation(summary, tmp_path)

    assert result["throughput_cards_per_hour"] == [40.0, 12.0, 4.0]
    assert len(result["signal_mortality"]) == 9
    assert len(result["bin_clustering"]) == 3
    assert result["bin_clustering"][0]["distinct_bins"] == 2
    for name in (
        "round_metrics.svg",
        "throughput.svg",
        "signal_mortality.svg",
        "bin_clustering.svg",
    ):
        content = (tmp_path / name).read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert "A3" in content or "Round" in content or "throughput" in content
    saved = json.loads((tmp_path / "presentation.json").read_text(encoding="utf-8"))
    assert saved == result
