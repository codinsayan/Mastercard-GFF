import json

import pandas as pd
import pytest

from loop.run_rounds import run_rounds


def test_round_zero_writes_frozen_output_contract(tmp_path):
    summaries = run_rounds(
        rounds=[0],
        seed=7,
        results_root=tmp_path,
        population_events=50,
        max_victims=20,
        use_codex=False,
        cem_samples=2,
        cem_iterations=1,
    )
    round_dir = tmp_path / "round_0"
    expected = {
        "events.parquet",
        "features.parquet",
        "scores.parquet",
        "metrics.json",
        "redteam_reasoning.json",
    }
    assert {path.name for path in round_dir.iterdir()} == expected
    events = pd.read_parquet(round_dir / "events.parquet")
    features = pd.read_parquet(round_dir / "features.parquet")
    scores = pd.read_parquet(round_dir / "scores.parquet")
    assert len(events) == len(features) == len(scores)
    metrics = json.loads((round_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == summaries[0]
    assert metrics["development_fixture"] is True
    assert "fpr_at_recall_0.85" in metrics["overall"]


@pytest.mark.parametrize("rounds", [[], [1], [0, 2], [0, 1, 1]])
def test_round_sequence_must_be_unique_contiguous_and_start_at_zero(tmp_path, rounds):
    with pytest.raises(ValueError, match="rounds"):
        run_rounds(rounds=rounds, results_root=tmp_path, population_events=50)

