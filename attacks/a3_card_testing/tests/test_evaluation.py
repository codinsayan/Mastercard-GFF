from __future__ import annotations

import json

from attacks.a3_card_testing.evaluation import (
    EvaluationConfig,
    build_a3_detector_report,
    build_dataset,
    run_evaluation,
)
from attacks.a3_card_testing.features import ROUND0_FEATURES
from attacks.a3_card_testing.generator import A3Generator


def _codex_proposal(_prompt: str) -> dict:
    return {
        "reasoning": "Reduce observable velocity while retaining bounded campaign utility.",
        "hypotheses": [
            {
                "signal_targeted": "a3_device_card_fanin_1h",
                "action": "rotate device fingerprints more frequently",
                "cost_to_attacker": "additional coordination",
            }
        ],
        "proposed_params": {
            **A3Generator().default_params(),
            "cards_per_device": 3,
            "n_merchants": 20,
        },
    }


def _tiny_config() -> EvaluationConfig:
    return EvaluationConfig(
        train_campaigns=2,
        test_campaigns=1,
        events_per_campaign=80,
        search_campaigns=1,
        search_events_per_campaign=60,
        cem_samples=4,
        cem_iterations=1,
    )


def test_a3_report_contains_metrics_state_and_caught_missed_examples():
    params = A3Generator().default_params()
    dataset = build_dataset(params, [123], 80)
    scores = dataset.features[ROUND0_FEATURES[0]].to_numpy(dtype=float)
    report = build_a3_detector_report(
        params,
        {"detector_metrics": {"roc_auc": 0.75}},
        dataset,
        scores,
        threshold=1.0,
    )

    assert report["attack_id"] == "A3"
    assert report["attack_state"]["current_params"] == params
    assert report["round_metrics"]["detector_metrics"]["roc_auc"] == 0.75
    assert report["caught_vs_missed"]
    assert {"risk_score", "caught"}.issubset(report["caught_vs_missed"][0])


def test_three_round_evaluation_is_measured_and_writes_conventional_artifacts(tmp_path):
    summary = run_evaluation(tmp_path, config=_tiny_config(), invoke=_codex_proposal)

    assert len(summary["rounds"]) == 3
    assert [item["target"] for item in summary["throughput"]] == [40, 12, 4]
    assert [item["requested"] for item in summary["throughput"]] == [40, 12, 4]
    assert all(item["actual"] > 0 for item in summary["throughput"])
    assert len(summary["signal_mortality"]) == 9
    for round_number in range(3):
        metrics = json.loads(
            (tmp_path / f"round_{round_number}" / "metrics.json").read_text(encoding="utf-8")
        )
        assert 0.0 <= metrics["detector_metrics"]["roc_auc"] <= 1.0
        assert len(metrics["per_feature_auc"]) == 9
    for round_number in (1, 2):
        reasoning = json.loads(
            (tmp_path / f"round_{round_number}" / "redteam_reasoning.json").read_text(
                encoding="utf-8"
            )
        )
        adaptation = json.loads(
            (tmp_path / f"round_{round_number}" / "adaptation.json").read_text(
                encoding="utf-8"
            )
        )
        assert reasoning["fallback_used"] is False
        assert adaptation["evaluated_candidates"] > 0
    assert (tmp_path / "summary.json").is_file()


def test_evaluation_is_deterministic_with_fixed_reasoner(tmp_path):
    first = run_evaluation(tmp_path / "first", config=_tiny_config(), invoke=_codex_proposal)
    second = run_evaluation(tmp_path / "second", config=_tiny_config(), invoke=_codex_proposal)

    assert first == second
