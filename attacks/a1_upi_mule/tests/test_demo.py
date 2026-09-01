"""Replay-only A1 demo view-model tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from attacks.a1_upi_mule.demo_data import (
    ArtifactError,
    GraphViewModel,
    RoundArtifacts,
    build_campaign_graph,
    build_replay_steps,
    build_signal_mortality,
    campaign_events,
    format_currency,
    format_duration,
    list_a1_campaigns,
    load_round_artifacts,
    render_graph_svg,
    short_node_label,
    validate_artifacts,
)


def _events() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01 12:00:00")
    rows = [
        ("e3", base + pd.Timedelta(8, unit="min"), "l2", "cash", 95, "MULE_L2", "CASHOUT"),
        ("e1", base, "victim", "l1", 100, "VICTIM", "MULE_L1"),
        ("e2", base + pd.Timedelta(5, unit="min"), "l1", "l2", 98, "MULE_L1", "MULE_L2"),
    ]
    return pd.DataFrame(
        [
            {
                "event_id": event_id,
                "timestamp": timestamp,
                "attack_id": "A1",
                "customer_id": source,
                "counterparty_id": target,
                "counterparty_account_age_days": 3,
                "amount": amount,
                "meta": {
                    "a1_campaign_id": "campaign-1",
                    "a1_source_role": source_role,
                    "a1_destination_role": destination_role,
                    "a1_call_active": source_role == "VICTIM",
                },
            }
            for event_id, timestamp, source, target, amount, source_role, destination_role in rows
        ]
    )


def _artifacts(events: pd.DataFrame | None = None) -> RoundArtifacts:
    return RoundArtifacts(
        round_number=0,
        events=_events() if events is None else events,
        features=pd.DataFrame({"event_id": ["e1", "e2", "e3"]}),
        scores=pd.DataFrame({"event_id": ["e3", "e1"], "risk_score": [0.8, 0.9]}),
        metrics={"per_attack": {"A1": {"recall": 1.0}}},
        reasoning={},
    )


def test_campaign_selection_is_a1_only_and_chronological():
    events = _events()
    legitimate = events.iloc[[0]].copy()
    legitimate["event_id"] = "legit"
    legitimate["attack_id"] = "LEGIT"
    combined = pd.concat([legitimate, events], ignore_index=True)
    assert list_a1_campaigns(combined) == ["campaign-1"]
    assert campaign_events(combined, "campaign-1")["event_id"].tolist() == ["e1", "e2", "e3"]


def test_replay_joins_scores_by_event_id_and_keeps_missing_visible():
    replay = build_replay_steps(_events(), _artifacts().scores)
    assert replay["event_id"].tolist() == ["e1", "e2", "e3"]
    assert replay["risk_score"].tolist()[0] == pytest.approx(0.9)
    assert np.isnan(replay["risk_score"].tolist()[1])
    assert replay.loc[1, "decision"] == "Score unavailable"
    assert replay["elapsed_seconds"].tolist() == [0.0, 300.0, 480.0]


def test_graph_edges_and_roles_are_stable_and_bounded():
    first = build_campaign_graph(_events(), max_nodes=4)
    second = build_campaign_graph(_events(), max_nodes=4)
    pd.testing.assert_frame_equal(first.nodes, second.nodes)
    pd.testing.assert_frame_equal(first.edges, second.edges)
    assert set(first.nodes["role"]) == {"VICTIM", "MULE_L1", "MULE_L2", "CASHOUT"}
    assert list(first.edges[["source", "target"]].itertuples(index=False, name=None)) == [
        ("victim", "l1"), ("l1", "l2"), ("l2", "cash")
    ]
    assert not first.sampled


def test_malformed_metadata_does_not_crash_view_models():
    events = _events()
    events.at[0, "meta"] = "malformed"
    replay = build_replay_steps(events, _artifacts().scores)
    assert replay.loc[replay["event_id"].eq("e3"), "source_role"].item() == "UNKNOWN"


def test_signal_mortality_contains_measured_values_only():
    metrics = {
        0: {"signal_auc": {"a1_age": 0.9}, "per_attack": {"A1": {}}},
        1: {"signal_auc": {"a1_age": 0.6, "a1_graph": 0.8}, "per_attack": {"A1": {}}},
    }
    result = build_signal_mortality(metrics)
    assert len(result) == 3
    assert not ((result["round"].eq(0)) & result["signal"].eq("a1_graph")).any()


def test_validation_names_missing_columns():
    artifacts = _artifacts(events=pd.DataFrame({"event_id": ["e1"]}))
    with pytest.raises(ArtifactError, match="events.parquet is missing columns"):
        validate_artifacts(artifacts)


def test_loader_reports_missing_files_with_command(tmp_path):
    with pytest.raises(ArtifactError, match="python -m loop.run_rounds"):
        load_round_artifacts(tmp_path, 0)


def test_loader_accepts_complete_cached_contract(tmp_path):
    round_dir = tmp_path / "round_0"
    round_dir.mkdir()
    artifacts = _artifacts()
    artifacts.events.to_parquet(round_dir / "events.parquet", index=False)
    artifacts.features.to_parquet(round_dir / "features.parquet", index=False)
    artifacts.scores.to_parquet(round_dir / "scores.parquet", index=False)
    (round_dir / "metrics.json").write_text(json.dumps(artifacts.metrics), encoding="utf-8")
    (round_dir / "redteam_reasoning.json").write_text("{}", encoding="utf-8")
    loaded = load_round_artifacts(tmp_path, 0)
    assert loaded.round_number == 0
    assert len(loaded.events) == 3


def test_format_currency_switches_to_lakh_notation():
    assert format_currency(2500) == "₹2,500"
    assert format_currency(240_000) == "₹2.40 lakh"


def test_format_duration_scales_units():
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m 5s"
    assert format_duration(3725) == "1h 2m"


def test_short_node_label_uses_trailing_digits():
    assert short_node_label("a1_mule_l1_0007") == "#0007"
    assert short_node_label("no-digits-here") == "here"


def test_render_graph_svg_is_deterministic_and_bounded():
    graph = build_campaign_graph(_events(), max_nodes=4)
    first = render_graph_svg(graph, active_edges=2)
    second = render_graph_svg(graph, active_edges=2)
    assert first == second
    assert first.count("<circle") >= len(graph.nodes)
    assert first.count("<line") == len(graph.edges)


def test_render_graph_svg_handles_empty_graph_without_crashing():
    empty = GraphViewModel(nodes=pd.DataFrame(columns=["node_id", "role", "x", "y"]),
                            edges=pd.DataFrame(columns=["event_id", "timestamp", "source", "target", "amount"]),
                            sampled=False)
    svg = render_graph_svg(empty, active_edges=0)
    assert svg.startswith("<svg")
    assert "No graph data" in svg


def test_render_graph_svg_escapes_malformed_node_ids():
    events = _events()
    events.at[0, "customer_id"] = "<script>alert(1)</script>"
    graph = build_campaign_graph(events)
    svg = render_graph_svg(graph, active_edges=1)
    assert "<script>" not in svg
