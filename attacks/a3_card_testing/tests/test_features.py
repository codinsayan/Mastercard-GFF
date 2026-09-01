from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attacks.a3_card_testing.features import FEATURES, A3Features
from attacks.a3_card_testing.generator import A3Generator
from core.population import BasePopulation, PopulationConfig


def _events(rows: list[tuple]) -> pd.DataFrame:
    normalised = [row + ("411111", "APPROVED", {}) if len(row) == 8 else row for row in rows]
    return pd.DataFrame(
        normalised,
        columns=[
            "event_id",
            "timestamp",
            "device_id",
            "card_id",
            "counterparty_id",
            "counterparty_type",
            "amount",
            "channel",
            "card_bin",
            "auth_result",
            "meta",
        ],
    ).assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))


def test_round0_features_have_contract_shape_and_preserve_input_order():
    events = _events(
        [
            (
                "e2",
                "2026-01-01 00:20:00",
                "d1",
                "c1",
                "m1",
                "MERCHANT",
                120.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
            (
                "e0",
                "2026-01-01 00:00:00",
                "d1",
                "c1",
                "m1",
                "MERCHANT",
                50.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
            (
                "e1",
                "2026-01-01 00:10:00",
                "d1",
                "c2",
                "m1",
                "MERCHANT",
                80.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
        ]
    )
    features = A3Features().build(events)

    assert A3Features.attack_id == "A3"
    assert list(features.index) == ["e2", "e0", "e1"]
    assert features.index.name == "event_id"
    assert tuple(features.columns) == FEATURES
    assert all(column.startswith("a3_") for column in features)
    assert all(np.issubdtype(dtype, np.floating) for dtype in features.dtypes)
    assert np.isfinite(features.to_numpy()).all()


def test_device_fanin_counts_distinct_cards_in_causal_open_left_window():
    events = _events(
        [
            ("e0", "2026-01-01 00:00:00", "d1", "c1", "m0", "MERCHANT", 20.0, "CNP"),
            ("e1", "2026-01-01 00:10:00", "d1", "c2", "m0", "MERCHANT", 20.0, "CNP"),
            ("e2", "2026-01-01 00:20:00", "d1", "c1", "m0", "MERCHANT", 20.0, "CNP"),
            ("e3", "2026-01-01 01:00:00", "d1", "c3", "m0", "MERCHANT", 20.0, "CNP"),
            ("e4", "2026-01-01 01:20:00", "d1", "c4", "m0", "MERCHANT", 20.0, "CNP"),
        ]
    )
    fanin = A3Features().build(events)["a3_device_card_fanin_1h"]

    assert fanin.to_dict() == {"e0": 1.0, "e1": 2.0, "e2": 2.0, "e3": 3.0, "e4": 2.0}


def test_merchant_micro_rate_counts_current_and_prior_micro_transactions():
    events = _events(
        [
            ("e0", "2026-01-01 00:00:00", "d0", "c0", "m1", "MERCHANT", 50.0, "CNP"),
            ("e1", "2026-01-01 00:10:00", "d1", "c1", "m1", "MERCHANT", 100.0, "CNP"),
            ("e2", "2026-01-01 00:20:00", "d2", "c2", "m1", "MERCHANT", 100.01, "CNP"),
            ("e3", "2026-01-01 00:30:00", "d3", "c3", "m2", "MERCHANT", 20.0, "CNP"),
            ("e4", "2026-01-01 01:00:00", "d4", "c4", "m1", "MERCHANT", 20.0, "CNP"),
        ]
    )
    rate = A3Features().build(events)["a3_merchant_micro_txn_rate_1h"]

    assert rate.to_dict() == {"e0": 1.0, "e1": 2.0, "e2": 2.0, "e3": 1.0, "e4": 2.0}


def test_amount_percentile_uses_only_strictly_earlier_cnp_amounts():
    events = _events(
        [
            ("e0", "2026-01-01 00:00:00", "d0", "c0", "m0", "MERCHANT", 50.0, "CNP"),
            ("e1", "2026-01-01 00:10:00", "d1", "c1", "m1", "MERCHANT", 80.0, "CNP"),
            ("e2", "2026-01-01 00:20:00", "d2", "c2", "m2", "MERCHANT", 60.0, "UPI_P2M"),
            ("e3", "2026-01-01 00:30:00", "d3", "c3", "m3", "MERCHANT", 60.0, "CNP"),
            ("e4", "2026-01-01 00:30:00", "d4", "c4", "m4", "MERCHANT", 1000.0, "CNP"),
            ("e5", "2026-01-01 00:40:00", "d5", "c5", "m5", "MERCHANT", 80.0, "CNP"),
        ]
    )
    percentile = A3Features().build(events)["a3_amount_percentile"]

    assert percentile["e0"] == 0.5
    assert percentile["e1"] == 1.0
    assert percentile["e2"] == pytest.approx(0.5)
    assert percentile["e3"] == pytest.approx(0.5)
    assert percentile["e4"] == 1.0
    assert percentile["e5"] == pytest.approx(0.75)


def test_first_seen_recency_uses_strictly_earlier_card_history():
    events = _events(
        [
            ("e0", "2026-01-01 00:00:00", "d0", "c1", "m0", "MERCHANT", 20.0, "CNP"),
            ("e1", "2026-01-01 00:00:00", "d1", "c1", "m1", "MERCHANT", 20.0, "CNP"),
            ("e2", "2026-01-01 00:20:00", "d2", "c1", "m2", "MERCHANT", 20.0, "CNP"),
            ("e3", "2026-01-01 00:30:00", "d3", "c2", "m3", "MERCHANT", 20.0, "CNP"),
        ]
    )
    recency = A3Features().build(events)["a3_card_first_seen_recency"]

    assert recency.to_dict() == {"e0": 0.0, "e1": 0.0, "e2": 1200.0, "e3": 0.0}


def test_same_timestamp_rolling_order_is_deterministic_by_event_id():
    events = _events(
        [
            ("e2", "2026-01-01 00:00:00", "d1", "c2", "m1", "MERCHANT", 20.0, "CNP"),
            ("e1", "2026-01-01 00:00:00", "d1", "c1", "m1", "MERCHANT", 20.0, "CNP"),
        ]
    )
    features = A3Features().build(events)

    assert features.loc["e1", "a3_device_card_fanin_1h"] == 1.0
    assert features.loc["e2", "a3_device_card_fanin_1h"] == 2.0
    assert features.loc["e1", "a3_merchant_micro_txn_rate_1h"] == 1.0
    assert features.loc["e2", "a3_merchant_micro_txn_rate_1h"] == 2.0
    assert features["a3_amount_percentile"].eq(0.5).all()


def test_future_event_does_not_change_existing_features():
    events = _events(
        [
            ("e0", "2026-01-01 00:00:00", "d1", "c1", "m1", "MERCHANT", 20.0, "CNP"),
            ("e1", "2026-01-01 00:10:00", "d1", "c2", "m1", "MERCHANT", 30.0, "CNP"),
        ]
    )
    before = A3Features().build(events)
    future = _events([("future", "2026-02-01", "d1", "c9", "m1", "MERCHANT", 9999.0, "CNP")])
    after = A3Features().build(pd.concat([events, future], ignore_index=True))

    pd.testing.assert_frame_equal(before, after.loc[before.index])


def test_merchant_set_jaccard_and_friction_are_card_attributed_and_causal():
    low = {"a3_merchant_friction_tier": "LOW"}
    high = {"a3_merchant_friction_tier": "HIGH"}
    events = _events(
        [
            (
                "e0",
                "2026-01-01 00:00:00",
                "d0",
                "c1",
                "m1",
                "MERCHANT",
                50.0,
                "CNP",
                "411111",
                "APPROVED",
                low,
            ),
            (
                "e1",
                "2026-01-01 00:05:00",
                "d1",
                "c2",
                "m1",
                "MERCHANT",
                50.0,
                "CNP",
                "411111",
                "APPROVED",
                high,
            ),
            (
                "e2",
                "2026-01-01 00:10:00",
                "d0",
                "c1",
                "m2",
                "MERCHANT",
                50.0,
                "CNP",
                "411111",
                "APPROVED",
                high,
            ),
        ]
    )
    features = A3Features().build(events)

    assert features["a3_merchant_set_jaccard"].to_dict() == {
        "e0": 0.0,
        "e1": 1.0,
        "e2": 0.5,
    }
    assert features["a3_merchant_friction_score"].to_dict() == {
        "e0": 0.0,
        "e1": 1.0,
        "e2": 0.5,
    }


def test_bin_features_count_distinct_active_cards_and_expire_at_one_hour():
    events = _events(
        [
            (
                "e0",
                "2026-01-01 00:00:00",
                "d0",
                "c1",
                "m1",
                "MERCHANT",
                50.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
            (
                "e1",
                "2026-01-01 00:10:00",
                "d1",
                "c2",
                "m2",
                "MERCHANT",
                50.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
            (
                "e2",
                "2026-01-01 00:20:00",
                "d2",
                "c3",
                "m3",
                "MERCHANT",
                50.0,
                "CNP",
                "522145",
                "APPROVED",
                {},
            ),
            (
                "e3",
                "2026-01-01 01:00:00",
                "d3",
                "c4",
                "m4",
                "MERCHANT",
                50.0,
                "CNP",
                "522145",
                "APPROVED",
                {},
            ),
        ]
    )
    features = A3Features().build(events)

    assert features.loc["e2", "a3_bin_cooccurrence_score"] == pytest.approx(1 / 3)
    assert features.loc["e2", "a3_bin_entropy_1h"] == pytest.approx(
        -(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3)) / np.log(2)
    )
    assert features.loc["e3", "a3_bin_entropy_1h"] == pytest.approx(
        -(1 / 3 * np.log(1 / 3) + 2 / 3 * np.log(2 / 3)) / np.log(2)
    )


def test_decline_retry_pattern_marks_only_small_prompt_same_card_perturbations():
    events = _events(
        [
            (
                "e0",
                "2026-01-01 00:00:00",
                "d0",
                "c1",
                "m1",
                "MERCHANT",
                100.0,
                "CNP",
                "411111",
                "DECLINED",
                {},
            ),
            (
                "e1",
                "2026-01-01 00:01:00",
                "d0",
                "c1",
                "m2",
                "MERCHANT",
                105.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
            (
                "e2",
                "2026-01-01 00:02:00",
                "d0",
                "c2",
                "m2",
                "MERCHANT",
                105.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
            (
                "e3",
                "2026-01-01 00:03:00",
                "d0",
                "c1",
                "m3",
                "MERCHANT",
                120.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
            (
                "e4",
                "2026-01-01 00:06:00",
                "d0",
                "c1",
                "m4",
                "MERCHANT",
                102.0,
                "CNP",
                "411111",
                "APPROVED",
                {},
            ),
        ]
    )
    pattern = A3Features().build(events)["a3_decline_retry_pattern"]

    assert pattern.to_dict() == {"e0": 0.0, "e1": 1.0, "e2": 0.0, "e3": 0.0, "e4": 0.0}


def test_missing_values_use_zero_and_neutral_conventions():
    events = _events([("e0", "2026-01-01", "", "", "", "", np.nan, "CNP")])
    features = A3Features().build(events)

    assert features.loc["e0"].to_dict() == {
        "a3_device_card_fanin_1h": 0.0,
        "a3_merchant_micro_txn_rate_1h": 0.0,
        "a3_amount_percentile": 0.5,
        "a3_card_first_seen_recency": 0.0,
        "a3_merchant_set_jaccard": 0.0,
        "a3_merchant_friction_score": 0.5,
        "a3_bin_cooccurrence_score": 0.0,
        "a3_bin_entropy_1h": 1.0,
        "a3_decline_retry_pattern": 0.0,
    }


def test_generated_events_build_without_non_finite_values():
    reference = pd.DataFrame(
        {"amount": [12.5, 48.0, 109.0, 799.0], "mcc": ["5411", "5812", "5732", "5999"]}
    )
    population = BasePopulation(reference, PopulationConfig(events=200)).generate(seed=101)
    events = A3Generator().generate(A3Generator().default_params(), population, seed=77)
    features = A3Features().build(events)

    assert len(features) == len(events)
    assert list(features.index) == list(events["event_id"])
    assert np.isfinite(features.to_numpy()).all()
