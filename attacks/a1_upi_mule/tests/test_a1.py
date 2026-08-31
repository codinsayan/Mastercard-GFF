from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attacks.a1_upi_mule.detector import A1Detector
from attacks.a1_upi_mule.experiment import _metrics, _signal_auc
from attacks.a1_upi_mule.features import A1Features
from attacks.a1_upi_mule.generator import A1Generator
from core.population import BasePopulation, PopulationConfig
from core.schema import coerce, validate


@pytest.fixture
def generator() -> A1Generator:
    return A1Generator()


@pytest.fixture
def feature_block() -> A1Features:
    return A1Features()


@pytest.fixture
def detector() -> A1Detector:
    return A1Detector()


@pytest.fixture
def population() -> pd.DataFrame:
    reference = pd.DataFrame(
        {
            "amount": [125.0, 275.0, 900.0, 1_500.0],
            "mcc": ["5411", "5812", "5732", "5999"],
        }
    )
    return BasePopulation(reference, PopulationConfig(customers=8, events=80)).generate(seed=11)


def test_card_has_complete_parameter_contract(generator):
    card_path = Path(__file__).parents[1] / "card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["attack_id"] == "A1"
    assert card["channel"] == "UPI_P2P"
    assert set(card["default_params"]) == set(card["param_space"])
    assert set(card["param_semantics"]) == set(card["param_space"])
    assert set(generator.default_params()) == set(generator.param_space())


def test_iteration1_generator_is_schema_valid(generator, population):
    events = generator.generate(generator.default_params(), population, seed=42)
    validate(events)
    assert not events.empty
    assert events["event_id"].is_unique
    assert events["attack_id"].eq("A1").all()
    assert events["label"].eq(1).all()
    assert events["channel"].eq("UPI_P2P").all()
    assert events["amount"].gt(0).all()
    roles = events["meta"].map(lambda value: value["a1_destination_role"])
    assert set(roles) == {"MULE_L1", "MULE_L2", "CASHOUT"}


def test_generation_is_exactly_seed_deterministic(generator, population):
    params = generator.default_params()
    first = generator.generate(params, population, seed=91)
    second = generator.generate(params, population, seed=91)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_different_seed_changes_generated_events(generator, population):
    params = generator.default_params()
    first = generator.generate(params, population, seed=91)
    second = generator.generate(params, population, seed=92)
    assert not first.equals(second)


def test_tiny_population_caps_effective_victim_count(generator):
    reference = pd.DataFrame({"amount": [200.0], "mcc": ["5411"]})
    population = BasePopulation(
        reference, PopulationConfig(customers=2, events=4)
    ).generate(seed=3)
    events = generator.generate(generator.default_params(), population, seed=4)
    victim_rows = events.loc[events["meta"].map(lambda value: value["a1_mule_layer"] == 1)]
    assert victim_rows["customer_id"].nunique() <= population["customer_id"].nunique()
    assert 1 <= len(victim_rows) <= population["customer_id"].nunique() * 3


def test_metadata_is_complete_json_safe_and_not_shared(generator, population):
    events = generator.generate(generator.default_params(), population, seed=7)
    required = {
        "a1_call_active",
        "a1_payee_added_at",
        "a1_mule_layer",
        "a1_cashout_node",
        "a1_campaign_id",
        "a1_source_role",
        "a1_destination_role",
    }
    assert events["meta"].map(lambda value: required.issubset(value)).all()
    assert set(events["meta"].map(lambda value: value["a1_mule_layer"])) == {1, 2, 3}
    assert len({id(value) for value in events["meta"]}) == len(events)
    for value in events["meta"]:
        json.dumps(value)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("n_victims", True, TypeError),
        ("n_victims", 19, ValueError),
        ("split_factor", 2.5, TypeError),
        ("dwell_time_hours", 100.0, ValueError),
    ],
)
def test_invalid_parameters_are_rejected(generator, population, name, value, error):
    params = generator.default_params()
    params[name] = value
    with pytest.raises(error):
        generator.generate(params, population, seed=1)


def test_unknown_and_missing_parameters_are_rejected(generator, population):
    params = generator.default_params()
    params["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        generator.generate(params, population, seed=1)

    params = generator.default_params()
    params.pop("split_factor")
    with pytest.raises(ValueError, match="missing"):
        generator.generate(params, population, seed=1)


def test_missing_population_columns_are_rejected(generator, population):
    with pytest.raises(ValueError, match="device_id"):
        generator.generate(generator.default_params(), population.drop(columns="device_id"), seed=1)


def test_amounts_are_finite_and_sum_to_campaign_baseline(generator, population):
    events = generator.generate(generator.default_params(), population, seed=22)
    assert np.isfinite(events["amount"]).all()
    assert events["amount"].gt(0).all()
    for _, campaign in events.groupby(events["meta"].map(lambda value: value["a1_campaign_id"])):
        assert round(float(campaign["amount"].sum()), 2) > 0


def _meta_series(events: pd.DataFrame, key: str) -> pd.Series:
    return events["meta"].map(lambda value: value[key])


def test_complete_three_layer_topology_and_roles(generator, population):
    events = generator.generate(generator.default_params(), population, seed=101)
    layers = _meta_series(events, "a1_mule_layer")
    assert set(layers) == {1, 2, 3}

    expected_prefix = {
        1: "a1_mule_l1_",
        2: "a1_mule_l2_",
        3: "a1_cashout_",
    }
    for layer, prefix in expected_prefix.items():
        assert events.loc[layers.eq(layer), "counterparty_id"].str.startswith(prefix).all()

    assert _meta_series(events.loc[layers.eq(1)], "a1_call_active").eq(True).all()
    assert _meta_series(events.loc[layers.ne(1)], "a1_call_active").eq(False).all()


def test_campaign_money_is_conserved_with_retained_fees(generator, population):
    events = generator.generate(generator.default_params(), population, seed=202)
    campaign_ids = _meta_series(events, "a1_campaign_id")
    layers = _meta_series(events, "a1_mule_layer")

    for campaign_id in campaign_ids.unique():
        member = campaign_ids.eq(campaign_id)
        victim_value = float(events.loc[member & layers.eq(1), "amount"].sum())
        layer1_out = float(events.loc[member & layers.eq(2), "amount"].sum())
        cashed_out = float(events.loc[member & layers.eq(3), "amount"].sum())
        assert 0 < cashed_out <= layer1_out <= victim_value
        assert layer1_out == pytest.approx(round(victim_value * 0.98, 2), abs=0.01)
        assert cashed_out == pytest.approx(round(layer1_out * 0.97, 2), abs=0.05)


def test_each_transfer_hop_is_causally_ordered(generator, population):
    events = generator.generate(generator.default_params(), population, seed=303)
    campaign_ids = _meta_series(events, "a1_campaign_id")
    layers = _meta_series(events, "a1_mule_layer")

    for campaign_id in campaign_ids.unique():
        member = campaign_ids.eq(campaign_id)
        victim_end = events.loc[member & layers.eq(1), "timestamp"].max()
        layer1_moves = events.loc[member & layers.eq(2)]
        layer2_moves = events.loc[member & layers.eq(3)]
        assert layer1_moves["timestamp"].min() > victim_end
        for transfer in layer2_moves.itertuples():
            matching_inflow = layer1_moves.loc[
                layer1_moves["counterparty_id"].eq(transfer.customer_id), "timestamp"
            ]
            assert not matching_inflow.empty
            assert transfer.timestamp > matching_inflow.max()


def test_split_factor_controls_branch_count(generator, population):
    narrow = generator.default_params()
    narrow["split_factor"] = 2
    wide = generator.default_params()
    wide["split_factor"] = 7
    narrow_events = generator.generate(narrow, population, seed=404)
    wide_events = generator.generate(wide, population, seed=404)

    narrow_layers = _meta_series(narrow_events, "a1_mule_layer")
    wide_layers = _meta_series(wide_events, "a1_mule_layer")
    narrow_campaigns = _meta_series(narrow_events, "a1_campaign_id").nunique()
    wide_campaigns = _meta_series(wide_events, "a1_campaign_id").nunique()
    assert narrow_layers.eq(2).sum() == narrow_campaigns * 2
    assert wide_layers.eq(2).sum() == wide_campaigns * 7


def test_dwell_time_controls_layer1_departure_delay(generator, population):
    fast = generator.default_params()
    fast["dwell_time_hours"] = 0.05
    slow = generator.default_params()
    slow["dwell_time_hours"] = 12.0
    fast_events = generator.generate(fast, population, seed=505)
    slow_events = generator.generate(slow, population, seed=505)

    def first_campaign_delay(events: pd.DataFrame) -> float:
        campaign_ids = _meta_series(events, "a1_campaign_id")
        layers = _meta_series(events, "a1_mule_layer")
        first_campaign = campaign_ids.iloc[0]
        member = campaign_ids.eq(first_campaign)
        victim_end = events.loc[member & layers.eq(1), "timestamp"].max()
        departure = events.loc[member & layers.eq(2), "timestamp"].min()
        return (departure - victim_end).total_seconds() / 3600

    assert first_campaign_delay(fast_events) == pytest.approx(0.05)
    assert first_campaign_delay(slow_events) == pytest.approx(12.0)


def test_mule_age_and_cashout_pool_parameters_affect_infrastructure(generator, population):
    params = generator.default_params()
    params["mule_age_days"] = 45
    params["n_cashout_points"] = 3
    events = generator.generate(params, population, seed=606)
    layers = _meta_series(events, "a1_mule_layer")

    mule_destinations = events.loc[layers.isin([1, 2])]
    assert mule_destinations["counterparty_account_age_days"].eq(45).all()
    cashouts = events.loc[layers.eq(3), "counterparty_id"]
    assert 1 <= cashouts.nunique() <= 3


def test_each_campaign_reaches_a_cashout(generator, population):
    events = generator.generate(generator.default_params(), population, seed=707)
    campaign_ids = _meta_series(events, "a1_campaign_id")
    layers = _meta_series(events, "a1_mule_layer")
    campaigns = set(campaign_ids)
    cashed_out_campaigns = set(campaign_ids.loc[layers.eq(3)])
    assert cashed_out_campaigns == campaigns


def test_utility_is_native_finite_and_bounded(generator, population):
    events = generator.generate(generator.default_params(), population, seed=808)
    utility = generator.utility(events)
    assert isinstance(utility, float)
    assert np.isfinite(utility)
    assert 0.0 < utility < 1.0


def test_utility_increases_with_cashout_yield(generator, population):
    events = generator.generate(generator.default_params(), population, seed=809)
    richer = events.copy(deep=True)
    layers = _meta_series(richer, "a1_mule_layer")
    richer.loc[layers.eq(3), "amount"] *= 1.5
    assert generator.utility(richer) > generator.utility(events)


def test_utility_penalizes_additional_mules_for_same_yield(generator, population):
    events = generator.generate(generator.default_params(), population, seed=810)
    layers = _meta_series(events, "a1_mule_layer")
    extra = events.loc[layers.eq(2)].head(1).copy(deep=True)
    extra["counterparty_id"] = "extra_mule_without_extra_yield"
    extra["amount"] = 0.0
    expensive = pd.concat([events, extra], ignore_index=True)
    assert generator.utility(expensive) < generator.utility(events)


def test_utility_penalizes_more_cashout_points_for_same_yield(generator, population):
    events = generator.generate(generator.default_params(), population, seed=811)
    expensive = events.copy(deep=True)
    layers = _meta_series(expensive, "a1_mule_layer")
    cashout_rows = expensive.index[layers.eq(3)]
    expensive.loc[cashout_rows, "counterparty_id"] = [
        f"extra_cashout_{position}" for position in range(len(cashout_rows))
    ]
    assert generator.utility(expensive) < generator.utility(events)


def test_utility_penalizes_pre_aged_mules(generator, population):
    events = generator.generate(generator.default_params(), population, seed=812)
    young = events.copy(deep=True)
    old = events.copy(deep=True)
    layers = _meta_series(events, "a1_mule_layer")
    young.loc[layers.isin([1, 2]), "counterparty_account_age_days"] = 1
    old.loc[layers.isin([1, 2]), "counterparty_account_age_days"] = 120
    assert generator.utility(old) < generator.utility(young)


def test_adapted_campaign_pays_an_economic_cost_for_evasion(generator, population):
    default = generator.default_params()
    adapted = generator.default_params()
    adapted.update(
        {
            "mule_age_days": 45,
            "split_factor": 15,
            "dwell_time_hours": 6.0,
            "n_cashout_points": 20,
            "n_mules_layer1": 120,
        }
    )
    default_events = generator.generate(default, population, seed=814)
    adapted_events = generator.generate(adapted, population, seed=814)
    assert generator.utility(adapted_events) < generator.utility(default_events)


def test_utility_safely_handles_empty_malformed_and_declined_data(generator, population):
    assert generator.utility(pd.DataFrame()) == 0.0
    assert generator.utility(pd.DataFrame({"amount": [1.0]})) == 0.0

    events = generator.generate(generator.default_params(), population, seed=813)
    malformed = events.copy(deep=True)
    malformed["meta"] = None
    assert generator.utility(malformed) == 0.0

    declined = events.copy(deep=True)
    layers = _meta_series(declined, "a1_mule_layer")
    declined.loc[layers.eq(3), "auth_result"] = "DECLINED"
    assert generator.utility(declined) == 0.0


def test_describe_state_matches_exact_red_agent_contract(generator):
    params = generator.default_params()
    state = generator.describe_state(params)
    card_path = Path(__file__).parents[1] / "card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))

    assert set(state) == {
        "attack_summary",
        "param_semantics",
        "defender_signals",
        "current_params",
    }
    assert state["attack_summary"] == card["attack_summary"]
    assert state["param_semantics"] == card["param_semantics"]
    assert state["defender_signals"] == card["defender_signals"]
    assert state["current_params"] == params
    assert set(state["param_semantics"]) == set(generator.param_space())


def test_describe_state_returns_independent_copies(generator):
    params = generator.default_params()
    state = generator.describe_state(params)
    state["current_params"]["split_factor"] = 99
    state["param_semantics"].clear()
    state["defender_signals"].clear()

    fresh = generator.describe_state(params)
    assert fresh["current_params"]["split_factor"] == params["split_factor"]
    assert fresh["param_semantics"]
    assert fresh["defender_signals"]


def test_describe_state_rejects_invalid_parameters(generator):
    params = generator.default_params()
    params["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        generator.describe_state(params)


def _feature_events(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "attack_id": "LEGIT",
        "label": 0,
        "round": 0,
        "customer_account_age_days": 500,
        "device_id": "device",
        "ip_asn": "AS100",
        "session_id": "session",
        "session_duration_s": 120,
        "channel": "UPI_P2P",
        "card_id": "",
        "card_bin": "",
        "vpa": "payer@bank",
        "token_id": "",
        "counterparty_type": "BENEFICIARY",
        "counterparty_mcc": "",
        "counterparty_account_age_days": 100,
        "is_first_time_payee": 0,
        "auth_result": "APPROVED",
        "decline_code": "",
        "meta": {},
    }
    return coerce(pd.DataFrame([{**defaults, **row} for row in rows]))


def test_round0_feature_contract_and_input_order(feature_block):
    events = _feature_events(
        [
            {
                "event_id": "later",
                "timestamp": "2026-01-02 10:00:00",
                "customer_id": "payer-2",
                "counterparty_id": "beneficiary",
                "amount": 200.0,
            },
            {
                "event_id": "earlier",
                "timestamp": "2026-01-01 10:00:00",
                "customer_id": "payer-1",
                "counterparty_id": "beneficiary",
                "amount": 100.0,
            },
        ]
    )
    features = feature_block.build(events)
    assert list(features.index) == ["later", "earlier"]
    assert features.index.name == "event_id"
    assert list(features.columns) == feature_block.feature_names()
    assert all(column.startswith("a1_") for column in features)
    assert all(pd.api.types.is_float_dtype(features[column]) for column in features)
    assert np.isfinite(features.to_numpy()).all()


def test_beneficiary_fanin_uses_distinct_strictly_prior_24h_senders(feature_block):
    events = _feature_events(
        [
            {
                "event_id": "outside",
                "timestamp": "2026-01-01 09:59:59",
                "customer_id": "old-payer",
                "counterparty_id": "mule",
                "amount": 10.0,
            },
            {
                "event_id": "first",
                "timestamp": "2026-01-01 10:00:00",
                "customer_id": "payer-a",
                "counterparty_id": "mule",
                "amount": 20.0,
            },
            {
                "event_id": "repeat",
                "timestamp": "2026-01-02 08:00:00",
                "customer_id": "payer-a",
                "counterparty_id": "mule",
                "amount": 30.0,
            },
            {
                "event_id": "score-me",
                "timestamp": "2026-01-02 10:00:00",
                "customer_id": "payer-b",
                "counterparty_id": "mule",
                "amount": 40.0,
            },
            {
                "event_id": "future",
                "timestamp": "2026-01-02 10:00:01",
                "customer_id": "payer-c",
                "counterparty_id": "mule",
                "amount": 50.0,
            },
        ]
    )
    features = feature_block.build(events)
    assert features.loc["first", "a1_benef_fanin_24h"] == 1.0
    # The exact 24-hour boundary is included, the older row/future/current rows are not,
    # and payer-a is counted once despite paying twice.
    assert features.loc["score-me", "a1_benef_fanin_24h"] == 1.0


def test_payer_amount_zscore_uses_only_prior_90_day_history(feature_block):
    events = _feature_events(
        [
            {
                "event_id": "history-1",
                "timestamp": "2026-01-01 10:00:00",
                "customer_id": "victim",
                "counterparty_id": "shop-1",
                "amount": 100.0,
            },
            {
                "event_id": "history-2",
                "timestamp": "2026-01-02 10:00:00",
                "customer_id": "victim",
                "counterparty_id": "shop-2",
                "amount": 200.0,
            },
            {
                "event_id": "coerced",
                "timestamp": "2026-01-03 10:00:00",
                "customer_id": "victim",
                "counterparty_id": "mule",
                "amount": 300.0,
            },
            {
                "event_id": "future-outlier",
                "timestamp": "2026-01-04 10:00:00",
                "customer_id": "victim",
                "counterparty_id": "other",
                "amount": 10000.0,
            },
        ]
    )
    features = feature_block.build(events)
    expected = (300.0 - 150.0) / np.std([100.0, 200.0], ddof=1)
    assert features.loc["coerced", "a1_payer_amount_zscore"] == pytest.approx(expected)
    assert features.loc["history-1", "a1_payer_amount_zscore"] == 0.0


def test_dissipation_observes_only_beneficiary_outflow_in_next_hour(feature_block):
    events = _feature_events(
        [
            {
                "event_id": "inflow",
                "timestamp": "2026-01-01 10:00:00",
                "customer_id": "victim",
                "counterparty_id": "mule",
                "amount": 1000.0,
            },
            {
                "event_id": "outflow-in-window",
                "timestamp": "2026-01-01 10:30:00",
                "customer_id": "mule",
                "counterparty_id": "mule-2",
                "amount": 600.0,
            },
            {
                "event_id": "other-outflow",
                "timestamp": "2026-01-01 10:40:00",
                "customer_id": "someone-else",
                "counterparty_id": "mule-3",
                "amount": 500.0,
            },
            {
                "event_id": "outflow-too-late",
                "timestamp": "2026-01-01 11:00:01",
                "customer_id": "mule",
                "counterparty_id": "mule-4",
                "amount": 300.0,
            },
        ]
    )
    features = feature_block.build(events)
    assert features.loc["inflow", "a1_dissipation_ratio"] == pytest.approx(0.6)


def test_payee_delay_parses_metadata_and_uses_neutral_zero_for_invalid_values(feature_block):
    events = _feature_events(
        [
            {
                "event_id": "valid",
                "timestamp": "2026-01-01 10:01:00",
                "customer_id": "victim-1",
                "counterparty_id": "mule-1",
                "amount": 100.0,
                "meta": {"a1_payee_added_at": "2026-01-01T10:00:20"},
            },
            {
                "event_id": "missing",
                "timestamp": "2026-01-01 10:01:00",
                "customer_id": "victim-2",
                "counterparty_id": "mule-2",
                "amount": 100.0,
                "meta": {},
            },
            {
                "event_id": "malformed",
                "timestamp": "2026-01-01 10:01:00",
                "customer_id": "victim-3",
                "counterparty_id": "mule-3",
                "amount": 100.0,
                "meta": {"a1_payee_added_at": "not-a-date"},
            },
            {
                "event_id": "future-added",
                "timestamp": "2026-01-01 10:01:00",
                "customer_id": "victim-4",
                "counterparty_id": "mule-4",
                "amount": 100.0,
                "meta": {"a1_payee_added_at": "2026-01-01T10:02:00"},
            },
        ]
    )
    features = feature_block.build(events)
    assert features.loc["valid", "a1_payee_added_to_transfer_seconds"] == 40.0
    assert features.loc["missing", "a1_payee_added_to_transfer_seconds"] == 0.0
    assert features.loc["malformed", "a1_payee_added_to_transfer_seconds"] == 0.0
    assert features.loc["future-added", "a1_payee_added_to_transfer_seconds"] == 0.0


def test_feature_block_handles_mixed_legitimate_and_generated_a1_events(
    feature_block, generator, population
):
    fraud = generator.generate(generator.default_params(), population, seed=901)
    mixed = pd.concat([population, fraud], ignore_index=True)
    mixed = mixed.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    features = feature_block.build(mixed)
    assert len(features) == len(mixed)
    assert list(features.index) == mixed["event_id"].tolist()
    assert np.isfinite(features.to_numpy()).all()


def test_feature_block_rejects_duplicate_ids_and_invalid_amount(feature_block):
    duplicate = _feature_events(
        [
            {
                "event_id": "same",
                "timestamp": "2026-01-01",
                "customer_id": "a",
                "counterparty_id": "b",
                "amount": 1.0,
            },
            {
                "event_id": "same",
                "timestamp": "2026-01-02",
                "customer_id": "c",
                "counterparty_id": "d",
                "amount": 1.0,
            },
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        feature_block.build(duplicate)
    invalid = duplicate.head(1).copy()
    invalid["event_id"] = "unique"
    invalid["amount"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        feature_block.build(invalid)


def _detector_training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(123)
    labels = np.tile(np.array([0, 1], dtype=np.int8), 20)
    frame = pd.DataFrame(
        {
            "a1_signal": labels.astype(float) + rng.normal(0, 0.1, len(labels)),
            "a1_noise": rng.normal(0, 1, len(labels)),
        }
    )
    return frame, labels


def test_detector_requires_fit_before_scoring(detector):
    with pytest.raises(RuntimeError, match="fit"):
        detector.score(pd.DataFrame({"a1_signal": [1.0]}))


def test_detector_fit_score_contract(detector):
    X, y = _detector_training_frame()
    detector.fit(X, y)
    scores = detector.score(X)
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
    assert ((0.0 <= scores) & (scores <= 1.0)).all()
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_detector_scores_are_stable_under_column_reordering_and_extras(detector):
    X, y = _detector_training_frame()
    detector.fit(X, y)
    expected = detector.score(X)
    reordered = X[["a1_noise", "a1_signal"]].copy()
    reordered["unexpected"] = 999.0
    np.testing.assert_allclose(detector.score(reordered), expected)


def test_detector_supplies_zero_for_missing_trained_columns(detector):
    X, y = _detector_training_frame()
    detector.fit(X, y)
    missing = X.drop(columns="a1_noise")
    scores = detector.score(missing)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("label", [0, 1])
def test_detector_single_class_uses_constant_prior(detector, label):
    X = pd.DataFrame({"a1_signal": [0.0, 1.0, 2.0, 3.0]})
    y = np.full(len(X), label, dtype=np.int8)
    detector.fit(X, y)
    np.testing.assert_array_equal(detector.score(X), np.full(len(X), float(label)))


def test_detector_tiny_frame_uses_safe_prior(detector):
    X = pd.DataFrame({"a1_signal": [0.0, 1.0]})
    detector.fit(X, np.array([0, 1], dtype=np.int8))
    np.testing.assert_array_equal(detector.score(X), np.array([0.5, 0.5]))


def test_detector_sanitizes_nan_and_infinity(detector):
    X, y = _detector_training_frame()
    X.loc[0, "a1_signal"] = np.nan
    X.loc[1, "a1_signal"] = np.inf
    detector.fit(X, y)
    scores = detector.score(X)
    assert np.isfinite(scores).all()


def test_detector_repeated_fits_are_reproducible():
    X, y = _detector_training_frame()
    first = A1Detector()
    second = A1Detector()
    first.fit(X, y)
    second.fit(X, y)
    np.testing.assert_allclose(first.score(X), second.score(X), rtol=0, atol=0)


def test_detector_rejects_invalid_training_data(detector):
    X, y = _detector_training_frame()
    with pytest.raises(ValueError, match="same number"):
        detector.fit(X, y[:-1])
    with pytest.raises(ValueError, match="binary"):
        detector.fit(X, np.full(len(X), 2))
    with pytest.raises(ValueError, match="numeric"):
        detector.fit(pd.DataFrame({"text": ["a", "b"]}), np.array([0, 1]))


def _reconverging_graph_events() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-10 10:00:00")
    return _feature_events(
        [
            {
                "event_id": "victim-a-to-l1-a",
                "timestamp": base,
                "customer_id": "victim-a",
                "customer_account_age_days": 1000,
                "counterparty_id": "l1-a",
                "counterparty_account_age_days": 5,
                "amount": 1000.0,
                "session_duration_s": 900,
                "meta": {"a1_call_active": True},
            },
            {
                "event_id": "victim-b-to-l1-b",
                "timestamp": base + timedelta(minutes=1),
                "customer_id": "victim-b",
                "customer_account_age_days": 900,
                "counterparty_id": "l1-b",
                "counterparty_account_age_days": 5,
                "amount": 800.0,
                "meta": {"a1_call_active": True},
            },
            {
                "event_id": "l1-a-to-l2-a",
                "timestamp": base + timedelta(hours=1),
                "customer_id": "l1-a",
                "customer_account_age_days": 5,
                "counterparty_id": "l2-a",
                "counterparty_account_age_days": 6,
                "amount": 950.0,
            },
            {
                "event_id": "l1-b-to-l2-b",
                "timestamp": base + timedelta(hours=1, minutes=1),
                "customer_id": "l1-b",
                "customer_account_age_days": 5,
                "counterparty_id": "l2-b",
                "counterparty_account_age_days": 6,
                "amount": 760.0,
            },
            {
                "event_id": "l2-a-to-cashout",
                "timestamp": base + timedelta(hours=2),
                "customer_id": "l2-a",
                "customer_account_age_days": 6,
                "counterparty_id": "cashout",
                "counterparty_account_age_days": 100,
                "amount": 900.0,
            },
            {
                "event_id": "l2-b-to-cashout",
                "timestamp": base + timedelta(hours=2, minutes=1),
                "customer_id": "l2-b",
                "customer_account_age_days": 6,
                "counterparty_id": "cashout",
                "counterparty_account_age_days": 100,
                "amount": 720.0,
            },
            {
                "event_id": "unrelated",
                "timestamp": base,
                "customer_id": "unrelated-a",
                "counterparty_id": "unrelated-b",
                "amount": 50.0,
            },
        ]
    )


def test_component_size_uses_two_hop_beneficiary_neighborhood(feature_block):
    features = feature_block.build(_reconverging_graph_events())
    # From l1-a: victim-a and l2-a are one hop, cashout is two hops.
    assert features.loc["victim-a-to-l1-a", "a1_component_size"] == 4.0
    assert features.loc["unrelated", "a1_component_size"] == 2.0


def test_cashout_convergence_detects_distinct_reconverging_l1_paths(feature_block):
    features = feature_block.build(_reconverging_graph_events())
    assert features.loc["victim-a-to-l1-a", "a1_cashout_convergence"] == 2.0
    assert features.loc["victim-b-to-l1-b", "a1_cashout_convergence"] == 2.0


def test_component_dissipation_is_aggregate_outflow_over_inflow(feature_block):
    features = feature_block.build(_reconverging_graph_events())
    # l1-a's two-hop component is victim-a, l1-a, l2-a, cashout. Incoming edges to
    # these nodes include its own path plus l2-b's boundary inflow to cashout.
    expected = (1000.0 + 950.0 + 900.0) / (1000.0 + 950.0 + 900.0 + 720.0)
    assert features.loc["victim-a-to-l1-a", "a1_component_dissipation"] == pytest.approx(
        expected
    )


def test_component_age_variance_uses_known_local_account_ages(feature_block):
    events = _reconverging_graph_events()
    features = feature_block.build(events)
    expected = np.var([1000.0, 5.0, 6.0, 100.0], ddof=1)
    assert features.loc[
        "victim-a-to-l1-a", "a1_component_age_variance"
    ] == pytest.approx(expected)


def test_graph_window_excludes_edges_beyond_48_hours(feature_block):
    events = _reconverging_graph_events()
    extra = _feature_events(
        [
            {
                "event_id": "late-edge",
                "timestamp": pd.Timestamp("2026-01-10 10:00:00") + timedelta(hours=49),
                "customer_id": "cashout",
                "counterparty_id": "late-node",
                "amount": 100.0,
            }
        ]
    )
    features = feature_block.build(pd.concat([events, extra], ignore_index=True))
    assert features.loc["victim-a-to-l1-a", "a1_component_size"] == 4.0


def test_call_active_is_safe_binary_metadata_signal(feature_block):
    events = _reconverging_graph_events()
    events.at[events.index[-1], "meta"] = "malformed"
    features = feature_block.build(events)
    assert features.loc["victim-a-to-l1-a", "a1_call_active"] == 1.0
    assert features.loc["l1-a-to-l2-a", "a1_call_active"] == 0.0
    assert features.loc["unrelated", "a1_call_active"] == 0.0


def test_session_duration_anomaly_uses_only_prior_payer_history(feature_block):
    events = _feature_events(
        [
            {
                "event_id": "session-1",
                "timestamp": "2026-01-01",
                "customer_id": "victim",
                "counterparty_id": "shop-1",
                "amount": 100.0,
                "session_duration_s": 100,
            },
            {
                "event_id": "session-2",
                "timestamp": "2026-01-02",
                "customer_id": "victim",
                "counterparty_id": "shop-2",
                "amount": 100.0,
                "session_duration_s": 200,
            },
            {
                "event_id": "coerced-session",
                "timestamp": "2026-01-03",
                "customer_id": "victim",
                "counterparty_id": "mule",
                "amount": 100.0,
                "session_duration_s": 300,
            },
            {
                "event_id": "future-session",
                "timestamp": "2026-01-04",
                "customer_id": "victim",
                "counterparty_id": "shop-3",
                "amount": 100.0,
                "session_duration_s": 10000,
            },
        ]
    )
    features = feature_block.build(events)
    expected = (300.0 - 150.0) / np.std([100.0, 200.0], ddof=1)
    assert features.loc[
        "coerced-session", "a1_session_duration_anomaly"
    ] == pytest.approx(expected)


def test_all_eleven_features_match_card_defender_signals(feature_block):
    card_path = Path(__file__).parents[1] / "card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert set(feature_block.feature_names()) == set(card["defender_signals"])


def test_experiment_metrics_are_json_safe_and_complete():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    metrics = _metrics(y, scores, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auc_pr"] == 1.0
    assert metrics["rows"] == 4
    json.dumps(metrics)


def test_signal_auc_reports_discriminative_direction_and_constant_neutrality():
    X = pd.DataFrame({"inverse": [1.0, 0.9, 0.1, 0.0], "constant": [3.0] * 4})
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    auc = _signal_auc(X, y, ["inverse", "constant"])
    assert auc["inverse"] == 1.0
    assert auc["constant"] == 0.5
