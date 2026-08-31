from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attacks.a1_upi_mule.generator import A1Generator
from core.population import BasePopulation, PopulationConfig
from core.schema import validate


@pytest.fixture
def generator() -> A1Generator:
    return A1Generator()


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
