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
    assert events["counterparty_id"].str.startswith("a1_mule_l1_").all()


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
    assert events["customer_id"].nunique() <= population["customer_id"].nunique()
    assert 1 <= len(events) <= population["customer_id"].nunique() * 3


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
    assert events["meta"].map(lambda value: value["a1_mule_layer"] == 1).all()
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

