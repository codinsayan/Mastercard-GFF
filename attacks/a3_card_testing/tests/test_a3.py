from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attacks.a3_card_testing.generator import A3Generator
from core.population import BasePopulation, PopulationConfig
from core.schema import validate


@pytest.fixture(scope="module")
def population() -> pd.DataFrame:
    reference = pd.DataFrame(
        {
            "amount": [12.5, 48.0, 109.0, 799.0],
            "mcc": ["5411", "5812", "5732", "5999"],
        }
    )
    return BasePopulation(
        reference,
        PopulationConfig(customers=50, merchants=25, vpas=20, days=7, events=200),
    ).generate(seed=101)


def _fraud(events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[events["attack_id"].eq("A3")].copy()


def _params(**updates) -> dict:
    params = A3Generator().default_params()
    params.update(updates)
    return params


def test_configuration_and_state_match_frozen_card():
    generator = A3Generator()
    params = generator.default_params()
    assert generator.attack_id == "A3"
    assert set(params) == set(generator.param_space())
    assert generator.param_space()["decline_rate"] == (0.5, 0.95)
    state = generator.describe_state(params)
    assert set(state) == {
        "attack_summary",
        "param_semantics",
        "defender_signals",
        "current_params",
    }
    assert set(state["param_semantics"]) == set(params)
    assert len(state["defender_signals"]) == 9


def test_generation_is_schema_valid_deterministic_and_preserves_population(population):
    generator = A3Generator()
    params = generator.default_params()
    before = population.copy(deep=True)
    before["meta"] = before["meta"].map(dict)

    first = generator.generate(params, population, seed=8675309)
    second = generator.generate(params, population, seed=8675309)

    validate(first)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_frame_equal(population, before, check_exact=True)
    preserved = first.set_index("event_id").loc[population["event_id"]].reset_index()
    pd.testing.assert_frame_equal(
        preserved.loc[:, population.columns], population.reset_index(drop=True), check_exact=True
    )


def test_primary_campaign_has_balanced_bins_device_rotation_and_bounded_amounts(population):
    generator = A3Generator()
    params = _params(
        cards_per_hour=17,
        n_bins=4,
        cards_per_device=3,
        amount_min=40,
        amount_max=45,
        merchant_friction_mix=0.0,
    )
    fraud = _fraud(generator.generate(params, population, seed=22))
    primary = fraud.sort_values(["timestamp", "event_id"]).drop_duplicates("card_id")

    assert primary["card_id"].nunique() == 17
    assert len(primary) == 17
    assert primary["amount"].between(40, 45, inclusive="both").all()
    np.testing.assert_allclose(primary["amount"], primary["amount"].round(2))
    bin_counts = primary["card_bin"].value_counts()
    assert len(bin_counts) == 4
    assert bin_counts.max() - bin_counts.min() <= 1
    expected_devices = np.arange(17) // 3
    observed_devices = pd.factorize(primary["device_id"], sort=False)[0]
    np.testing.assert_array_equal(observed_devices, expected_devices)
    start = population["timestamp"].max() + pd.Timedelta(seconds=1)
    assert primary["timestamp"].min() == start
    assert primary["timestamp"].max() < start + pd.Timedelta(hours=1)


def test_campaign_uses_requested_pool_and_synthesizes_only_shortfall(population):
    generator = A3Generator()
    legitimate_merchants = set(
        population.loc[
            population["counterparty_type"].eq("MERCHANT"), "counterparty_id"
        ]
    )
    requested = len(legitimate_merchants) + 3
    params = _params(cards_per_hour=60, n_merchants=requested, merchant_friction_mix=0.0)
    fraud = _fraud(generator.generate(params, population, seed=31))
    used = set(fraud["counterparty_id"])

    assert used.issubset(legitimate_merchants | {f"A3-MERCHANT-{i:04d}" for i in range(3)})
    assert all(merchant in legitimate_merchants for merchant in used if not merchant.startswith("A3-"))


def test_high_friction_attempts_get_one_low_friction_retry(population):
    generator = A3Generator()
    params = _params(
        cards_per_hour=60,
        n_merchants=3,
        merchant_friction_mix=0.8,
        decline_rate=0.5,
    )
    fraud = _fraud(generator.generate(params, population, seed=44)).sort_values(
        ["timestamp", "event_id"]
    )
    grouped = fraud.groupby("card_id", sort=False)

    assert grouped.size().max() == 2
    assert grouped.size().min() == 1
    assert (grouped.size() == 2).any()
    for _, attempts in grouped:
        attempts = attempts.sort_values(["timestamp", "event_id"])
        primary = attempts.iloc[0]
        if primary["meta"]["a3_merchant_friction_tier"] == "HIGH":
            assert len(attempts) == 2
            retry = attempts.iloc[1]
            assert primary["auth_result"] == "DECLINED"
            assert primary["decline_code"] == "STRONG_AUTH_REQUIRED"
            assert retry["timestamp"] - primary["timestamp"] == pd.Timedelta(seconds=60)
            assert retry["meta"]["a3_merchant_friction_tier"] == "LOW"
            assert retry["counterparty_id"] != primary["counterparty_id"]
            assert retry["amount"] != primary["amount"]
            for column in ("card_id", "card_bin", "device_id", "session_id"):
                assert retry[column] == primary[column]
            for key in ("a3_dump_id", "a3_bot_session_id"):
                assert retry["meta"][key] == primary["meta"][key]
        else:
            assert len(attempts) == 1


def test_fraud_rows_have_required_identity_and_metadata(population):
    fraud = _fraud(A3Generator().generate(_params(), population, seed=55))
    required_meta = {
        "a3_bin_prefix",
        "a3_dump_id",
        "a3_bot_session_id",
        "a3_merchant_friction_tier",
        "a3_attempt_index",
    }
    assert fraud["attack_id"].eq("A3").all()
    assert fraud["label"].eq(1).all()
    assert fraud["channel"].eq("CNP").all()
    assert fraud["counterparty_type"].eq("MERCHANT").all()
    assert fraud["round"].eq(population["round"].iloc[0]).all()
    assert fraud["meta"].map(lambda value: isinstance(value, dict)).all()
    assert fraud["meta"].map(lambda value: set(value) == required_meta).all()
    assert all(
        meta["a3_bin_prefix"] == card_bin
        for meta, card_bin in zip(fraud["meta"], fraud["card_bin"], strict=True)
    )
    assert sorted(meta["a3_attempt_index"] for meta in fraud["meta"]) == list(range(len(fraud)))


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        ({"amount_min": 100, "amount_max": 50}, ValueError, "amount_min"),
        ({"cards_per_hour": 3.5}, TypeError, "integer"),
        ({"decline_rate": 0.2}, ValueError, "within"),
        ({"unknown": 1}, ValueError, "unknown"),
    ],
)
def test_invalid_parameters_are_rejected(population, updates, exception, message):
    with pytest.raises(exception, match=message):
        A3Generator().generate(_params(**updates), population, seed=1)


def test_mixed_round_population_is_rejected(population):
    mixed = population.copy()
    mixed.loc[mixed.index[0], "round"] = 1
    with pytest.raises(ValueError, match="exactly one round"):
        A3Generator().generate(_params(), mixed, seed=1)


def test_utility_counts_distinct_approved_a3_cards_and_clips():
    events = pd.DataFrame(
        {
            "attack_id": ["A3", "A3", "A3", "LEGIT"],
            "label": [1, 1, 1, 0],
            "auth_result": ["APPROVED", "APPROVED", "DECLINED", "APPROVED"],
            "card_id": ["c1", "c1", "c2", "c3"],
        }
    )
    assert A3Generator().utility(events) == pytest.approx(1 / 60)
