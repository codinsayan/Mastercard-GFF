from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attacks.a2_agent_inject.assets import load_catalogue, load_embedding_cache
from attacks.a2_agent_inject.features import FEATURE_NAMES, Features
from attacks.a2_agent_inject.generator import Generator
from core.population import BasePopulation, PopulationConfig
from core.schema import validate


@pytest.fixture(scope="module")
def population() -> pd.DataFrame:
    reference = pd.DataFrame(
        {
            "amount": [90.0, 450.0, 1800.0, 7200.0, 72000.0],
            "mcc": ["5411", "5812", "5941", "5732", "5946"],
        }
    )
    config = PopulationConfig(customers=100, merchants=40, vpas=60, days=7, events=250)
    return BasePopulation(reference, config).generate(seed=99)


@pytest.fixture(scope="module")
def generated(population: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = Generator()
    round0 = generator.generate(generator.default_params(), population, seed=1234)
    evolved_params = {**generator.default_params(), "injection_semantic_distance": 0.25}
    evolved = generator.generate(evolved_params, population, seed=1234)
    return round0, evolved


def test_committed_assets_are_complete() -> None:
    catalogue = load_catalogue()
    cache = load_embedding_cache()
    assert len(catalogue) == 15
    assert sum(len(products) for products in catalogue.values()) == 150
    assert len(cache["item_vectors"]) == 150
    assert len(cache["intent_vectors"]) == 15
    assert cache["dimension"] == 384


def test_generator_produces_valid_balanced_sessions(
    generated: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    round0, _ = generated
    validate(round0)
    assert len(round0) == 600
    assert round0["label"].value_counts().to_dict() == {0: 300, 1: 300}
    assert set(round0["channel"]) == {"AGENTIC"}
    assert set(round0["auth_result"]) == {"APPROVED"}
    for amount, meta in zip(round0["amount"], round0["meta"], strict=True):
        assert amount <= meta["a2_spend_cap"] + 0.01


def test_adaptation_reduces_realized_semantic_distance(
    generated: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    round0, evolved = generated

    def distances(frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[frame["label"].eq(1), "meta"].map(
            lambda meta: np.mean(meta["a2_realized_semantic_distances"])
        )
        return values.to_numpy(dtype=float)

    assert distances(evolved).mean() < distances(round0).mean() - 0.15


def test_features_show_evasion_and_durable_price_signal(
    generated: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    round0, evolved = generated
    feature_block = Features()
    round0_features = feature_block.build(round0)
    evolved_features = feature_block.build(evolved)
    assert tuple(round0_features.columns) == FEATURE_NAMES
    assert np.isfinite(round0_features.to_numpy()).all()

    fraud = round0["label"].to_numpy(dtype=bool)
    evolved_fraud = evolved["label"].to_numpy(dtype=bool)
    round0_cosine = round0_features["a2_intent_cart_cosine"].to_numpy()
    evolved_cosine = evolved_features["a2_intent_cart_cosine"].to_numpy()
    assert round0_cosine[fraud].mean() < round0_cosine[~fraud].mean()
    assert evolved_cosine[evolved_fraud].mean() > round0_cosine[fraud].mean()
    assert (
        evolved_features.loc[evolved_fraud, "a2_price_zscore_in_category"].mean()
        > evolved_features.loc[~evolved_fraud, "a2_price_zscore_in_category"].mean() + 3.0
    )


def test_lower_price_multiplier_reduces_attacker_utility(population: pd.DataFrame) -> None:
    generator = Generator()
    high = {**generator.default_params(), "injection_semantic_distance": 0.25}
    low = {
        **high,
        "price_multiplier": 3.0,
        "cart_mutated_post_load": 0,
    }
    high_events = generator.generate(high, population, seed=777)
    low_events = generator.generate(low, population, seed=777)
    assert generator.utility(low_events) < generator.utility(high_events) * 0.1
