"""Frozen integration checks for independently developed attack packages.

Each completed package exposes ``GENERATOR``, ``FEATURE_BLOCK``, and ``DETECTOR``
from its ``__init__.py``. Incomplete packages skip locally; the merge gate requires
all four cases to run without skips.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.population import BasePopulation, PopulationConfig
from core.schema import validate

PACKAGES = {
    "a1": "attacks.a1_upi_mule",
    "a2": "attacks.a2_agent_inject",
    "a3": "attacks.a3_card_testing",
    "a4": "attacks.a4_synth_identity",
}
STATE_KEYS = {"attack_summary", "param_semantics", "defender_signals", "current_params"}


def _components(key: str):
    package_name = PACKAGES[key]
    module = importlib.import_module(package_name)
    missing = [name for name in ("GENERATOR", "FEATURE_BLOCK", "DETECTOR") if not hasattr(module, name)]
    if missing:
        pytest.skip(f"{key.upper()} not implemented yet; missing exports: {missing}")
    card = json.loads((Path(module.__file__).parent / "card.json").read_text(encoding="utf-8"))
    return module.GENERATOR, module.FEATURE_BLOCK, module.DETECTOR, card


@pytest.mark.parametrize("attack_key", PACKAGES, ids=PACKAGES)
def test_attack_contract(attack_key):
    generator, feature_block, detector, card = _components(attack_key)
    attack_id = attack_key.upper()
    params = generator.default_params()
    reference = pd.DataFrame({
        "amount": [12.5, 48.0, 109.0, 799.0],
        "mcc": ["5411", "5812", "5732", "5999"],
    })
    population = BasePopulation(reference, PopulationConfig(events=200)).generate(seed=101)
    first = generator.generate(params, population=population, seed=8675309)
    second = generator.generate(params, population=population, seed=8675309)

    validate(first)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert set(first["label"].unique()).issubset({0, 1})
    assert (first.loc[first["label"].eq(1), "attack_id"] == attack_id).all()

    features = feature_block.build(first)
    assert features.index.name == "event_id"
    assert features.columns.size > 0
    assert all(column.startswith(f"{attack_key}_") for column in features)

    space = generator.param_space()
    assert set(space) == set(params) == set(card["default_params"]) == set(card["param_space"])
    assert set(card["param_semantics"]) == set(space)
    assert set(card["defender_signals"]) == set(features.columns)

    state = generator.describe_state(params)
    assert set(state) == STATE_KEYS
    assert state["current_params"] == params
    assert set(state["param_semantics"]) == set(space)
    assert set(state["defender_signals"]) == set(features.columns)

    utility = generator.utility(first)
    assert isinstance(utility, float) and 0.0 <= utility <= 1.0

    X = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.arange(len(X), dtype=np.int8) % 2
    detector.fit(X, y)
    scores = detector.score(X)
    assert isinstance(scores, np.ndarray)
    assert np.issubdtype(scores.dtype, np.floating)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all() and ((0.0 <= scores) & (scores <= 1.0)).all()
