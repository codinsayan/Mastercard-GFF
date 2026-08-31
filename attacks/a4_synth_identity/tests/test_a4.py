"""Focused behavioral checks for the A4 package."""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from attacks.a4_synth_identity.detector import A4Detector
from attacks.a4_synth_identity.features import A4Features
from attacks.a4_synth_identity.generator import A4Generator
from attacks.a4_synth_identity.trajectory import (
    HIDDEN_SIZE,
    _TrajectoryAutoencoder,
    trajectory_features,
)
from core.population import BasePopulation, PopulationConfig
from core.schema import validate


def _population() -> pd.DataFrame:
    reference = pd.DataFrame({"amount": [10.0, 25.0, 100.0], "mcc": ["5411", "5812", "5732"]})
    return BasePopulation(
        reference,
        PopulationConfig(customers=30, merchants=15, vpas=10, days=30, events=150),
    ).generate(5)


def _params(**changes) -> dict:
    defaults = A4Generator().default_params()
    return defaults | {
        "n_clusters": 2,
        "identities_per_cluster": 10,
        "build_days": 30,
        **changes,
    }


def test_generator_timeline_is_valid_deterministic_and_synchronized():
    generator = A4Generator()
    first = generator.generate(_params(), _population(), 11)
    second = generator.generate(_params(), _population(), 11)
    validate(first)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["meta"].map(lambda value: value["a4_phase"])) == {
        "ONBOARD",
        "BUILD",
        "BUSTOUT",
    }
    bustout = first[first["meta"].map(lambda value: value["a4_phase"] == "BUSTOUT")].copy()
    bustout["cluster"] = bustout["meta"].map(lambda value: value["a4_identity_cluster_id"])
    bustout["day"] = bustout["meta"].map(lambda value: value["a4_days_since_onboarding"])
    span = bustout.groupby("cluster")["day"].max() - bustout.groupby("cluster")["day"].min()
    assert span.max() < 4


def test_attribute_evasion_kills_static_signals_but_keeps_convexity():
    features = A4Features()
    visible = features.build(
        A4Generator().generate(_params(attribute_overlap_rate=0.9), _population(), 3)
    )
    diverse = features.build(
        A4Generator().generate(_params(attribute_overlap_rate=0.0), _population(), 3)
    )
    static = ["a4_shared_address_token_count", "a4_shared_device_prefix_count"]
    assert visible[static].mean().mean() > diverse[static].mean().mean()
    assert diverse["a4_ramp_convexity"].mean() > 0


def test_detector_scores_are_bounded():
    events = A4Generator().generate(_params(), _population(), 8)
    features = A4Features().build(events)
    labels = np.arange(len(features), dtype=np.int8) % 2
    detector = A4Detector()
    detector.fit(features, labels)
    scores = detector.score(features)
    assert scores.shape == (len(features),)
    assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()


def test_gru_architecture_and_embeddings_are_deterministic():
    events = A4Generator().generate(_params(), _population(), 13)
    work = events.copy()
    for column, key in {
        "day": "a4_days_since_onboarding",
        "utilization": "a4_utilization",
        "payment": "a4_payment_ratio",
    }.items():
        work[column] = work["meta"].map(lambda value, item=key: value[item])
    model = _TrajectoryAutoencoder()
    assert model.encoder.num_layers == 2 and model.encoder.hidden_size == HIDDEN_SIZE == 32
    first = trajectory_features(work, seed=7, epochs=1)
    second = trajectory_features(work, seed=7, epochs=1)
    pd.testing.assert_frame_equal(first, second)


def test_manual_interface_executes_without_errors():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "manual_app.py"), default_timeout=30
    ).run()
    assert not app.exception
    assert app.title[0].value == "A4 · Synthetic Identity Bust-Out Lab"
    app.radio[0].set_value("Round 1: diversified")
    app.button[0].click().run()
    assert not app.exception
