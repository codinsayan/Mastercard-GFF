"""Focused behavioral checks for the A4 package."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_build_window_excludes_every_bustout_row():
    """The build window must stop at an account's first spike, not at a fixed threshold.

    A draw-down opens below 0.85 before crossing it. When the window was defined as
    ``utilization < 0.85`` that opening row -- carrying a zero repayment -- fell back
    inside the account's build statistics, so a4_payment_ratio_regularity scored a
    missed payment rather than repayment regularity.
    """
    events = A4Generator().generate(_params(), _population(), 17)
    day = events["meta"].map(lambda value: value["a4_days_since_onboarding"])
    utilization = events["meta"].map(lambda value: value["a4_utilization"])
    phase = events["meta"].map(lambda value: value["a4_phase"])
    frame = pd.DataFrame(
        {"customer": events["customer_id"], "day": day, "u": utilization, "phase": phase}
    )

    leaked = 0
    for _, group in frame.groupby("customer", sort=False):
        spike = group.loc[group["u"] >= 0.85, "day"]
        if not len(spike):
            continue
        in_build = group[group["day"] < spike.min()]
        leaked += int(in_build["phase"].eq("BUSTOUT").sum())
    assert leaked == 0

    # and the feature it corrupted now points the way its description claims:
    # synthetic accounts repay more consistently than organic ones.
    features = A4Features(trajectory_epochs=0).build(events)
    assert features["a4_payment_ratio_regularity"].gt(0.3).all()


def test_demo_app_is_replay_only_and_every_screen_renders():
    """The judging demo must replay artifacts: no training, no generation, no network.

    Enforced structurally rather than by convention -- importing the generator, the
    detector or torch is what would let a screen start computing on stage.
    """
    app = Path(__file__).resolve().parents[1] / "demo_app.py"
    tree = ast.parse(app.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    heavy = {name for name in imported if name.split(".")[0] in {"torch", "sklearn"}}
    assert not heavy, f"demo_app.py must not import {heavy}"
    attack_modules = {
        name
        for name in imported
        if name.startswith("attacks.") and not name.endswith("build_demo_assets")
    }
    assert not attack_modules, f"demo_app.py must not import {attack_modules}"
    for screen in (
        "1 · The innocent transaction",
        "3 · Signal mortality",
        "4 · The attacker's words",
        "5 · Scoreboard & caveats",
    ):
        test = AppTest.from_file(str(app), default_timeout=120).run()
        assert not test.exception
        test.sidebar.radio[0].set_value(screen).run()
        assert not test.exception, f"{screen} raised"


def test_demo_assets_are_small_and_complete():
    assets_path = Path(__file__).resolve().parents[3] / "results" / "a4" / "demo_assets.json"
    if not assets_path.exists():
        pytest.skip("demo assets not built; run build_demo_assets.py")
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    assert set(assets) >= {"hook", "cohort", "rounds", "reference"}
    assert {"0", "1", "2"} == set(assets["rounds"])
    # committed to git, so it has to stay small
    assert assets_path.stat().st_size < 1_000_000
    # the hook table must show ordinary build rows and the draw-down that follows
    phases = {row["phase"] for row in assets["hook"]["rows"]}
    assert "BUILD" in phases and "BUSTOUT" in phases
