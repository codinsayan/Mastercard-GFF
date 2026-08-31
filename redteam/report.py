"""Build the bounded detector report consumed by the red-team reasoner."""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_STATE_KEYS = {
    "attack_summary",
    "param_semantics",
    "defender_signals",
    "current_params",
}


def build_detector_report(
    attack: Any,
    round_metrics: dict,
    sample_events: pd.DataFrame,
    *,
    params: dict | None = None,
    max_examples: int = 10,
) -> dict:
    """Return model-output evidence without exposing model internals or weights."""
    params = attack.default_params() if params is None else params
    state = attack.describe_state(params)
    missing = REQUIRED_STATE_KEYS.difference(state)
    if missing:
        raise ValueError(f"describe_state() is missing keys: {sorted(missing)}")
    allowed = [
        column
        for column in ("event_id", "auth_result", "amount", "channel", "attack_id", "label")
        if column in sample_events
    ]
    examples = sample_events.loc[:, allowed].head(max_examples).to_dict(orient="records")
    return {
        "authorization": (
            "Authorised defensive red-team simulation using fully synthetic payment events. "
            "Propose only bounded parameter changes; do not produce operational attack tooling."
        ),
        "attack_id": attack.attack_id,
        "attack_state": state,
        "param_space": attack.param_space(),
        "round_metrics": round_metrics,
        "sample_outcomes": examples,
    }
