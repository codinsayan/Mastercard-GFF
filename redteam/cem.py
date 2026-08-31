"""Small deterministic Cross-Entropy Method parameter refiner."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def refine(
    proposal: dict[str, Any],
    attack: Any,
    objective: Callable[[dict[str, float]], float] | None = None,
    *,
    seed: int = 42,
    samples: int = 64,
    iterations: int = 5,
    elite_fraction: float = 0.2,
) -> dict[str, float]:
    """Refine a proposal inside ``attack.param_space()``; higher scores are better.

    Without an objective, return the bounded proposal as a safe integration-stage seed.
    """
    if samples < 2 or iterations < 1 or not 0 < elite_fraction <= 1:
        raise ValueError("invalid CEM search settings")
    space = attack.param_space()
    defaults = attack.default_params()
    names = list(space)
    low = np.array([space[name][0] for name in names], dtype=float)
    high = np.array([space[name][1] for name in names], dtype=float)
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)) or np.any(low > high):
        raise ValueError("parameter bounds must be finite and ordered")

    proposed = proposal.get("proposed_params", proposal)
    center = np.array(
        [float(proposed.get(name, defaults[name])) for name in names], dtype=float
    )
    center = np.clip(center, low, high)
    integer_params = {
        name
        for name in names
        if isinstance(defaults[name], int) and not isinstance(defaults[name], bool)
    }

    def as_params(values: np.ndarray) -> dict[str, float]:
        params = {}
        for name, value in zip(names, values, strict=True):
            params[name] = round(value) if name in integer_params else float(value)
        return params

    if objective is None or not names:
        return as_params(center)

    rng = np.random.default_rng(seed)
    spread = np.maximum((high - low) / 4, 1e-9)
    best = center.copy()
    best_score = float(objective(as_params(best)))
    if not math.isfinite(best_score):
        best_score = -math.inf
    elite_count = max(1, math.ceil(samples * elite_fraction))

    for _ in range(iterations):
        candidates = np.clip(
            rng.normal(center, spread, size=(samples, len(names))), low, high
        )
        candidates[0] = center
        scores = np.array([float(objective(as_params(row))) for row in candidates])
        scores[~np.isfinite(scores)] = -math.inf
        winner = int(np.argmax(scores))
        if scores[winner] > best_score:
            best, best_score = candidates[winner].copy(), scores[winner]
        elite = candidates[np.argsort(scores)[-elite_count:]]
        center = elite.mean(axis=0)
        spread = np.maximum(elite.std(axis=0), (high - low) * 0.005)

    return as_params(np.clip(best, low, high))
