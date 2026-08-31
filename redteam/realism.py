"""Realism and attacker-utility budget gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BudgetViolation(ValueError):
    """Raised when proposed attack events exceed a frozen budget."""


@dataclass(frozen=True)
class BudgetResult:
    accepted: bool
    utility: float
    max_ks: float
    reason: str


def ks_distance(first: pd.Series, second: pd.Series) -> float:
    """Return the exact two-sample Kolmogorov-Smirnov distance."""
    left = pd.to_numeric(first, errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(second, errors="coerce").dropna().to_numpy(dtype=float)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if not len(left) or not len(right):
        raise ValueError("KS inputs must each contain a finite numeric value")
    left.sort()
    right.sort()
    points = np.concatenate((left, right))
    left_cdf = np.searchsorted(left, points, side="right") / len(left)
    right_cdf = np.searchsorted(right, points, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def check_budgets(
    attack: Any,
    events: pd.DataFrame,
    reference_events: pd.DataFrame | None = None,
    *,
    min_utility: float = 0.1,
    max_ks: float = 0.1,
    realism_columns: tuple[str, ...] = ("amount",),
) -> BudgetResult:
    """Check attacker utility and optional marginal realism constraints."""
    utility = attack.utility(events)
    if isinstance(utility, bool) or not isinstance(utility, (int, float)):
        raise TypeError("utility() must return a float")
    utility = float(utility)
    if not math.isfinite(utility) or not 0 <= utility <= 1:
        raise ValueError("utility() must return a finite value in [0, 1]")

    distances = []
    if reference_events is not None:
        for column in realism_columns:
            if column not in events or column not in reference_events:
                raise ValueError(f"realism column is missing: {column}")
            distances.append(ks_distance(events[column], reference_events[column]))
    observed_ks = max(distances, default=0.0)
    reasons = []
    if utility < min_utility:
        reasons.append(f"utility {utility:.3f} is below {min_utility:.3f}")
    if observed_ks > max_ks:
        reasons.append(f"KS distance {observed_ks:.3f} exceeds {max_ks:.3f}")
    return BudgetResult(not reasons, utility, observed_ks, "; ".join(reasons))


def enforce(
    attack: Any,
    events: pd.DataFrame,
    reference_events: pd.DataFrame | None = None,
    **limits: Any,
) -> pd.DataFrame:
    """Return accepted events or raise ``BudgetViolation`` with the failed budget."""
    result = check_budgets(attack, events, reference_events, **limits)
    if not result.accepted:
        raise BudgetViolation(result.reason)
    return events
