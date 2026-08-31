"""Shared binary fraud metrics, per-feature AUC, and leave-one-attack-out evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return the frozen fraud metrics at threshold 0.5 and operating points."""
    labels = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=float)
    if labels.shape != values.shape or len(np.unique(labels)) != 2:
        raise ValueError("metrics require equal-length scores and both binary classes")
    predictions = values >= 0.5
    fpr, tpr, _ = roc_curve(labels, values)
    recall_at_target = float(tpr[fpr <= 0.001].max(initial=0))
    target_positions = np.flatnonzero(tpr >= 0.85)
    fpr_at_recall = float(fpr[target_positions].min()) if len(target_positions) else 1.0
    return {
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "auc_pr": float(average_precision_score(labels, values)),
        "recall_at_fpr_0.001": recall_at_target,
        "fpr_at_recall_0.85": fpr_at_recall,
    }


def per_feature_auc(features: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """Return direction-independent univariate AUC for each finite feature."""
    y = np.asarray(labels, dtype=np.int8)
    if len(features) != len(y) or len(np.unique(y)) != 2:
        raise ValueError("feature AUC requires aligned rows and both binary classes")
    result = {}
    for name in features:
        values = pd.to_numeric(features[name], errors="coerce").fillna(0).to_numpy()
        auc = float(roc_auc_score(y, values))
        result[name] = max(auc, 1 - auc)
    return result


def leave_one_attack_out(
    features: pd.DataFrame,
    labels: np.ndarray,
    attack_ids: np.ndarray,
    estimator_factory: Callable[[], ClassifierMixin],
) -> dict[str, dict[str, float]]:
    """Train without each attack and score that unseen attack against held-out legitimate rows."""
    y = np.asarray(labels, dtype=np.int8)
    attacks = np.asarray(attack_ids)
    if not (len(features) == len(y) == len(attacks)):
        raise ValueError("LOAO inputs must have equal lengths")
    legitimate = np.flatnonzero(attacks == "LEGIT")
    midpoint = len(legitimate) // 2
    if midpoint == 0:
        raise ValueError("LOAO requires at least two legitimate examples")
    output = {}
    for held_out in sorted(set(attacks).difference({"LEGIT"})):
        train = (attacks != held_out) & ~np.isin(np.arange(len(attacks)), legitimate[midpoint:])
        test = (attacks == held_out) | np.isin(np.arange(len(attacks)), legitimate[midpoint:])
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            continue
        estimator = estimator_factory()
        estimator.fit(features.loc[train], y[train])
        output[held_out] = binary_metrics(
            y[test], estimator.predict_proba(features.loc[test])[:, 1]
        )
    return output
