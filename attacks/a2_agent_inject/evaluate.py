"""Reproduce the A2 round-0, evasion, and recovery measurements."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from core.population import PopulationConfig

from .detector import Detector
from .features import DURABLE_FEATURES, FEATURE_NAMES, ROUND0_FEATURES, Features
from .generator import Generator
from .reference import build_sparkov_population


def _population() -> pd.DataFrame:
    config = PopulationConfig(customers=800, merchants=200, vpas=300, days=30, events=1500)
    return build_sparkov_population(config, seed=2026)


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def _threshold_for_recall(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    positive_scores = scores[y == 1]
    if not len(positive_scores):
        raise ValueError("at least one positive example is required")
    return float(np.quantile(positive_scores, 1.0 - target, method="lower"))


def _signal_auc(y: np.ndarray, values: pd.Series) -> float:
    observed = float(roc_auc_score(y, values.to_numpy(dtype=float)))
    return max(observed, 1.0 - observed)


def run() -> dict[str, object]:
    generator = Generator()
    feature_block = Features()
    population = _population()
    round0_params = generator.default_params()
    evolved_params = {**round0_params, "injection_semantic_distance": 0.25}
    low_profit_params = {
        **evolved_params,
        "price_multiplier": 3.0,
        "cart_mutated_post_load": 0,
    }

    round0_train = generator.generate(round0_params, population, seed=1101)
    round0_test = generator.generate(round0_params, population, seed=1102)
    evolved_train = generator.generate(evolved_params, population, seed=1201)
    evolved_test = generator.generate(evolved_params, population, seed=1202)
    low_profit = generator.generate(low_profit_params, population, seed=1301)
    x_round0_train = feature_block.build(round0_train)
    x_round0_test = feature_block.build(round0_test)
    x_evolved_train = feature_block.build(evolved_train)
    x_evolved_test = feature_block.build(evolved_test)
    y_round0_train = round0_train["label"].to_numpy(dtype=np.int8)
    y_round0_test = round0_test["label"].to_numpy(dtype=np.int8)
    y_evolved_train = evolved_train["label"].to_numpy(dtype=np.int8)
    y_evolved_test = evolved_test["label"].to_numpy(dtype=np.int8)

    fragile_detector = Detector(ROUND0_FEATURES, random_state=42)
    fragile_detector.fit(x_round0_train, y_round0_train)
    fragile_train_scores = fragile_detector.score(x_round0_train)
    fragile_threshold = _threshold_for_recall(
        y_round0_train,
        fragile_train_scores,
        target=0.94,
    )
    round0_scores = fragile_detector.score(x_round0_test)
    evolved_scores = fragile_detector.score(x_evolved_test)

    durable_detector = Detector(DURABLE_FEATURES, random_state=42)
    durable_detector.fit(x_evolved_train, y_evolved_train)
    durable_train_scores = durable_detector.score(x_evolved_train)
    durable_threshold = _threshold_for_recall(
        y_evolved_train,
        durable_train_scores,
        target=0.9,
    )
    recovered_scores = durable_detector.score(x_evolved_test)

    round0_signal_auc = {
        feature: _signal_auc(y_round0_test, x_round0_test[feature])
        for feature in FEATURE_NAMES
    }
    evolved_signal_auc = {
        feature: _signal_auc(y_evolved_test, x_evolved_test[feature])
        for feature in FEATURE_NAMES
    }
    base_utility = generator.utility(round0_test)
    constrained_utility = generator.utility(low_profit)
    utility_reduction = (
        1.0 - constrained_utility / base_utility if base_utility > 0 else 0.0
    )
    return {
        "sample_size_per_dataset": len(round0_test),
        "operating_thresholds": {
            "fragile_round0": round(fragile_threshold, 6),
            "durable_round1": round(durable_threshold, 6),
        },
        "round0": _metrics(y_round0_test, round0_scores, fragile_threshold),
        "round1_pre_fix": _metrics(y_evolved_test, evolved_scores, fragile_threshold),
        "round1_post_fix": _metrics(
            y_evolved_test,
            recovered_scores,
            durable_threshold,
        ),
        "signal_auc": {
            feature: {
                "round0": round(round0_signal_auc[feature], 4),
                "round1": round(evolved_signal_auc[feature], 4),
            }
            for feature in FEATURE_NAMES
        },
        "utility": {
            "round0_40x": round(base_utility, 4),
            "constrained_3x_preloaded": round(constrained_utility, 4),
            "reduction": round(utility_reduction, 4),
        },
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
