"""Reproducible A1 signal-mortality experiment for Iterations 8 and 9."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core.features import build_core_features
from core.population import BasePopulation, PopulationConfig

from .detector import A1Detector
from .features import A1Features
from .generator import A1Generator

ROUND0_FEATURES = [
    "a1_benef_account_age_days",
    "a1_benef_fanin_24h",
    "a1_dissipation_ratio",
    "a1_payer_amount_zscore",
    "a1_payee_added_to_transfer_seconds",
]
DURABLE_FEATURES = A1Features.feature_names()
DEFAULT_OUTPUT = Path(__file__).with_name("results") / "iteration8_metrics.json"
MAX_REFERENCE_ROWS = 500_000


@dataclass
class EvaluationSet:
    X: pd.DataFrame
    y: np.ndarray
    fraud_events: pd.DataFrame


def _reference_frame() -> pd.DataFrame:
    """Small development fixture; final fidelity runs must use Sparkov/IEEE-CIS."""
    return pd.DataFrame(
        {
            "amount": [50.0, 85.0, 120.0, 180.0, 250.0, 400.0, 650.0, 900.0, 1500.0],
            "mcc": ["5411", "5812", "4121", "5499", "5732", "5999", "5311", "5541", "5942"],
        }
    )


def _reference_provenance(path: Path, frame: pd.DataFrame, source_rows: int) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "kind": "external_reference",
        "source_file": path.name,
        "sha256": digest.hexdigest(),
        "source_rows": source_rows,
        "legitimate_rows_used": len(frame),
    }


def _load_external_reference(path: Path, *, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Load legitimate amount/category marginals from Sparkov or IEEE-CIS."""
    if not path.is_file():
        raise FileNotFoundError(f"reference dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        raw = pd.read_csv(path)
    else:
        raise ValueError("reference must be a CSV, TXT, or Parquet file")

    amount_name = next(
        (name for name in ("amount", "amt", "TransactionAmt") if name in raw), None
    )
    if amount_name is None:
        raise ValueError("reference needs amount, amt, or TransactionAmt")
    category_name = next(
        (name for name in ("mcc", "category", "ProductCD") if name in raw), None
    )
    label_name = next(
        (name for name in ("isFraud", "is_fraud", "label") if name in raw), None
    )
    source_rows = len(raw)
    if label_name is not None:
        labels = pd.to_numeric(raw[label_name], errors="coerce")
        raw = raw.loc[labels.eq(0)]
    normalized = pd.DataFrame(
        {
            "amount": pd.to_numeric(raw[amount_name], errors="coerce"),
            "mcc": raw[category_name].astype("string") if category_name else "UNKNOWN",
        }
    )
    normalized = normalized.loc[
        np.isfinite(normalized["amount"]) & normalized["amount"].gt(0)
    ].dropna(subset=["mcc"])
    if normalized.empty:
        raise ValueError("reference contains no usable legitimate positive-amount rows")
    if len(normalized) > MAX_REFERENCE_ROWS:
        normalized = normalized.sample(MAX_REFERENCE_ROWS, random_state=seed)
    normalized = normalized.reset_index(drop=True)
    provenance = _reference_provenance(path, normalized, source_rows)
    provenance["format"] = (
        "IEEE-CIS"
        if amount_name == "TransactionAmt"
        else "Sparkov"
        if amount_name == "amt" or "sparkov" in path.name.lower()
        else "normalized"
    )
    provenance["amount_column"] = amount_name
    provenance["category_column"] = category_name or "UNKNOWN"
    provenance["label_column"] = label_name
    return normalized, provenance


def _enrich_legitimate_a1_context(population: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Add deterministic, plausible A1 context to the synthetic legitimate fixture.

    This prevents missing metadata and uniformly old VPAs from becoming label shortcuts.
    It is development calibration, not a substitute for a public reference dataset.
    """
    enriched = population.copy(deep=True)
    rng = np.random.default_rng(seed)
    p2p = enriched["channel"].eq("UPI_P2P").to_numpy()
    first_time = enriched["is_first_time_payee"].eq(1).to_numpy()

    # Model a mixed VPA portfolio with genuinely new, established, and old accounts.
    age_bucket = rng.choice(3, len(enriched), p=[0.12, 0.23, 0.65])
    ages = np.select(
        [age_bucket == 0, age_bucket == 1],
        [rng.integers(1, 31, len(enriched)), rng.integers(31, 181, len(enriched))],
        default=rng.integers(181, 3651, len(enriched)),
    )
    enriched["counterparty_account_age_days"] = enriched[
        "counterparty_account_age_days"
    ].astype("int64")
    enriched.loc[p2p, "counterparty_account_age_days"] = ages[p2p]

    # First-time legitimate payees can be paid shortly after addition too. Existing
    # payees retain the neutral missing value used by the feature implementation.
    delays = np.clip(rng.lognormal(np.log(900.0), 1.35, len(enriched)), 20, 86_400 * 30)
    timestamps = pd.to_datetime(enriched["timestamp"])
    meta_values: list[dict] = []
    for index, value in enumerate(enriched["meta"]):
        meta = dict(value) if isinstance(value, dict) else {}
        if p2p[index] and first_time[index]:
            added_at = timestamps.iloc[index] - pd.to_timedelta(delays[index], unit="s")
            meta["a1_payee_added_at"] = added_at.isoformat()
        meta_values.append(meta)
    enriched["meta"] = meta_values
    return enriched


def _build_set(
    population: pd.DataFrame,
    legitimate_ids: set[str],
    params: dict,
    seed: int,
) -> EvaluationSet:
    generator = A1Generator()
    fraud = generator.generate(params, population, seed=seed)
    events = pd.concat([population, fraud], ignore_index=True)
    events = events.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    attack_features = A1Features().build(events)
    core_features = build_core_features(events)
    features = core_features.join(attack_features, how="inner")
    selected_ids = legitimate_ids | set(fraud["event_id"].astype(str))
    selected = events["event_id"].astype(str).isin(selected_ids)
    selected_events = events.loc[selected]
    X = features.reindex(selected_events["event_id"].astype("string")).copy()
    y = selected_events["label"].to_numpy(dtype=np.int8)
    return EvaluationSet(X=X, y=y, fraud_events=fraud)


def _threshold_at_training_fpr(scores: np.ndarray, y: np.ndarray, target_fpr: float) -> float:
    legitimate = scores[y == 0]
    if len(legitimate) == 0:
        return 0.5
    return float(np.quantile(legitimate, 1.0 - target_fpr, method="higher"))


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    fpr, tpr, _ = roc_curve(y, scores)
    qualifying = fpr[tpr >= 0.85]
    return {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "auc_pr": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_fpr_0.001": float(tpr[fpr <= 0.001].max(initial=0.0)),
        "fpr_at_recall_0.85": float(qualifying.min()) if len(qualifying) else 1.0,
        "threshold": float(threshold),
        "rows": len(y),
        "fraud_rows": int(y.sum()),
    }


def _signal_auc(X: pd.DataFrame, y: np.ndarray, names: list[str]) -> dict[str, float]:
    result = {}
    for name in names:
        values = X[name].to_numpy(dtype=float)
        if len(np.unique(values)) < 2:
            result[name] = 0.5
            continue
        auc = float(roc_auc_score(y, values))
        result[name] = max(auc, 1.0 - auc)
    return result


def _infrastructure(events: pd.DataFrame) -> dict:
    layers = events["meta"].map(lambda value: value.get("a1_mule_layer", 0))
    return {
        "mules": int(events.loc[layers.isin([1, 2]), "counterparty_id"].nunique()),
        "cashout_points": int(events.loc[layers.eq(3), "counterparty_id"].nunique()),
        "cashout_value": float(events.loc[layers.eq(3), "amount"].sum()),
        "utility": A1Generator().utility(events),
    }


def run_experiment(
    *,
    population_events: int = 6_000,
    output: Path = DEFAULT_OUTPUT,
    reference_path: Path | None = None,
) -> dict:
    generator = A1Generator()
    if reference_path is None:
        reference = _reference_frame()
        reference_info = {
            "kind": "development_fixture",
            "description": "embedded amount/MCC fixture",
            "legitimate_rows_used": len(reference),
        }
    else:
        reference, reference_info = _load_external_reference(reference_path)
    population = BasePopulation(
        reference,
        PopulationConfig(customers=500, events=population_events),
    ).generate(seed=42)
    population = _enrich_legitimate_a1_context(population, seed=43)
    ordered_legit = population.sort_values(["timestamp", "event_id"], kind="stable")
    split = int(len(ordered_legit) * 0.7)
    train_ids = set(ordered_legit.iloc[:split]["event_id"].astype(str))
    test_ids = set(ordered_legit.iloc[split:]["event_id"].astype(str))

    default = generator.default_params()
    default["n_victims"] = 20
    adapted = default.copy()
    adapted.update(
        {
            "mule_age_days": 45,
            "split_factor": 15,
            "dwell_time_hours": 6.0,
            "n_cashout_points": 20,
            "n_mules_layer1": 120,
        }
    )

    default_train = _build_set(population, train_ids, default, seed=101)
    default_test = _build_set(population, test_ids, default, seed=102)
    adapted_train = _build_set(population, train_ids, adapted, seed=201)
    adapted_test = _build_set(population, test_ids, adapted, seed=202)
    core_names = [name for name in default_train.X if not name.startswith("a1_")]
    old_names = core_names + ROUND0_FEATURES
    durable_names = core_names + DURABLE_FEATURES

    frozen = A1Detector()
    frozen.fit(default_train.X[old_names], default_train.y)
    training_scores = frozen.score(default_train.X[old_names])
    frozen_threshold = _threshold_at_training_fpr(training_scores, default_train.y, 0.001)
    default_scores = frozen.score(default_test.X[old_names])
    adapted_frozen_scores = frozen.score(adapted_test.X[old_names])

    retrained_old = A1Detector()
    retrained_old.fit(adapted_train.X[old_names], adapted_train.y)
    retrained_threshold = _threshold_at_training_fpr(
        retrained_old.score(adapted_train.X[old_names]), adapted_train.y, 0.001
    )
    adapted_retrained_scores = retrained_old.score(adapted_test.X[old_names])

    durable = A1Detector()
    durable.fit(adapted_train.X[durable_names], adapted_train.y)
    durable_threshold = _threshold_at_training_fpr(
        durable.score(adapted_train.X[durable_names]), adapted_train.y, 0.001
    )
    adapted_durable_scores = durable.score(adapted_test.X[durable_names])

    result = {
        "methodology": {
            "population_events": population_events,
            "reference": reference_info,
            "fixture_corrections": [
                "legitimate first-time UPI payees receive deterministic payee-addition history",
                "legitimate UPI beneficiary ages use a new/established/old mixture",
            ],
            "campaign_seeds": {
                "default_train": 101,
                "default_test": 102,
                "adapted_train": 201,
                "adapted_test": 202,
            },
            "legitimate_split": "earliest 70% train, latest 30% test",
            "feature_latency": "graph and dissipation features are delayed monitoring signals",
        },
        "params": {"round_0": default, "adapted": adapted},
        "experiments": {
            "round_0": _metrics(default_test.y, default_scores, frozen_threshold),
            "adapted_frozen_detector": _metrics(
                adapted_test.y, adapted_frozen_scores, frozen_threshold
            ),
            "adapted_retrained_old_features": _metrics(
                adapted_test.y, adapted_retrained_scores, retrained_threshold
            ),
            "adapted_durable_features": _metrics(
                adapted_test.y, adapted_durable_scores, durable_threshold
            ),
        },
        "signal_auc": {
            "round_0": _signal_auc(default_test.X, default_test.y, ROUND0_FEATURES),
            "adapted": _signal_auc(adapted_test.X, adapted_test.y, DURABLE_FEATURES),
        },
        "infrastructure": {
            "round_0": _infrastructure(default_test.fraud_events),
            "adapted": _infrastructure(adapted_test.fraud_events),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-events", type=int, default=6_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    result = run_experiment(
        population_events=args.population_events,
        output=args.output,
        reference_path=args.reference,
    )
    print(json.dumps(result["experiments"], indent=2))


if __name__ == "__main__":
    main()
