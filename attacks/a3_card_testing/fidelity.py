"""Phase-3 fidelity benchmark for a locally supplied Sparkov or IEEE-CIS dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from core.population import BasePopulation, PopulationConfig
from redteam.realism import ks_distance

DatasetFormat = Literal["auto", "sparkov", "ieee-cis"]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "fidelity.json"
FIDELITY_FEATURES = (
    "amount",
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "category_frequency",
)


@dataclass(frozen=True)
class FidelityConfig:
    seed: int = 42
    max_rows: int = 50_000
    discriminator_test_size: float = 0.30


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError("dataset must be a CSV or Parquet file")


def _resolve_dataset(path: str | Path, dataset_format: DatasetFormat) -> Path:
    resolved = Path(path)
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"dataset path does not exist: {resolved}")
    candidates = {
        "sparkov": ("fraudTrain.csv", "fraudTest.csv"),
        "ieee-cis": ("train_transaction.csv", "transaction_train.csv"),
        "auto": (
            "fraudTrain.csv",
            "fraudTest.csv",
            "train_transaction.csv",
            "transaction_train.csv",
        ),
    }[dataset_format]
    for name in candidates:
        candidate = resolved / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no supported {dataset_format} transaction file found under {resolved}"
    )


def _detect_format(frame: pd.DataFrame, requested: DatasetFormat) -> Literal["sparkov", "ieee-cis"]:
    if requested != "auto":
        return requested
    if {"amt", "trans_date_trans_time"}.issubset(frame.columns):
        return "sparkov"
    if {"TransactionAmt", "TransactionDT"}.issubset(frame.columns):
        return "ieee-cis"
    raise ValueError(
        "could not identify dataset; expected Sparkov amt/trans_date_trans_time or "
        "IEEE-CIS TransactionAmt/TransactionDT columns"
    )


def _normalize_real(
    frame: pd.DataFrame,
    dataset_format: Literal["sparkov", "ieee-cis"],
) -> tuple[pd.DataFrame, int]:
    if dataset_format == "sparkov":
        required = {"amt", "trans_date_trans_time"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Sparkov columns missing: {sorted(missing)}")
        fraud_column = "is_fraud" if "is_fraud" in frame else None
        legitimate = frame.loc[frame[fraud_column].eq(0)].copy() if fraud_column else frame.copy()
        normalized = pd.DataFrame(
            {
                "amount": pd.to_numeric(legitimate["amt"], errors="coerce"),
                "timestamp": pd.to_datetime(
                    legitimate["trans_date_trans_time"], errors="coerce"
                ),
                "mcc": legitimate.get("category", pd.Series("UNKNOWN", index=legitimate.index)),
            }
        )
    else:
        required = {"TransactionAmt", "TransactionDT"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"IEEE-CIS columns missing: {sorted(missing)}")
        fraud_column = "isFraud" if "isFraud" in frame else None
        legitimate = frame.loc[frame[fraud_column].eq(0)].copy() if fraud_column else frame.copy()
        seconds = pd.to_numeric(legitimate["TransactionDT"], errors="coerce")
        normalized = pd.DataFrame(
            {
                "amount": pd.to_numeric(legitimate["TransactionAmt"], errors="coerce"),
                "timestamp": pd.Timestamp("2017-12-01") + pd.to_timedelta(seconds, unit="s"),
                "mcc": legitimate.get("ProductCD", pd.Series("UNKNOWN", index=legitimate.index)),
            }
        )
    before_cleaning = len(normalized)
    normalized["mcc"] = normalized["mcc"].fillna("UNKNOWN").astype(str)
    normalized = normalized.loc[
        normalized["amount"].gt(0)
        & np.isfinite(normalized["amount"])
        & normalized["timestamp"].notna()
    ].reset_index(drop=True)
    if len(normalized) < 30:
        raise ValueError("dataset must contain at least 30 valid legitimate transactions")
    return normalized, before_cleaning - len(normalized)


def _sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows < 30:
        raise ValueError("max_rows must be at least 30")
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _comparison_features(
    amount: pd.Series,
    timestamp: pd.Series,
    categories: pd.Series,
) -> pd.DataFrame:
    numeric_amount = pd.to_numeric(amount, errors="coerce").astype(float)
    times = pd.to_datetime(timestamp, errors="coerce")
    category = categories.fillna("UNKNOWN").astype(str)
    frequencies = category.map(category.value_counts(normalize=True))
    result = pd.DataFrame(
        {
            "amount": numeric_amount,
            "log_amount": np.log1p(numeric_amount),
            "hour_of_day": times.dt.hour.astype(float),
            "day_of_week": times.dt.dayofweek.astype(float),
            "category_frequency": frequencies.astype(float),
        }
    )
    return result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _correlation_distance(real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[float, dict]:
    real_correlation = real.corr(method="pearson").fillna(0.0)
    synthetic_correlation = synthetic.corr(method="pearson").fillna(0.0)
    difference = real_correlation.to_numpy() - synthetic_correlation.to_numpy()
    distance = float(np.linalg.norm(difference, ord="fro") / math.sqrt(difference.size))
    return distance, {
        "real": real_correlation.to_dict(),
        "synthetic": synthetic_correlation.to_dict(),
    }


def _discriminator(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: FidelityConfig,
) -> dict:
    count = min(len(real), len(synthetic))
    real_values = real.iloc[:count].copy()
    synthetic_values = synthetic.iloc[:count].copy()
    values = pd.concat([real_values, synthetic_values], ignore_index=True)
    labels = np.concatenate((np.zeros(count, dtype=np.int8), np.ones(count, dtype=np.int8)))
    X_train, X_test, y_train, y_test = train_test_split(
        values,
        labels,
        test_size=config.discriminator_test_size,
        random_state=config.seed,
        stratify=labels,
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=config.seed,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_class": "synthetic",
        "model": "LightGBM",
        "feature_importance_gain": {
            name: float(value)
            for name, value in zip(
                values.columns,
                model.booster_.feature_importance(importance_type="gain"),
                strict=True,
            )
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blocked_result() -> dict:
    return {
        "status": "blocked_no_dataset",
        "reason": (
            "No local Sparkov or IEEE-CIS transaction dataset was supplied; external-data "
            "metrics were not computed."
        ),
        "supported_datasets": ["sparkov", "ieee-cis"],
        "ks_statistics": None,
        "correlation_matrix_distance": None,
        "discriminator": None,
    }


def run_fidelity(
    dataset_path: str | Path,
    *,
    dataset_format: DatasetFormat = "auto",
    output_path: str | Path = DEFAULT_OUTPUT,
    config: FidelityConfig | None = None,
) -> dict:
    """Measure synthetic base-population fidelity against a real local dataset."""
    config = FidelityConfig() if config is None else config
    path = _resolve_dataset(dataset_path, dataset_format)
    raw = _read_table(path)
    detected_format = _detect_format(raw, dataset_format)
    normalized, dropped_rows = _normalize_real(raw, detected_format)
    real = _sample(normalized, config.max_rows, config.seed)

    generated_rows = max(100, len(real) * 3)
    synthetic_population = BasePopulation(
        real.loc[:, ["amount", "mcc"]],
        PopulationConfig(
            customers=min(20_000, max(50, len(real) // 10)),
            merchants=min(2_000, max(25, real["mcc"].nunique() * 2)),
            vpas=min(5_000, max(25, len(real) // 20)),
            days=min(90, max(1, int((real["timestamp"].max() - real["timestamp"].min()).days) + 1)),
            events=generated_rows,
        ),
    ).generate(seed=config.seed)
    synthetic_events = synthetic_population.loc[
        synthetic_population["channel"].isin(["CNP", "CARD_PRESENT"])
    ].head(len(real))
    real_features = _comparison_features(real["amount"], real["timestamp"], real["mcc"])
    synthetic_features = _comparison_features(
        synthetic_events["amount"],
        synthetic_events["timestamp"],
        synthetic_events["counterparty_mcc"],
    )
    common_count = min(len(real_features), len(synthetic_features))
    real_features = real_features.iloc[:common_count]
    synthetic_features = synthetic_features.iloc[:common_count]
    ks_statistics = {
        feature: ks_distance(real_features[feature], synthetic_features[feature])
        for feature in FIDELITY_FEATURES
    }
    correlation_distance, correlations = _correlation_distance(
        real_features, synthetic_features
    )
    result = {
        "status": "complete",
        "dataset": {
            "format": detected_format,
            "path": str(path),
            "sha256": _sha256(path),
            "rows_read": len(raw),
            "valid_legitimate_rows": len(normalized),
            "rows_sampled": len(real),
            "rows_dropped_during_cleaning": dropped_rows,
        },
        "synthetic": {
            "generator": "core.population.BasePopulation",
            "seed": config.seed,
            "rows_generated": len(synthetic_population),
            "card_rows_compared": len(synthetic_events),
        },
        "comparison_features": list(FIDELITY_FEATURES),
        "ks_statistics": ks_statistics,
        "ks_summary": {
            "mean": float(np.mean(list(ks_statistics.values()))),
            "maximum": float(max(ks_statistics.values())),
        },
        "correlation_matrix_distance": {
            "metric": "root-mean-square correlation difference",
            "value": correlation_distance,
            "matrices": correlations,
        },
        "discriminator": _discriminator(real_features, synthetic_features, config),
        "methodology": {
            "real_rows": "legitimate rows only when a fraud label is available",
            "synthetic_rows": "CNP and CARD_PRESENT base-population rows",
            "sampling": "deterministic sample without replacement",
            "discriminator_split": "stratified deterministic train/test split",
        },
    }
    write_result(output_path, result)
    return result


def write_result(output_path: str | Path, result: dict) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--format", choices=("auto", "sparkov", "ieee-cis"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=FidelityConfig.max_rows)
    parser.add_argument("--seed", type=int, default=FidelityConfig.seed)
    args = parser.parse_args()
    if args.dataset is None:
        result = blocked_result()
        write_result(args.output, result)
    else:
        result = run_fidelity(
            args.dataset,
            dataset_format=args.format,
            output_path=args.output,
            config=FidelityConfig(seed=args.seed, max_rows=args.max_rows),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
