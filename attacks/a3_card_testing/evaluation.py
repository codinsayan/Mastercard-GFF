"""Reproducible A3 detector evaluation and adaptive red-team rounds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from core.population import BasePopulation, PopulationConfig
from redteam.cem import refine
from redteam.llm_agent import invoke_codex, propose
from redteam.realism import ks_distance
from redteam.report import build_detector_report

from .detector import A3Detector
from .features import FEATURES, ROUND0_FEATURES, A3Features
from .generator import A3Generator

JsonObject = dict[str, Any]
Invoke = Callable[[str], str | JsonObject | Any]
A3_PROPOSAL_SCHEMA = Path(__file__).with_name("proposal.schema.json")


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed experiment sizes and seeds recorded with every result."""

    train_campaigns: int = 16
    test_campaigns: int = 8
    events_per_campaign: int = 500
    search_campaigns: int = 3
    search_events_per_campaign: int = 200
    cem_samples: int = 16
    cem_iterations: int = 3
    train_seed: int = 10_000
    test_seed: int = 20_000
    search_seed: int = 30_000
    detector_fpr_quantile: float = 0.99
    min_utility: float = 0.005
    max_amount_ks: float = 0.50


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: np.ndarray
    events: pd.DataFrame
    campaign_ids: np.ndarray


class _TargetedSearchSpace:
    """Delegate A3 parameters while fixing only the required throughput frontier."""

    def __init__(
        self,
        generator: A3Generator,
        current_params: dict[str, int | float],
        target_cards_per_hour: int,
    ) -> None:
        self._generator = generator
        self._defaults = {**current_params, "cards_per_hour": target_cards_per_hour}
        self._target = target_cards_per_hour

    def default_params(self) -> dict[str, int | float]:
        return dict(self._defaults)

    def param_space(self) -> dict[str, tuple[float, float]]:
        space = self._generator.param_space()
        space["cards_per_hour"] = (self._target, self._target)
        return space


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [12.5, 48.0, 109.0, 799.0],
            "mcc": ["5411", "5812", "5732", "5999"],
        }
    )


def build_dataset(
    params: dict[str, int | float],
    campaign_seeds: range | list[int],
    events_per_campaign: int,
) -> Dataset:
    """Generate independent campaigns and aligned A3 features."""
    generator = A3Generator()
    feature_block = A3Features()
    feature_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    campaign_ids: list[np.ndarray] = []
    for campaign_seed in campaign_seeds:
        population = BasePopulation(
            _reference(),
            PopulationConfig(
                customers=100,
                merchants=50,
                vpas=30,
                days=14,
                events=events_per_campaign,
            ),
        ).generate(seed=campaign_seed)
        events = generator.generate(params, population, seed=campaign_seed + 1_000_000)
        feature_frames.append(feature_block.build(events).reset_index(drop=True))
        event_frames.append(events.reset_index(drop=True))
        labels.append(events["label"].to_numpy(dtype=np.int8))
        campaign_ids.append(np.full(len(events), campaign_seed, dtype=np.int64))
    return Dataset(
        features=pd.concat(feature_frames, ignore_index=True),
        labels=np.concatenate(labels),
        events=pd.concat(event_frames, ignore_index=True),
        campaign_ids=np.concatenate(campaign_ids),
    )


def _threshold(scores: np.ndarray, labels: np.ndarray, quantile: float) -> float:
    legitimate_scores = scores[labels == 0]
    if not len(legitimate_scores):
        raise ValueError("training data must contain legitimate events")
    return float(np.quantile(legitimate_scores, quantile))


def detector_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> JsonObject:
    predictions = scores > threshold
    negatives = labels == 0
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(predictions[negatives].mean()),
        "threshold": float(threshold),
        "rows": len(labels),
        "fraud_rows": int(labels.sum()),
    }


def per_feature_auc(dataset: Dataset, feature_names: tuple[str, ...]) -> JsonObject:
    result: JsonObject = {}
    for name in feature_names:
        raw_auc = float(roc_auc_score(dataset.labels, dataset.features[name]))
        result[name] = {
            "roc_auc": raw_auc,
            "direction_normalized_auc": max(raw_auc, 1.0 - raw_auc),
        }
    return result


def _fit_round(
    params: dict[str, int | float],
    feature_names: tuple[str, ...],
    config: EvaluationConfig,
) -> tuple[A3Detector, float, Dataset, JsonObject]:
    train = build_dataset(
        params,
        range(config.train_seed, config.train_seed + config.train_campaigns),
        config.events_per_campaign,
    )
    test = build_dataset(
        params,
        range(config.test_seed, config.test_seed + config.test_campaigns),
        config.events_per_campaign,
    )
    detector = A3Detector()
    detector.fit(train.features.loc[:, feature_names], train.labels)
    training_scores = detector.score(train.features.loc[:, feature_names])
    threshold = _threshold(training_scores, train.labels, config.detector_fpr_quantile)
    test_scores = detector.score(test.features.loc[:, feature_names])
    result = {
        "params": dict(params),
        "requested_cards_per_hour": int(params["cards_per_hour"]),
        "actual_cards_per_hour": _actual_throughput(test),
        "utility": _average_utility(test),
        "detector_metrics": detector_metrics(test.labels, test_scores, threshold),
        "per_feature_auc": per_feature_auc(test, FEATURES),
    }
    return detector, threshold, test, result


def _actual_throughput(dataset: Dataset) -> float:
    rates = []
    for campaign_id in np.unique(dataset.campaign_ids):
        events = dataset.events.loc[dataset.campaign_ids == campaign_id]
        attack_rows = events.loc[events["attack_id"].eq("A3") & events["label"].eq(1)]
        timestamps = attack_rows["timestamp"]
        elapsed = (timestamps.max() - timestamps.min()).total_seconds() / 3600
        rates.append(attack_rows["card_id"].nunique() / max(elapsed, 1.0))
    return float(np.mean(rates))


def _average_utility(dataset: Dataset) -> float:
    generator = A3Generator()
    utilities = [
        generator.utility(dataset.events.loc[dataset.campaign_ids == campaign_id])
        for campaign_id in np.unique(dataset.campaign_ids)
    ]
    return float(np.mean(utilities))


def _scored_examples(
    dataset: Dataset,
    scores: np.ndarray,
    threshold: float,
    max_each: int = 5,
) -> pd.DataFrame:
    examples = dataset.events.loc[:, [
        "event_id",
        "auth_result",
        "amount",
        "channel",
        "attack_id",
        "label",
    ]].copy()
    examples["risk_score"] = scores
    examples["caught"] = scores > threshold
    fraud = examples.loc[examples["label"].eq(1)]
    caught = fraud.loc[fraud["caught"]].head(max_each)
    missed = fraud.loc[~fraud["caught"]].head(max_each)
    return pd.concat([caught, missed], ignore_index=True)


def build_a3_detector_report(
    params: dict[str, int | float],
    round_result: JsonObject,
    dataset: Dataset,
    scores: np.ndarray,
    threshold: float,
) -> JsonObject:
    """Build the shared report plus explicit A3 caught-versus-missed evidence."""
    examples = _scored_examples(dataset, scores, threshold)
    report = build_detector_report(
        A3Generator(),
        round_result,
        examples,
        params=params,
    )
    report["caught_vs_missed"] = examples.to_dict(orient="records")
    return report


def _merge_params(current: dict[str, int | float], proposed: dict[str, float]) -> dict:
    merged = {**current, **proposed}
    integer_names = {
        name for name, value in A3Generator().default_params().items() if isinstance(value, int)
    }
    return {
        name: round(value) if name in integer_names else float(value)
        for name, value in merged.items()
    }


def _adapt(
    *,
    current_params: dict[str, int | float],
    target_cards_per_hour: int,
    frozen_detector: A3Detector,
    frozen_features: tuple[str, ...],
    report: JsonObject,
    baseline_search: Dataset,
    output_dir: Path,
    round_number: int,
    config: EvaluationConfig,
    invoke: Invoke | None,
) -> tuple[dict[str, int | float], JsonObject]:
    generator = A3Generator()
    reasoner_report = {
        **report,
        "adaptation_goal": {
            "target_cards_per_hour": target_cards_per_hour,
            "instruction": (
                "Propose bounded changes that reduce detector risk while preserving utility. "
                "Treat the target throughput as an explicit economic constraint, not a final "
                "hardcoded parameter vector."
            ),
        },
    }

    def proposal_budget(proposed: dict[str, float]) -> bool:
        candidate_params = _merge_params(current_params, proposed)
        candidate = build_dataset(
            candidate_params,
            [config.search_seed],
            config.search_events_per_campaign,
        )
        return _average_utility(candidate) >= config.min_utility

    reasoning_path = output_dir / f"round_{round_number}" / "redteam_reasoning.json"
    proposal_kwargs: JsonObject = {
        "fallback": lambda _: current_params,
        "budget_check": proposal_budget,
        "cache_path": reasoning_path,
    }
    if invoke is not None:
        proposal_kwargs["invoke"] = invoke
    proposal = propose(reasoner_report, **proposal_kwargs)

    # The requested throughput is a CEM center candidate, not the selected result.
    # CEM still evaluates it against every other sampled complete parameter vector.
    refinement_proposal = {
        **proposal,
        "proposed_params": {
            **current_params,
            **proposal["proposed_params"],
            "cards_per_hour": target_cards_per_hour,
        },
    }

    baseline_amounts = baseline_search.events.loc[
        baseline_search.events["attack_id"].eq("A3"), "amount"
    ]
    objective_cache: dict[tuple, float] = {}

    def objective(candidate: dict[str, float]) -> float:
        params = _merge_params(current_params, candidate)
        key = tuple(params.items())
        if key in objective_cache:
            return objective_cache[key]
        if params["cards_per_hour"] != target_cards_per_hour:
            objective_cache[key] = -1_000_000.0
            return objective_cache[key]
        try:
            dataset = build_dataset(
                params,
                range(config.search_seed, config.search_seed + config.search_campaigns),
                config.search_events_per_campaign,
            )
            utility = _average_utility(dataset)
            candidate_amounts = dataset.events.loc[dataset.events["attack_id"].eq("A3"), "amount"]
            amount_ks = ks_distance(candidate_amounts, baseline_amounts)
            if utility < config.min_utility or amount_ks > config.max_amount_ks:
                score = -1_000_000.0
            else:
                risk = frozen_detector.score(dataset.features.loc[:, frozen_features])
                fraud_risk = float(risk[dataset.labels == 1].mean())
                score = -fraud_risk
        except (TypeError, ValueError):
            score = -1_000_000.0
        objective_cache[key] = score
        return score

    refined = refine(
        refinement_proposal,
        _TargetedSearchSpace(generator, current_params, target_cards_per_hour),
        objective,
        seed=config.search_seed + round_number,
        samples=config.cem_samples,
        iterations=config.cem_iterations,
    )
    params = _merge_params(current_params, refined)
    adaptation = {
        "target_cards_per_hour": target_cards_per_hour,
        "proposal": proposal,
        "cem_seed_params": refinement_proposal["proposed_params"],
        "refined_params": params,
        "objective": (
            "minimize frozen-detector fraud risk subject to the target-throughput, "
            "utility, and amount-KS constraints"
        ),
        "evaluated_candidates": len(objective_cache),
    }
    _write_json(output_dir / f"round_{round_number}" / "adaptation.json", adaptation)
    return params, adaptation


def _mortality(rounds: list[JsonObject]) -> list[JsonObject]:
    rows = []
    for feature in FEATURES:
        values = [
            round_result["per_feature_auc"][feature]["direction_normalized_auc"]
            for round_result in rounds
        ]
        rows.append(
            {
                "feature": feature,
                "round_0_auc": values[0],
                "round_1_auc": values[1],
                "round_2_auc": values[2],
                "round_0_to_1_delta": values[1] - values[0],
            }
        )
    return rows


def run_evaluation(
    output_dir: str | Path,
    *,
    config: EvaluationConfig | None = None,
    invoke: Invoke | None = None,
) -> JsonObject:
    """Run three measured A3 rounds and write reproducible JSON artifacts."""
    config = EvaluationConfig() if config is None else config
    output_path = Path(output_dir)
    generator = A3Generator()

    round0_detector, round0_threshold, round0_test, round0 = _fit_round(
        generator.default_params(), ROUND0_FEATURES, config
    )
    round0_scores = round0_detector.score(round0_test.features.loc[:, ROUND0_FEATURES])
    report1 = build_a3_detector_report(
        round0["params"], round0, round0_test, round0_scores, round0_threshold
    )
    _write_json(output_path / "round_0" / "metrics.json", round0)
    _write_json(output_path / "round_0" / "detector_report.json", report1)

    search0 = build_dataset(
        round0["params"],
        range(config.search_seed, config.search_seed + config.search_campaigns),
        config.search_events_per_campaign,
    )
    round1_params, adaptation1 = _adapt(
        current_params=round0["params"],
        target_cards_per_hour=12,
        frozen_detector=round0_detector,
        frozen_features=ROUND0_FEATURES,
        report=report1,
        baseline_search=search0,
        output_dir=output_path,
        round_number=1,
        config=config,
        invoke=invoke,
    )
    round1_detector, round1_threshold, round1_test, round1 = _fit_round(
        round1_params, FEATURES, config
    )
    old_round1_scores = round0_detector.score(round1_test.features.loc[:, ROUND0_FEATURES])
    round1["frozen_previous_detector_metrics"] = detector_metrics(
        round1_test.labels, old_round1_scores, round0_threshold
    )
    round1["adaptation"] = adaptation1
    round1_scores = round1_detector.score(round1_test.features.loc[:, FEATURES])
    report2 = build_a3_detector_report(
        round1_params, round1, round1_test, round1_scores, round1_threshold
    )
    _write_json(output_path / "round_1" / "metrics.json", round1)
    _write_json(output_path / "round_1" / "detector_report.json", report2)

    search1 = build_dataset(
        round1_params,
        range(config.search_seed, config.search_seed + config.search_campaigns),
        config.search_events_per_campaign,
    )
    round2_params, adaptation2 = _adapt(
        current_params=round1_params,
        target_cards_per_hour=4,
        frozen_detector=round1_detector,
        frozen_features=FEATURES,
        report=report2,
        baseline_search=search1,
        output_dir=output_path,
        round_number=2,
        config=config,
        invoke=invoke,
    )
    _, _, round2_test, round2 = _fit_round(round2_params, FEATURES, config)
    old_round2_scores = round1_detector.score(round2_test.features.loc[:, FEATURES])
    round2["frozen_previous_detector_metrics"] = detector_metrics(
        round2_test.labels, old_round2_scores, round1_threshold
    )
    round2["adaptation"] = adaptation2
    _write_json(output_path / "round_2" / "metrics.json", round2)

    summary = {
        "experiment": asdict(config),
        "rounds": [round0, round1, round2],
        "throughput": [
            {
                "round": index,
                "target": 40 if index == 0 else 12 if index == 1 else 4,
                "requested": result["requested_cards_per_hour"],
                "actual": result["actual_cards_per_hour"],
                "utility": result["utility"],
            }
            for index, result in enumerate((round0, round1, round2))
        ],
        "signal_mortality": _mortality([round0, round1, round2]),
    }
    _write_json(output_path / "summary.json", summary)
    return summary


def _write_json(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Exercise the standard deterministic fallback instead of invoking Codex CLI.",
    )
    args = parser.parse_args()

    invoke: Invoke | None = lambda prompt: invoke_codex(
        prompt, schema_path=A3_PROPOSAL_SCHEMA
    )
    if args.offline:
        def unavailable(_: str) -> str:
            raise OSError("offline evaluation requested")

        invoke = unavailable
    summary = run_evaluation(args.output_dir, invoke=invoke)
    print(json.dumps(summary["throughput"], indent=2))


if __name__ == "__main__":
    main()
