"""Deterministic adaptive-round orchestration with Codex-to-CEM fallback."""

from __future__ import annotations

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core.features import build_core_features
from core.population import BasePopulation, PopulationConfig
from core.schema import coerce, validate
from redteam.cem import refine
from redteam.llm_agent import invoke_codex, propose
from redteam.realism import check_budgets
from redteam.report import build_detector_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def _development_population(seed: int, events: int) -> pd.DataFrame:
    reference = pd.DataFrame(
        {
            "amount": [25.0, 50.0, 90.0, 150.0, 250.0, 500.0, 900.0, 1500.0],
            "mcc": ["5411", "5812", "4121", "5499", "5732", "5999", "5311", "5942"],
        }
    )
    return BasePopulation(
        reference,
        PopulationConfig(customers=max(50, events // 10), events=events),
    ).generate(seed=seed)


def check_phase0(seed: int = 42, reference_path: str | Path | None = None) -> dict[str, int]:
    """Exercise the shared schema, population, features, and proposal schema."""
    schema_path = PROJECT_ROOT / "redteam" / "proposal.schema.json"
    proposal_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if set(proposal_schema.get("required", [])) != {
        "reasoning", "hypotheses", "proposed_params"
    }:
        raise RuntimeError("Codex proposal schema is incomplete")
    if reference_path is None:
        events = _development_population(seed, 250)
    else:
        config = PopulationConfig(customers=50, merchants=25, vpas=40, days=7, events=250)
        events = BasePopulation.from_path(reference_path, config).generate(seed=seed)
    validate(events)
    features = build_core_features(events)
    if len(features) != len(events) or not features.index.equals(pd.Index(events["event_id"])):
        raise RuntimeError("core features are not aligned to event_id")
    return {"events": len(events), "features": len(features.columns)}


def _components(attack_id: str):
    if attack_id != "A1":
        raise ValueError("only A1 is integrated; add remaining packages after their merge")
    from attacks.a1_upi_mule import DETECTOR, FEATURE_BLOCK, GENERATOR

    return GENERATOR, FEATURE_BLOCK, DETECTOR.__class__


def _recall_at_fpr(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    legitimate = np.sort(scores[y == 0])
    fraud = scores[y == 1]
    if not len(legitimate) or not len(fraud):
        return 0.0
    threshold = float(np.quantile(legitimate, 1.0 - target, method="higher"))
    return float(np.mean(fraud >= threshold))


def _metric_bundle(y: np.ndarray, scores: np.ndarray, utility: float) -> dict[str, float]:
    predicted = scores >= 0.5
    fpr, tpr, _ = roc_curve(y, scores)
    qualifying = fpr[tpr >= 0.85]
    return {
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "auc_pr": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_fpr_0.001": _recall_at_fpr(y, scores, 0.001),
        "fpr_at_recall_0.85": float(qualifying.min()) if len(qualifying) else 1.0,
        "utility": float(utility),
    }


def _feature_auc(X: pd.DataFrame, y: np.ndarray, prefix: str) -> dict[str, float]:
    result = {}
    for column in X.columns:
        if not column.startswith(prefix):
            continue
        values = X[column].to_numpy(dtype=float)
        if len(np.unique(values)) < 2:
            result[column] = 0.5
        else:
            auc = float(roc_auc_score(y, values))
            result[column] = max(auc, 1.0 - auc)
    return result


def _write_default_reasoning(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "runner": {"runner": "round_zero", "model": "none"},
                "fallback_used": False,
                "report": {}, "prompt": "", "raw_responses": [],
                "proposal": {
                    "reasoning": "Round 0 uses the attack's frozen default parameters.",
                    "hypotheses": [], "proposed_params": params,
                },
                "failures": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_rounds(
    *, rounds: list[int], seed: int = 42, attack_id: str = "A1",
    results_root: str | Path = DEFAULT_RESULTS, population_events: int = 500,
    max_victims: int | None = None, reference_path: str | Path | None = None,
    use_codex: bool = False,
    codex_timeout: float = 120, cem_samples: int = 16, cem_iterations: int = 3,
) -> dict[int, dict[str, Any]]:
    """Run contiguous rounds and write all frozen outputs; offline CEM is default."""
    if not rounds or sorted(set(rounds)) != rounds or rounds[0] != 0:
        raise ValueError("rounds must be unique, sorted, and start at 0")
    if any(right != left + 1 for left, right in pairwise(rounds)):
        raise ValueError("rounds must be contiguous")
    generator, feature_block, detector_type = _components(attack_id)
    results_root = Path(results_root)
    if reference_path is None:
        population = _development_population(seed, population_events)
    else:
        config = PopulationConfig(
            customers=max(50, population_events // 10), events=population_events
        )
        population = BasePopulation.from_path(reference_path, config).generate(seed=seed)
    params = generator.default_params()
    if max_victims is not None:
        low = generator.param_space()["n_victims"][0]
        params["n_victims"] = max(low, min(max_victims, params["n_victims"]))
    summaries: dict[int, dict[str, Any]] = {}

    for round_number in rounds:
        round_dir = results_root / f"round_{round_number}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if round_number == 0:
            _write_default_reasoning(round_dir / "redteam_reasoning.json", params)
        fraud = generator.generate(params, population, seed=seed + round_number * 1000).copy()
        fraud["round"] = np.int8(round_number)
        legitimate = population.copy()
        legitimate["round"] = np.int8(round_number)
        events = coerce(pd.concat([legitimate, fraud], ignore_index=True))
        events = events.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
        validate(events)
        features = build_core_features(events).join(feature_block.build(events), how="inner")
        labels = events.set_index("event_id").reindex(features.index)["label"].to_numpy(dtype=np.int8)
        detector = detector_type()
        detector.fit(features, labels)
        scores = detector.score(features)
        per_attack = _metric_bundle(labels, scores, generator.utility(fraud))
        metrics = {
            "round": round_number, "params": params,
            "per_attack": {attack_id: per_attack},
            "signal_auc": _feature_auc(features, labels, attack_id.lower() + "_"),
            "overall": {
                "auc_pr": per_attack["auc_pr"],
                "fpr_at_recall_0.85": per_attack["fpr_at_recall_0.85"],
            },
            "development_fixture": True,
        }
        score_frame = pd.DataFrame(
            {"event_id": features.index.astype("string"), "risk_score": scores}
        )
        events.to_parquet(round_dir / "events.parquet", index=False)
        features.reset_index().to_parquet(round_dir / "features.parquet", index=False)
        score_frame.to_parquet(round_dir / "scores.parquet", index=False)
        (round_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summaries[round_number] = metrics
        if round_number == rounds[-1]:
            break

        report = build_detector_report(generator, metrics, fraud.head(10), params=params)
        next_dir = results_root / f"round_{round_number + 1}"

        current_params = params.copy()

        def fallback(_: dict[str, Any], snapshot=current_params) -> dict[str, Any]:
            return snapshot

        if use_codex:
            invoke = lambda prompt: invoke_codex(prompt, timeout=codex_timeout)
        else:
            def invoke(_prompt: str):
                raise OSError("Codex disabled; using deterministic pure-CEM fallback")

        def proposal_budget(
            candidate: dict[str, float],
            snapshot=current_params,
            candidate_round=round_number + 1,
        ) -> bool:
            try:
                complete = snapshot | candidate
                proposed_events = generator.generate(
                    complete, population, seed=seed + candidate_round * 1000
                )
                return check_budgets(
                    generator, proposed_events, population, min_utility=0.02, max_ks=0.95
                ).accepted
            except (TypeError, ValueError):
                return False

        proposal = propose(
            report, invoke=invoke, fallback=fallback, attempts=3 if use_codex else 1,
            budget_check=proposal_budget,
            cache_path=next_dir / "redteam_reasoning.json",
        )
        fitted_detector = detector
        fitted_columns = list(features.columns)

        def objective(
            candidate: dict[str, float],
            candidate_round=round_number + 1,
            columns=tuple(fitted_columns),
            scorer=fitted_detector,
        ) -> float:
            try:
                candidate_events = generator.generate(
                    candidate, population, seed=seed + candidate_round * 1000
                )
                budget = check_budgets(
                    generator, candidate_events, population, min_utility=0.02, max_ks=0.95
                )
                if not budget.accepted:
                    return -math.inf
                candidate_all = coerce(pd.concat([population, candidate_events], ignore_index=True))
                candidate_all = candidate_all.sort_values(
                    ["timestamp", "event_id"], kind="stable"
                ).reset_index(drop=True)
                candidate_X = build_core_features(candidate_all).join(
                    feature_block.build(candidate_all), how="inner"
                ).reindex(columns=columns, fill_value=0.0)
                candidate_scores = scorer.score(candidate_X)
                fraud_ids = set(candidate_events["event_id"].astype(str))
                fraud_mask = candidate_X.index.astype(str).isin(fraud_ids)
                evasion = 1.0 - float(candidate_scores[fraud_mask].mean())
                return evasion + 0.1 * budget.utility
            except (TypeError, ValueError, RuntimeError):
                return -math.inf

        params = refine(
            proposal, generator, objective, seed=seed + round_number,
            samples=cem_samples, iterations=cem_iterations,
        )
        cache_path = next_dir / "redteam_reasoning.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["proposal"]["cem_refined_params"] = params
        cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive fraud simulation rounds")
    parser.add_argument("--round", type=int)
    parser.add_argument("--rounds", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--population-events", type=int, default=500)
    parser.add_argument("--max-victims", type=int)
    parser.add_argument("--cem-samples", type=int, default=16)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--use-codex", action="store_true")
    parser.add_argument("--codex-timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = args.rounds if args.rounds is not None else [args.round or 0]
    if args.dry_run:
        checked = check_phase0(args.seed, args.reference)
        print(
            "Phase 0 ready; "
            f"validated {checked['events']} events and {checked['features']} core features; "
            f"selected rounds={selected}, seed={args.seed}"
        )
        return
    run_rounds(
        rounds=selected, seed=args.seed, results_root=args.results_root,
        population_events=args.population_events, max_victims=args.max_victims,
        reference_path=args.reference,
        use_codex=args.use_codex, codex_timeout=args.codex_timeout,
        cem_samples=args.cem_samples, cem_iterations=args.cem_iterations,
    )


if __name__ == "__main__":
    main()
