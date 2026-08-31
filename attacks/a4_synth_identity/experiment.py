"""Reproducible account-level A4 round experiment and output writer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attacks.a4_synth_identity.detector import A4Detector
from attacks.a4_synth_identity.features import FEATURE_NAMES, A4Features
from attacks.a4_synth_identity.generator import A4Generator
from core.population import BasePopulation, PopulationConfig
from eval.metrics import binary_metrics, per_feature_auc
from redteam.cem import refine
from redteam.llm_agent import invoke_codex, propose
from redteam.realism import check_budgets, ks_distance
from redteam.report import build_detector_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PARQUET = PROJECT_ROOT / "data" / "processed" / "sparkov_reference.parquet"
# Marginals are fitted to the public Sparkov transaction set when it is present; see
# data/README.md for the download step. The inline table is only a shape-compatible
# stand-in so tests and a fresh clone still run without the 338 MB source file.
FALLBACK_REFERENCE = pd.DataFrame(
    {
        "amount": [8.0, 15.0, 35.0, 80.0, 250.0, 900.0, 2_500.0],
        "mcc": ["5411", "5812", "5812", "5732", "5999", "5311", "5944"],
    }
)


def load_reference(path: str | Path | None = None) -> pd.DataFrame:
    """Return the real reference marginals, or the stand-in table when absent."""
    source = Path(path) if path is not None else REFERENCE_PARQUET
    if not source.exists():
        if path is not None:
            raise FileNotFoundError(f"reference data not found: {source}")
        return FALLBACK_REFERENCE
    frame = (
        pd.read_parquet(source)
        if source.suffix.lower() == ".parquet"
        else pd.read_csv(source, usecols=lambda name: name in {"amount", "amt", "mcc", "category"})
    )
    frame = frame.rename(columns={"amt": "amount", "category": "mcc"})
    if "mcc" not in frame:
        frame["mcc"] = "UNKNOWN"
    return frame[["amount", "mcc"]]


REFERENCE = load_reference()
REFERENCE_IS_REAL = len(REFERENCE) > len(FALLBACK_REFERENCE)
# The brief's mortality story concerns the three cheap identity-overlap signals. The age
# feature remains measured, but is not allowed to conceal their Round-1 failure.
STATIC_FEATURES = list(FEATURE_NAMES[:3])
DURABLE_FEATURES = list(FEATURE_NAMES[4:])


@dataclass
class RoundData:
    events: pd.DataFrame
    features: pd.DataFrame
    accounts: pd.DataFrame
    utility: float


def add_organic_credit_metadata(events: pd.DataFrame, days: int, seed: int) -> pd.DataFrame:
    """Give legitimate card traffic noisy linear or concave credit trajectories."""
    rng = np.random.default_rng(seed)
    output = events.loc[events["channel"].isin(["CNP", "CARD_PRESENT"])].copy()
    customer_codes, customers = pd.factorize(output["customer_id"], sort=True)
    first_timestamp = output.groupby("customer_id")["timestamp"].transform("min")
    elapsed = (output["timestamp"] - first_timestamp).dt.days.clip(lower=0).to_numpy()
    exponent = rng.uniform(0.65, 1.15, len(customers))
    limits = rng.integers(35_000, 150_001, len(customers)).astype(float)
    progress = np.clip(elapsed / max(1, days - 1), 0, 1)
    # Real portfolios contain revolvers who run close to their limit, so a minority of
    # organic accounts must be able to cross the 0.85 line the bust-out features test.
    # Without this the sync feature is zero for every legitimate account by construction.
    peak = np.clip(rng.beta(2.0, 3.2, len(customers)) * 1.45, 0.08, 0.98)
    utilization = np.clip(
        0.05
        + (peak[customer_codes] - 0.05) * progress ** exponent[customer_codes]
        + rng.normal(0, 0.06, len(output)),
        0.01,
        0.99,
    )
    # Organic repayment behaviour is heterogeneous: some people habitually pay in full,
    # others vary month to month. A single uniform draw made every account equally erratic.
    payment_center = rng.uniform(0.70, 1.0, len(customers))
    payment_spread = rng.uniform(0.005, 0.18, len(customers))
    payment = np.clip(
        rng.normal(payment_center[customer_codes], payment_spread[customer_codes]), 0.4, 1.0
    )
    output["meta"] = [
        {
            "a4_identity_cluster_id": "",
            "a4_playbook_id": "organic",
            "a4_days_since_onboarding": int(day),
            "a4_credit_limit": float(limit),
            "a4_utilization": round(float(use), 4),
            "a4_payment_ratio": round(float(paid), 4),
            "a4_phase": "BUILD",
            "a4_address_token": f"organic-address-{customer}",
            "a4_device_prefix": f"organic-device-{customer}",
            "a4_phone_block": f"organic-phone-{customer}",
        }
        for customer, day, limit, use, paid in zip(
            output["customer_id"],
            elapsed,
            limits[customer_codes],
            utilization,
            payment,
            strict=True,
        )
    ]
    # Each organic account gets one genuinely large purchase, placed on whichever of its
    # events falls closest to a per-customer target day. Grouping by a single sort keeps
    # this linear; scanning per customer was quadratic in the population size.
    target_days = rng.integers(20, max(21, days), len(customers))
    large_multipliers = rng.uniform(0.22, 0.38, len(customers))
    distance = np.abs(elapsed - target_days[customer_codes])
    order = np.lexsort((distance, customer_codes))
    first_of_customer = np.flatnonzero(
        np.concatenate(([True], np.diff(customer_codes[order]) != 0))
    )
    chosen = order[first_of_customer]
    amounts = output["amount"].to_numpy(dtype=float).copy()
    amounts[chosen] = np.maximum(
        amounts[chosen],
        limits[customer_codes[chosen]] * large_multipliers[customer_codes[chosen]],
    )
    output["amount"] = amounts
    return output


# Share of base-population events landing on the card channels A4 observes.
CARD_CHANNEL_SHARE = 0.48
MAX_POPULATION_EVENTS = 60_000
# A4 writes under its own subdirectory. results/round_N/ is reserved by the frozen
# round format for the merged all-attack run, and would be overwritten by the shared
# orchestrator in loop/run_rounds.py.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "a4"


def organic_event_budget(customers: int, build_days: int) -> int:
    """Size the population so organic accounts carry a history as deep as a synthetic one.

    A synthetic identity emits one build transaction every three days. When organic
    accounts carried only two or three events each, per-account statistics such as
    repayment regularity were estimated from far fewer samples for one class than the
    other, which separated the classes on sample size rather than on behaviour.
    """
    per_account = max(4.0, build_days / 3)
    return int(min(MAX_POPULATION_EVENTS, customers * per_account / CARD_CHANNEL_SHARE))


def make_round_data(
    params: dict[str, Any],
    seed: int,
    round_number: int,
    *,
    population_events: int | None = None,
    trajectory_epochs: int = 4,
) -> RoundData:
    """Generate independent legitimate/fraud accounts and one latest snapshot per account."""
    identities = int(params["n_clusters"] * params["identities_per_cluster"])
    days = max(120, int(params["build_days"] + params["bustout_sync_window_days"] + 2))
    customers = max(300, identities * 2)
    if population_events is None:
        population_events = organic_event_budget(customers, int(params["build_days"]))
    config = PopulationConfig(
        customers=customers,
        merchants=150,
        vpas=100,
        days=days,
        events=population_events,
    )
    population = BasePopulation(REFERENCE, config).generate(seed + 10_000)
    generator = A4Generator()
    fraud = generator.generate(params, population, seed)
    legitimate = add_organic_credit_metadata(population, days, seed + 20_000)
    fraud["round"] = np.int8(round_number)
    legitimate["round"] = np.int8(round_number)
    events = pd.concat([legitimate, fraud], ignore_index=True).sort_values(
        ["timestamp", "event_id"], kind="stable"
    )
    features = A4Features(trajectory_epochs=trajectory_epochs).build(events)
    latest = (
        events.sort_values(["timestamp", "event_id"])
        .groupby("customer_id", sort=False)
        .tail(1)
        .set_index("event_id")
    )
    accounts = latest[["customer_id", "attack_id", "label"]].join(features.loc[latest.index])
    return RoundData(events, features, accounts, generator.utility(fraud))


def _fit(data: list[RoundData], columns: list[str]) -> A4Detector:
    frame = pd.concat([item.accounts for item in data])
    detector = A4Detector()
    detector.fit(frame[columns], frame["label"].to_numpy())
    return detector


def _measure(
    detector: A4Detector, data: RoundData, columns: list[str]
) -> tuple[dict[str, float], np.ndarray]:
    scores = detector.score(data.accounts[columns])
    return binary_metrics(data.accounts["label"].to_numpy(), scores), scores


class _SearchSpace:
    def __init__(self, current: dict[str, Any], names: set[str]) -> None:
        generator = A4Generator()
        self._defaults = {name: current[name] for name in names}
        self._space = {name: generator.param_space()[name] for name in names}

    def default_params(self) -> dict[str, Any]:
        return dict(self._defaults)

    def param_space(self) -> dict[str, tuple[float, float]]:
        return dict(self._space)


def _adapt(
    current: dict[str, Any],
    report: dict[str, Any],
    detector: A4Detector,
    columns: list[str],
    output_path: Path,
    *,
    seed: int,
    use_codex: bool,
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    generator = A4Generator()
    population = BasePopulation(
        REFERENCE,
        PopulationConfig(customers=50, merchants=30, vpas=20, days=120, events=500),
    ).generate(seed + 30_000)

    def budget(params: dict[str, float]) -> bool:
        merged = current | params
        in_bounds = all(
            low <= merged[name] <= high for name, (low, high) in generator.param_space().items()
        )
        if not in_bounds:
            return False
        test_params = merged | {"n_clusters": 2, "identities_per_cluster": 10}
        fraud = generator.generate(test_params, population, seed + 40_000)
        build = fraud[fraud["meta"].map(lambda value: value["a4_phase"] == "BUILD")]
        return (
            generator.utility(fraud) >= 0.1
            and ks_distance(build["amount"], population["amount"]) <= 0.35
        )

    def offline(_: str):
        raise OSError("Codex disabled; using deterministic CEM fallback")

    proposal = propose(
        report,
        invoke=invoke_codex if use_codex else offline,
        fallback=lambda _: fallback,
        budget_check=budget,
        cache_path=output_path,
        attempts=3 if use_codex else 1,
    )
    proposed = current | proposal["proposed_params"]
    changed = {
        name for name, value in proposal["proposed_params"].items() if value != current[name]
    }
    if not changed:
        proposed |= fallback
        changed = set(fallback)
    search = _SearchSpace(proposed, changed)

    def objective(candidate: dict[str, float]) -> float:
        test_params = proposed | candidate | {"n_clusters": 2, "identities_per_cluster": 10}
        fraud = generator.generate(test_params, population, seed + 40_000)
        features = A4Features(trajectory_epochs=0).build(fraud)
        latest = fraud.groupby("customer_id", sort=False).tail(1).set_index("event_id")
        evasion = 1 - float(detector.score(features.loc[latest.index, columns]).mean())
        build = fraud[fraud["meta"].map(lambda value: value["a4_phase"] == "BUILD")]
        realism = 1 - min(1.0, ks_distance(build["amount"], population["amount"]))
        return 0.65 * evasion + 0.25 * generator.utility(fraud) + 0.10 * realism

    refined = refine(
        {"proposed_params": {name: proposed[name] for name in changed}},
        search,
        objective,
        seed=seed,
        samples=8,
        iterations=2,
        elite_fraction=0.25,
    )
    return current | proposal["proposed_params"] | refined, proposal


def _write_round(
    output_root: Path,
    round_number: int,
    data: RoundData,
    scores: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    destination = output_root / f"round_{round_number}"
    destination.mkdir(parents=True, exist_ok=True)
    data.events.to_parquet(destination / "events.parquet", index=False)
    data.features.reset_index().to_parquet(destination / "features.parquet", index=False)
    account_scores = pd.Series(scores, index=data.accounts["customer_id"].to_numpy())
    event_scores = data.events["customer_id"].map(account_scores).fillna(0)
    pd.DataFrame({"event_id": data.events["event_id"], "a4_score": event_scores}).to_parquet(
        destination / "scores.parquet", index=False
    )
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )


def run_experiment(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    seed: int = 42,
    use_codex: bool = False,
    quick: bool = False,
) -> dict[str, Any]:
    """Run A4's static failure, durable recovery, and second adaptation."""
    output_root = Path(output_root)
    generator = A4Generator()
    base = generator.default_params()
    if quick:
        base |= {"n_clusters": 4, "identities_per_cluster": 15, "build_days": 45}
    # The event budget is always derived from the campaign size so that organic accounts
    # keep a history comparable to a synthetic one; quick mode shrinks the campaign, not
    # the per-account depth.
    population_events = None
    epochs = 1 if quick else 4

    r0_train = make_round_data(
        base, seed, 0, population_events=population_events, trajectory_epochs=epochs
    )
    r0_test = make_round_data(
        base, seed + 1, 0, population_events=population_events, trajectory_epochs=epochs
    )
    static_detector = _fit([r0_train], STATIC_FEATURES)
    r0_metrics, r0_scores = _measure(static_detector, r0_test, STATIC_FEATURES)
    r0_auc = per_feature_auc(
        r0_test.accounts[list(FEATURE_NAMES)], r0_test.accounts["label"].to_numpy()
    )
    r0_report = build_detector_report(
        generator,
        {"detector": r0_metrics, "signal_auc": r0_auc},
        r0_test.events,
        params=base,
    )
    r1_params, r1_proposal = _adapt(
        base,
        r0_report,
        static_detector,
        STATIC_FEATURES,
        output_root / "round_1" / "redteam_reasoning.json",
        seed=seed + 100,
        use_codex=use_codex,
        fallback={"attribute_overlap_rate": 0.02},
    )
    r1_train = make_round_data(
        r1_params, seed + 2, 1, population_events=population_events, trajectory_epochs=epochs
    )
    r1_test = make_round_data(
        r1_params, seed + 3, 1, population_events=population_events, trajectory_epochs=epochs
    )
    r1_pre, _ = _measure(static_detector, r1_test, STATIC_FEATURES)
    durable_detector = _fit([r0_train, r1_train], DURABLE_FEATURES)
    r1_post, r1_scores = _measure(durable_detector, r1_test, DURABLE_FEATURES)
    r1_auc = per_feature_auc(
        r1_test.accounts[list(FEATURE_NAMES)], r1_test.accounts["label"].to_numpy()
    )
    r1_report = build_detector_report(
        generator,
        {"pre_fix": r1_pre, "post_fix": r1_post, "signal_auc": r1_auc},
        r1_test.events,
        params=r1_params,
    )
    r2_params, r2_proposal = _adapt(
        r1_params,
        r1_report,
        durable_detector,
        DURABLE_FEATURES,
        output_root / "round_2" / "redteam_reasoning.json",
        seed=seed + 200,
        use_codex=use_codex,
        fallback={
            "playbook_variance": 0.6,
            "build_days": 150,
            "utilization_curve_exponent": 1.15,
            "bustout_sync_window_days": 20,
        },
    )
    r2_train = make_round_data(
        r2_params, seed + 4, 2, population_events=population_events, trajectory_epochs=epochs
    )
    r2_test = make_round_data(
        r2_params, seed + 5, 2, population_events=population_events, trajectory_epochs=epochs
    )
    r2_pre, _ = _measure(durable_detector, r2_test, DURABLE_FEATURES)
    recovered_detector = _fit([r0_train, r1_train, r2_train], list(FEATURE_NAMES))
    r2_post, r2_scores = _measure(recovered_detector, r2_test, list(FEATURE_NAMES))
    r2_auc = per_feature_auc(
        r2_test.accounts[list(FEATURE_NAMES)], r2_test.accounts["label"].to_numpy()
    )

    build_r2 = r2_test.events[
        r2_test.events["attack_id"].eq("A4")
        & r2_test.events["meta"].map(lambda value: value.get("a4_phase") == "BUILD")
    ]
    legitimate_r2 = r2_test.events[r2_test.events["attack_id"].eq("LEGIT")]
    utility_budget = check_budgets(
        generator,
        r2_test.events[r2_test.events["attack_id"].eq("A4")],
    )
    build_ks = ks_distance(build_r2["amount"], legitimate_r2["amount"])
    budget = {
        "accepted": utility_budget.accepted and build_ks <= 0.35,
        "utility": utility_budget.utility,
        "max_ks": build_ks,
        "reason": utility_budget.reason
        or (f"build amount KS {build_ks:.3f} exceeds 0.350" if build_ks > 0.35 else ""),
    }
    rounds = {
        "reference": {
            "source": str(REFERENCE_PARQUET) if REFERENCE_IS_REAL else "inline stand-in table",
            "is_real_reference_data": REFERENCE_IS_REAL,
            "rows": len(REFERENCE),
            "distinct_mcc": int(REFERENCE["mcc"].nunique()),
        },
        "0": {
            "params": base,
            "post_fix": r0_metrics,
            "signal_auc": r0_auc,
            "utility": r0_test.utility,
        },
        "1": {
            "params": r1_params,
            "proposal": r1_proposal,
            "pre_fix": r1_pre,
            "post_fix": r1_post,
            "signal_auc": r1_auc,
            "utility": r1_test.utility,
        },
        "2": {
            "params": r2_params,
            "proposal": r2_proposal,
            "pre_fix": r2_pre,
            "post_fix": r2_post,
            "signal_auc": r2_auc,
            "utility": r2_test.utility,
            "build_amount_ks": build_ks,
            "budget": budget,
        },
    }
    _write_round(output_root, 0, r0_test, r0_scores, rounds["0"])
    _write_round(output_root, 1, r1_test, r1_scores, rounds["1"])
    _write_round(output_root, 2, r2_test, r2_scores, rounds["2"])
    (output_root / "a4_summary.json").write_text(
        json.dumps(rounds, indent=2, default=float), encoding="utf-8"
    )
    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three-round A4 experiment")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codex", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Override the reference marginals (defaults to data/processed/sparkov_reference.parquet)",
    )
    args = parser.parse_args()
    if args.reference is not None:
        global REFERENCE, REFERENCE_IS_REAL
        REFERENCE = load_reference(args.reference)
        REFERENCE_IS_REAL = len(REFERENCE) > len(FALLBACK_REFERENCE)
    if not REFERENCE_IS_REAL:
        print(
            "WARNING: using the inline stand-in reference table. Download the Sparkov "
            "source described in data/README.md for reportable fidelity.",
        )
    rounds = run_experiment(args.output, seed=args.seed, use_codex=args.codex, quick=args.quick)
    print(json.dumps(rounds, indent=2, default=float))


if __name__ == "__main__":
    main()
