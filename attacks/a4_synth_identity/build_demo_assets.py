"""Distil the full round outputs into one small file the judge demo can replay.

``results/a4/round_*/events.parquet`` is git-ignored and runs to megabytes, so a fresh
clone has nothing for the demo to draw. This script reduces those rounds to the handful
of series the screens actually plot and writes them to ``results/a4/demo_assets.json``,
which is small enough to commit. Run it after every experiment run:

    python -m attacks.a4_synth_identity.build_demo_assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "a4"
OUTPUT_NAME = "demo_assets.json"
ROUNDS = ("0", "1", "2")
COHORT_ACCOUNTS = 40
SCORE_BINS = 20


def _meta(frame: pd.DataFrame, key: str) -> pd.Series:
    return frame["meta"].map(lambda value, item=key: value.get(item))


def _decorate(events: pd.DataFrame) -> pd.DataFrame:
    """Lift the per-event credit metadata out of ``meta`` into real columns."""
    out = events.copy()
    out["day"] = pd.to_numeric(_meta(out, "a4_days_since_onboarding"), errors="coerce")
    out["utilization"] = pd.to_numeric(_meta(out, "a4_utilization"), errors="coerce")
    out["payment_ratio"] = pd.to_numeric(_meta(out, "a4_payment_ratio"), errors="coerce")
    out["phase"] = _meta(out, "a4_phase")
    out["cluster"] = _meta(out, "a4_identity_cluster_id")
    return out.dropna(subset=["day", "utilization"])


def _series(group: pd.DataFrame) -> list[list[float]]:
    ordered = group.sort_values("day")
    return [
        [float(day), round(float(use), 4)]
        for day, use in zip(ordered["day"], ordered["utilization"], strict=True)
    ]


def build_hook(events: pd.DataFrame) -> dict[str, Any]:
    """The innocent-looking build rows, plus the full arc of the account they belong to.

    Rows are sampled across the build phase and the bust-out is appended, so the table
    shows the contrast the pitch depends on: three ordinary purchases labelled fraud,
    then the draw-down that makes them fraud in retrospect.
    """
    fraud = events[events["attack_id"].eq("A4")]
    build = fraud[fraud["phase"].eq("BUILD")]
    if build.empty:
        raise SystemExit("no build rows found; run the experiment first")
    counts = build.groupby("customer_id").size()
    account = str(counts.idxmax())
    arc = fraud[fraud["customer_id"].eq(account)].sort_values("day")
    account_build = arc[arc["phase"].eq("BUILD")]
    positions = [int(len(account_build) * fraction) for fraction in (0.15, 0.5, 0.85)]
    limit = len(account_build) - 1
    sampled = account_build.iloc[sorted({min(position, limit) for position in positions})]
    bustout = arc[arc["phase"].eq("BUSTOUT")].tail(1)
    rows = pd.concat([sampled, bustout])
    return {
        "account_id": str(account),
        "rows": [
            {
                "event_id": str(row.event_id),
                "timestamp": str(row.timestamp.date()),
                "day": int(row.day),
                "phase": str(row.phase),
                "utilization": round(float(row.utilization), 4),
                "payment_ratio": round(float(row.payment_ratio), 4),
                "amount": round(float(row.amount), 2),
                "label": int(row.label),
            }
            for row in rows.itertuples()
        ],
        "trajectory": [
            {
                "day": int(row.day),
                "utilization": round(float(row.utilization), 4),
                "payment_ratio": round(float(row.payment_ratio), 4),
                "amount": round(float(row.amount), 2),
                "phase": str(row.phase),
            }
            for row in arc.itertuples()
        ],
    }


def build_cohort(events: pd.DataFrame) -> dict[str, Any]:
    """One synthetic cluster against a comparable sample of organic accounts."""
    fraud = events[events["attack_id"].eq("A4")]
    legit = events[events["attack_id"].eq("LEGIT")]
    cluster = fraud["cluster"].value_counts().idxmax()
    members = fraud[fraud["cluster"].eq(cluster)]
    chosen = sorted(members["customer_id"].unique())[:COHORT_ACCOUNTS]
    organic_ids = sorted(legit["customer_id"].unique())[:COHORT_ACCOUNTS]
    return {
        "cluster_id": str(cluster),
        "synthetic": [
            {"customer_id": str(cid), "points": _series(members[members["customer_id"].eq(cid)])}
            for cid in chosen
        ],
        "organic": [
            {"customer_id": str(cid), "points": _series(legit[legit["customer_id"].eq(cid)])}
            for cid in organic_ids
        ],
    }


def build_spike_days(events: pd.DataFrame) -> dict[str, list[int]]:
    """The day each account first crosses 85% utilization -- the synchronisation picture."""
    spikes = events[events["utilization"].ge(0.85)]
    first = spikes.groupby(["attack_id", "customer_id"])["day"].min().reset_index()
    return {
        "synthetic": sorted(int(v) for v in first.loc[first["attack_id"].eq("A4"), "day"]),
        "organic": sorted(int(v) for v in first.loc[first["attack_id"].eq("LEGIT"), "day"]),
    }


def build_score_histogram(events: pd.DataFrame, scores: pd.DataFrame) -> dict[str, Any]:
    """Pre-binned account-level score distribution, so the demo does no maths on stage."""
    merged = events[["event_id", "customer_id", "label"]].merge(scores, on="event_id", how="inner")
    per_account = merged.groupby("customer_id").agg(
        label=("label", "max"), score=("a4_score", "max")
    )
    edges = np.linspace(0.0, 1.0, SCORE_BINS + 1)
    fraud, _ = np.histogram(per_account.loc[per_account["label"].eq(1), "score"], bins=edges)
    legit, _ = np.histogram(per_account.loc[per_account["label"].eq(0), "score"], bins=edges)
    return {
        "bin_left": [round(float(v), 3) for v in edges[:-1]],
        "fraud": [int(v) for v in fraud],
        "legit": [int(v) for v in legit],
    }


def build(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    summary_path = root / "a4_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}; run the experiment first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    rounds: dict[str, Any] = {}
    hook: dict[str, Any] | None = None
    cohort: dict[str, Any] | None = None
    for name in ROUNDS:
        events_path = root / f"round_{name}" / "events.parquet"
        scores_path = root / f"round_{name}" / "scores.parquet"
        if not events_path.exists():
            raise SystemExit(f"missing {events_path}; run the experiment first")
        events = _decorate(pd.read_parquet(events_path))
        rounds[name] = {
            "spike_days": build_spike_days(events),
            "score_histogram": build_score_histogram(events, pd.read_parquet(scores_path)),
            "accounts": {
                "synthetic": int(events.loc[events["attack_id"].eq("A4"), "customer_id"].nunique()),
                "organic": int(events.loc[events["attack_id"].eq("LEGIT"), "customer_id"].nunique()),
            },
        }
        if name == "0":
            hook = build_hook(events)
            cohort = build_cohort(events)

    return {
        "note": "Precomputed for replay. Regenerate with build_demo_assets.py after any run.",
        "reference": summary.get("reference", {}),
        "hook": hook,
        "cohort": cohort,
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    assets = build(args.root)
    destination = args.root / OUTPUT_NAME
    destination.write_text(json.dumps(assets, separators=(",", ":")), encoding="utf-8")
    size_kb = destination.stat().st_size / 1024
    print(f"Wrote {destination} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
