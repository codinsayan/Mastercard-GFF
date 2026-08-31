"""Round orchestration entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.features import build_core_features
from core.population import BasePopulation, PopulationConfig
from core.schema import validate


def check_phase0(seed: int = 42, reference_path: str | Path | None = None) -> dict[str, int]:
    """Exercise the shared schema, population, features, and proposal schema."""
    schema_path = Path(__file__).resolve().parents[1] / "redteam" / "proposal.schema.json"
    proposal_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if set(proposal_schema.get("required", [])) != {
        "reasoning",
        "hypotheses",
        "proposed_params",
    }:
        raise RuntimeError("Codex proposal schema is incomplete")

    config = PopulationConfig(customers=50, merchants=25, vpas=40, days=7, events=250)
    if reference_path is None:
        reference = pd.DataFrame(
            {
                "amount": [8.5, 19.0, 42.0, 115.0, 699.0],
                "mcc": ["5411", "5812", "5812", "5732", "5999"],
            }
        )
        population = BasePopulation(reference, config)
    else:
        population = BasePopulation.from_path(reference_path, config)
    events = population.generate(seed=seed)
    validate(events)
    features = build_core_features(events)
    if len(features) != len(events) or not features.index.equals(pd.Index(events["event_id"])):
        raise RuntimeError("core features are not aligned to event_id")
    return {"events": len(events), "features": len(features.columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive fraud simulation rounds")
    parser.add_argument("--round", type=int)
    parser.add_argument("--rounds", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", type=Path)
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
    raise SystemExit("No attacks are registered yet; merge A1-A4 before running rounds")


if __name__ == "__main__":
    main()
