"""Round orchestration entry point."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive fraud simulation rounds")
    parser.add_argument("--round", type=int)
    parser.add_argument("--rounds", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = args.rounds if args.rounds is not None else [args.round or 0]
    if args.dry_run:
        print(f"Phase 0 ready; selected rounds={selected}, seed={args.seed}")
        return
    raise SystemExit("No attacks are registered yet; merge A1-A4 before running rounds")


if __name__ == "__main__":
    main()

