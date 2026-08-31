"""Fetch and normalize the shared public reference transaction set.

The base population in ``core/population.py`` fits its amount and MCC marginals to real
data, so every attack -- not just A4 -- depends on this file.

``data/processed/sparkov_reference.parquet`` is committed, so a normal clone needs no
setup. Run this script only to regenerate that file or to fetch the raw source:

    python data/fetch_reference.py

It downloads the Sparkov transaction set (Apache-2.0, no account required) to
``data/raw/`` and rewrites the normalized ``amount``/``mcc`` reference that
``BasePopulation`` consumes. The 338 MB raw file stays git-ignored; only the small
derived table is committed. See data/README.md for attribution.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://huggingface.co/datasets/pointe77/credit-card-transaction/"
    "resolve/main/credit_card_transaction_train.csv"
)
DATA_DIR = Path(__file__).resolve().parent
RAW_PATH = DATA_DIR / "raw" / "sparkov_fraudTrain.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "sparkov_reference.parquet"
EXPECTED_MINIMUM_ROWS = 1_000_000


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {destination}")

    def report(block: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(100.0, 100.0 * block * block_size / total)
            print(f"\r  {done:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()


def normalize(source: Path, destination: Path) -> pd.DataFrame:
    """Keep legitimate rows only; the base population models honest traffic."""
    frame = pd.read_csv(source, usecols=["amt", "category", "is_fraud"])
    legitimate = frame.loc[frame["is_fraud"].eq(0), ["amt", "category"]]
    legitimate = legitimate.rename(columns={"amt": "amount", "category": "mcc"})
    legitimate = legitimate.loc[legitimate["amount"].gt(0)].reset_index(drop=True)
    if len(legitimate) < EXPECTED_MINIMUM_ROWS:
        raise SystemExit(
            f"expected at least {EXPECTED_MINIMUM_ROWS:,} legitimate rows, got {len(legitimate):,}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    legitimate.to_parquet(destination, index=False)
    return legitimate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the raw file exists"
    )
    args = parser.parse_args()

    if RAW_PATH.exists() and not args.force:
        print(f"Raw file already present: {RAW_PATH} (use --force to re-download)")
    else:
        download(SOURCE_URL, RAW_PATH)

    reference = normalize(RAW_PATH, PROCESSED_PATH)
    print(
        f"Wrote {PROCESSED_PATH} "
        f"({len(reference):,} legitimate rows, {reference['mcc'].nunique()} categories)"
    )
    print("Verify the shared foundation with:")
    print(f"  python -m loop.run_rounds --dry-run --reference {PROCESSED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
