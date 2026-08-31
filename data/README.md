# Data

**Every attack depends on this, not just A4.** `core/population.py` fits the base population's
amount and MCC marginals to a real reference set, and that population is the honest traffic all
four attacks are injected into. Without a real reference the population falls back to a small
stand-in table and every fidelity claim built on it is unsupported -- silently, with no error.

**`processed/sparkov_reference.parquet` is committed to the repository.** Clone the repo and
you already have it -- no setup step, and every teammate is fitted to a byte-identical
reference. Do not delete it.

To regenerate it, or to refresh the 338 MB raw source:

```bash
python data/fetch_reference.py
```

It downloads the Sparkov transaction set to `raw/` and rewrites the normalized `amount`/`mcc`
file in `processed/`. The raw source stays git-ignored; only the 3.4 MB derived table is
committed, because a missing copy degrades all four attacks silently rather than loudly.

### Attribution

`processed/sparkov_reference.parquet` is derived from the Sparkov credit-card transaction set,
obtained from the `pointe77/credit-card-transaction` mirror on Hugging Face and distributed
under the Apache License 2.0. The committed file is a modified subset: legitimate rows only,
reduced to the `amount` and `mcc` columns, with Sparkov's `amt` and `category` renamed. The
original is unmodified in every other respect and is not redistributed here.

Put any additional source files, such as IEEE-CIS, in `raw/`, which is ignored by Git. Do not
commit large or licensed source datasets; commit only small derived reference tables, and only
with the attribution their licence requires.

`BasePopulation.from_path()` accepts CSV or Parquet. It recognizes Sparkov's `amt` and
`category` columns and IEEE-CIS's `TransactionAmt` and `ProductCD` columns, normalizing them
to the required `amount` and `mcc` inputs. All other source columns are deliberately ignored.

The committed Sparkov reference is a small derived subset, attributed above. Full source
datasets are not redistributed here; download those from their publisher under the publisher's
terms. Verify the shared foundation with:

```bash
python -m loop.run_rounds --dry-run
python -m loop.run_rounds --dry-run --reference data/processed/sparkov_reference.parquet
```
