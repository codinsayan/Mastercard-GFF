# Data

**Every attack depends on this, not just A4.** `core/population.py` fits the base population's
amount and MCC marginals to a real reference set, and that population is the honest traffic all
four attacks are injected into. If a teammate skips this step their population silently falls
back to a small stand-in table and any fidelity claim they make is unsupported.

One command, once per clone:

```bash
python data/fetch_reference.py
```

It downloads the Sparkov transaction set (Apache-2.0, no Kaggle account needed) to `raw/` and
writes the normalized `amount`/`mcc` file to `processed/sparkov_reference.parquet`. Both
directories are git-ignored: the raw file is 338 MB and this repository does not redistribute
source datasets.

Put Sparkov or IEEE-CIS source files in `raw/` (ignored by Git). Generated, normalized
reference data belongs in `processed/`. Do not commit licensed or large source datasets.

`BasePopulation.from_path()` accepts CSV or Parquet. It recognizes Sparkov's `amt` and
`category` columns and IEEE-CIS's `TransactionAmt` and `ProductCD` columns, normalizing them
to the required `amount` and `mcc` inputs. All other source columns are deliberately ignored.

The repository does not redistribute either dataset. Download one from its publisher under
the publisher's terms, then verify the shared foundation with:

```bash
python -m loop.run_rounds --dry-run
python -m loop.run_rounds --dry-run --reference data/raw/<sparkov-or-ieee-file.csv>
```
