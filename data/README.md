# Data

**Every attack depends on this, not just A4.** `core/population.py` fits the base
population's amount and MCC marginals to an external reference set, and that population
is the honest traffic into which all four attacks are injected. Without the committed
reference, the population falls back to a small stand-in table and fidelity claims silently
lose their external calibration.

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

### Attribution and licensing

`processed/sparkov_reference.parquet` is derived from the Sparkov credit-card transaction set,
obtained from the `pointe77/credit-card-transaction` mirror on Hugging Face and distributed
under the Apache License 2.0. The committed file is a modified subset: legitimate rows only,
reduced to the `amount` and `mcc` columns, with Sparkov's `amt` and `category` renamed. The
raw source is not redistributed here. Sparkov is synthetic external reference data, not
real customer transaction data.

Put any additional source files, such as IEEE-CIS, in `raw/`, which is ignored by Git. Do not
commit large or licensed source datasets; commit only small derived reference tables, and
only with the attribution their licence requires.

`BasePopulation.from_path()` accepts CSV or Parquet. It recognizes Sparkov's `amt` and
`category` columns and IEEE-CIS's `TransactionAmt` and `ProductCD` columns, normalizing them
to the required `amount` and `mcc` inputs. All other source columns are deliberately ignored.

For A1 external validation, use the committed normalized Sparkov Parquet, a Sparkov CSV
containing `amt`, `category`, and `is_fraud`, or IEEE-CIS `train_transaction.csv` containing
`TransactionAmt`, `ProductCD`, and `isFraud`. The validator excludes source fraud rows so
only legitimate external amount/category marginals calibrate the shared population. It
records the source filename, detected format, row counts, and SHA-256 digest in the output.

IEEE-CIS is subject to Kaggle competition rules. Each teammate must sign in and accept
those rules before downloading it; do not commit or redistribute the source file.

```powershell
python -m attacks.a1_upi_mule.experiment `
  --reference data/processed/sparkov_reference.parquet `
  --population-events 6000 `
  --output attacks/a1_upi_mule/results/iteration8_metrics_external.json
```

The repository does not redistribute the raw Sparkov or IEEE-CIS datasets. Download raw
sources only under their publisher's terms, then verify the shared foundation with:

```bash
python -m loop.run_rounds --dry-run
python -m loop.run_rounds --dry-run --reference data/processed/sparkov_reference.parquet
```
