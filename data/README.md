# Data

Put Sparkov or IEEE-CIS source files in `raw/` (ignored by Git). Generated, normalized
reference data belongs in `processed/`. Do not commit licensed or large source datasets.
The small derived `processed/sparkov_reference.parquet` is intentionally committed so
every teammate and the replay demo use a byte-identical external reference.

### Committed Sparkov reference

`processed/sparkov_reference.parquet` contains 1,289,169 legitimate Sparkov transactions
reduced to `amount` and `mcc`. It is derived from the Sparkov credit-card transaction set
hosted by `pointe77/credit-card-transaction` on Hugging Face under Apache License 2.0.
The raw source remains ignored and is not redistributed. Run `python
data/fetch_reference.py` only when intentionally regenerating the derived artifact.

`BasePopulation.from_path()` accepts CSV or Parquet. It recognizes Sparkov's `amt` and
`category` columns and IEEE-CIS's `TransactionAmt` and `ProductCD` columns, normalizing them
to the required `amount` and `mcc` inputs. All other source columns are deliberately ignored.

For A1 external validation, use the committed normalized Sparkov Parquet, a Sparkov CSV containing `amt`,
`category`, and `is_fraud`, or IEEE-CIS `train_transaction.csv` containing
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
