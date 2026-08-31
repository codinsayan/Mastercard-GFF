# Data

Put Sparkov or IEEE-CIS source files in `raw/` (ignored by Git). Generated, normalized
reference data belongs in `processed/`. Do not commit licensed or large source datasets.

`BasePopulation.from_path()` accepts CSV or Parquet. It recognizes Sparkov's `amt` and
`category` columns and IEEE-CIS's `TransactionAmt` and `ProductCD` columns, normalizing them
to the required `amount` and `mcc` inputs. All other source columns are deliberately ignored.

For A1 external validation, use either the Sparkov transaction CSV containing `amt`,
`category`, and `is_fraud`, or IEEE-CIS `train_transaction.csv` containing
`TransactionAmt`, `ProductCD`, and `isFraud`. The validator excludes source fraud rows so
only legitimate external amount/category marginals calibrate the shared population. It
records the source filename, detected format, row counts, and SHA-256 digest in the output.

IEEE-CIS is subject to Kaggle competition rules. Each teammate must sign in and accept
those rules before downloading it; do not commit or redistribute the source file.

```powershell
# Put exactly one authorized source file here, for example:
# data/raw/fraudTrain.csv
# data/raw/train_transaction.csv

python -m attacks.a1_upi_mule.experiment `
  --reference data/raw/fraudTrain.csv `
  --population-events 6000 `
  --output attacks/a1_upi_mule/results/iteration8_metrics_external.json
```

The repository does not redistribute either dataset. Download one from its publisher under
the publisher's terms, then verify the shared foundation with:

```bash
python -m loop.run_rounds --dry-run
python -m loop.run_rounds --dry-run --reference data/raw/<sparkov-or-ieee-file.csv>
```
