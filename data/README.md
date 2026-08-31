# Data

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
