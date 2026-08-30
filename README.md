# Mastercard GFF Red Team / Blue Team

Shared Phase 0 foundation for the adaptive payment-fraud simulation described in
`00_TEAM_BRIEF_AND_INTEGRATION.md`.

## Setup

Python 3.11 or 3.12 is required (3.13+ is intentionally excluded because the ML stack may lag it).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

Place the public Sparkov or IEEE-CIS reference transaction files under `data/raw/`.
Raw data and generated Parquet outputs are intentionally ignored by Git.

