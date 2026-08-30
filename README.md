# Mastercard GFF Red Team / Blue Team

Shared Phase 0 foundation for the adaptive payment-fraud simulation. The authoritative
contract is now `00_TEAM_BRIEF_AND_INTEGRATION (1).md`; it supersedes the original brief.

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

Attack owners must read `attacks/TEAMMATE_CONTRACT.md`. The updated contract requires
`describe_state()`, complete parameter/feature semantics in `card.json`, and exports of
`GENERATOR`, `FEATURE_BLOCK`, and `DETECTOR` from each attack package.

The red-team reasoner is provider-neutral: inject an API callable during precomputation,
then cache the exchange as `results/round_N/redteam_reasoning.json`. No live API call is
made by the judging dashboard.
