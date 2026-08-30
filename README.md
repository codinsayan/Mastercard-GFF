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

The red-team reasoner will run through the Codex CLI with ChatGPT subscription authentication;
it does not use OpenAI API credits. Sign in locally with `codex login`, then the Python
orchestrator will invoke `codex exec` in read-only, non-interactive mode with a JSON output
schema. The full exchange is cached as `results/round_N/redteam_reasoning.json`, and the
judging dashboard never invokes Codex or requires credentials.

This follows OpenAI's documented [ChatGPT subscription authentication](https://learn.chatgpt.com/docs/auth)
and [`codex exec` non-interactive workflow](https://learn.chatgpt.com/docs/non-interactive-mode).
