# 00 — TEAM BRIEF & INTEGRATION CONTRACT

**Read this file completely before writing a single line of code.**
Everyone reads this. Then you read only your own numbered file.

---

## 1. What we are building and why

### The hackathon

We are competing in the **Mastercard Innovation Challenge 2026**, hosted at the Global Fintech Fest (GFF) 2026 in Mumbai, 9–11 September.

The challenge is a **red team / blue team** problem. Normally a fraud hackathon asks you to build a fraud detector. This one asks you to do three things at once:

1. **Identify** — research how criminals are using Generative AI to attack payment systems.
2. **Generate** — write software that recreates those attacks as realistic payment data.
3. **Defend** — build a model that detects those attacks accurately, without falsely blocking honest customers.

We are judged on five things:

| Criterion | What it means for us |
|---|---|
| Diversity of attacks identified | How many genuinely different attack types we cover |
| Fidelity of attacks in simulation | Does our fake data actually look like real payment data |
| Detection efficacy | Precision, recall, F1, AUC on our attacks |
| Novelty | Is the approach original |
| Real-world feasibility | Could a bank actually run this |

### The thing that wins

Most teams will submit three disconnected pieces: a slide of attack ideas, a data generation script, and an XGBoost classifier. That reads as three assignments stapled together.

Our differentiator is the **closed loop**:

> The attacker adapts to the defender, and the defender improves because of it.

Concretely: we build attacks, we train a detector on them, then an automated **red agent** studies what the detector missed and mutates the attacks to be sneakier. The detector's old signals stop working. We then build *new* signals. We repeat this three times and show a chart of it happening.

That chart is our submission.

### The single most important concept: signal mortality

Every attack can be detected by several signals. Some signals are **fragile** — the attacker can defeat them cheaply. Some are **durable** — defeating them costs the criminal real money or destroys the point of the crime.

Round 0 detectors always sit on fragile signals, because those are the obvious ones.

The red agent's real job is to **kill our fragile signals for us**, forcing us onto durable ones we would never have prioritised on our own. That is literally what "the defender improves because of it" means.

So every person on this team produces the same artifact for their attack:

| Signal | Round 0 AUC | Round 1 AUC | Verdict |
|---|---|---|---|
| fragile_signal_x | 0.88 | 0.53 | **DEAD** |
| durable_signal_y | — | 0.84 | new, replaces above |

Four of these tables, one per attack, is the spine of our pitch.

---

## 2. The four attacks

Each person owns exactly one attack, end to end: the generator, the detector head, and the signal mortality story.

| Person | Attack | Rail / channel | Detector mechanism | File |
|---|---|---|---|---|
| **A** | Deepfake coercion → UPI mule cash-out | UPI real-time payments | Network graph | `01_PERSON_A_*.md` |
| **B** | Prompt-injected AI shopping agent | Agentic commerce | Semantic / NLP | `02_PERSON_B_*.md` |
| **C** | Agentic card testing | Card-not-present | Velocity + bipartite graph | `03_PERSON_C_*.md` |
| **D** | Synthetic identity bust-out | Onboarding / KYC + credit | Temporal sequence | `04_PERSON_D_*.md` |

This split is deliberate. Four different rails, four different detection mechanisms, **zero overlap**. That is what stops our model ensemble from looking like padding — each component exists because exactly one attack requires it.

---

## 3. Architecture

```
                     ┌─────────────────────────┐
                     │   BASE POPULATION       │
                     │  legit customers, cards │
                     │  merchants, VPAs, time  │
                     └───────────┬─────────────┘
                                 │
        ┌──────────┬─────────────┼─────────────┬──────────┐
        ▼          ▼             ▼             ▼          │
   ┌────────┐ ┌────────┐   ┌────────┐   ┌────────┐        │
   │  A1    │ │  A2    │   │  A3    │   │  A4    │        │
   │ UPI    │ │ agent  │   │ card   │   │ synth  │        │
   │ mule   │ │ inject │   │ test   │   │ ident  │        │
   └───┬────┘ └───┬────┘   └───┬────┘   └───┬────┘        │
       └──────────┴─────────────┴───────────┘             │
                       │                                  │
                       ▼                                  │
             ┌──────────────────┐                         │
             │ SHARED EVENT LOG │  (one schema, all rows) │
             └────────┬─────────┘                         │
                      ▼                                   │
             ┌──────────────────┐                         │
             │ FEATURE BUILDER  │  core + per-attack      │
             └────────┬─────────┘                         │
                      ▼                                   │
        ┌─────────────────────────────┐                   │
        │ DETECTOR ENSEMBLE           │                   │
        │ LGBM · GRU · IForest ·      │                   │
        │ semantic · meta-learner     │                   │
        └────────┬────────────────────┘                   │
                 │ what got missed                        │
                 ▼                                        │
        ┌─────────────────────────────┐                   │
        │ RED AGENT (CEM search)      │───────────────────┘
        │ mutates attack params       │   next round
        └─────────────────────────────┘
```

The red agent only ever sees **approve or decline** — exactly like a real fraudster probing a live system. It never sees model internals.

Two constraints keep it honest, and both must be stated in our pitch:

- **Realism budget** — a mutation is rejected if the resulting data drifts too far from real-world distributions. Without this, the attacker just cheats into impossible territory and our improvements are fake.
- **Utility budget** — a mutation is rejected if the attack no longer achieves its criminal goal (too few cards validated, too little money moved). Without this, "evasion" includes simply not committing fraud.

---

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Data | pandas, NumPy, PyArrow | |
| Storage | Parquet + DuckDB | fast replay for the demo |
| Tabular model | LightGBM | trains in seconds, wins on tabular |
| Sequence model | PyTorch (small GRU) | Person D only |
| Novelty detection | scikit-learn IsolationForest | |
| Semantic | `sentence-transformers`, `all-MiniLM-L6-v2` | Person B only |
| Graph | NetworkX | Persons A and C |
| Red agent | Cross-Entropy Method, ~80 lines NumPy | no RL framework |
| Dashboard | Streamlit + Plotly | |
| LLM content | any API, **cached to JSON** | generated once, never live |

**Explicitly skipped:** Kafka, GNNs, Docker, cloud deploy, RL frameworks. These earn zero marks and eat hours. The streaming architecture goes in the slides as a diagram, not in the repo.

**Hard rule:** no LLM API call and no model training happens during the judging demo. Everything is precomputed to Parquet. The dashboard only replays.

```bash
pip install pandas numpy pyarrow duckdb lightgbm scikit-learn \
            networkx sentence-transformers torch streamlit plotly pytest
```

---

## 5. PREREQUISITE — Phase 0, before anyone splits off

**Do not start your individual attack until Phase 0 is merged to `main`.**

Phase 0 takes about 3–4 hours. Sit together for it. One person types, others review. Everything below is shared property — **once merged, nobody edits these files without telling the whole team.**

### 5.1 Repo skeleton

```
mc-redteam/
├── core/
│   ├── schema.py          # column contract + validator
│   ├── population.py      # base legit population generator
│   ├── features.py        # CORE feature builder (shared)
│   └── registry.py        # attack registration
├── attacks/
│   ├── a1_upi_mule/       # PERSON A owns this folder only
│   ├── a2_agent_inject/   # PERSON B
│   ├── a3_card_testing/   # PERSON C
│   └── a4_synth_identity/ # PERSON D
├── redteam/
│   ├── cem.py             # cross-entropy method search
│   └── realism.py         # realism + utility budget
├── models/
│   └── meta.py            # meta-learner that fuses heads
├── eval/
│   └── metrics.py         # AUC-PR, recall@FPR, per-feature AUC, LOAO
├── loop/
│   └── run_rounds.py      # orchestrator
├── app/
│   └── dashboard.py       # Streamlit
├── data/                  # reference dataset, cached LLM content
├── results/               # round outputs (parquet + json)
└── tests/
    └── test_contract.py   # THE integration guarantee
```

### 5.2 The event schema — FROZEN, do not change

`core/schema.py`:

```python
REQUIRED_COLUMNS = {
    # identity of the event
    "event_id":        "string",
    "timestamp":       "datetime64[ns]",
    "attack_id":       "string",    # "LEGIT" | "A1" | "A2" | "A3" | "A4"
    "label":           "int8",      # 0 = legit, 1 = fraud
    "round":           "int8",      # which loop round produced this

    # payer side
    "customer_id":              "string",
    "customer_account_age_days":"int32",
    "device_id":                "string",
    "ip_asn":                   "string",
    "session_id":               "string",
    "session_duration_s":       "int32",
    "channel":                  "string",  # CNP|UPI_P2P|UPI_P2M|AGENTIC|CARD_PRESENT

    # instrument (nullable per channel)
    "card_id":   "string",
    "card_bin":  "string",
    "vpa":       "string",
    "token_id":  "string",

    # receiving side
    "counterparty_id":              "string",  # merchant_id OR beneficiary_id
    "counterparty_type":            "string",  # MERCHANT | BENEFICIARY
    "counterparty_mcc":             "string",
    "counterparty_account_age_days":"int32",

    # money
    "amount":             "float64",
    "is_first_time_payee":"int8",

    # outcome
    "auth_result": "string",   # APPROVED | DECLINED
    "decline_code":"string",

    # escape hatch for attack-specific fields
    "meta": "object",          # dict, serialised to JSON on write
}
```

**Why `meta` exists:** your attack needs fields nobody else needs (call-active flag, cart contents, BIN dump id, credit utilisation). Put them in `meta` as a dict. You never have to change the shared schema, so you never have to coordinate with anyone.

`core/schema.py` also exposes:

```python
def validate(df: pd.DataFrame) -> None:
    """Raises if any required column is missing or wrongly typed."""
```

Call it at the end of your generator. Always.

### 5.3 The four interfaces — FROZEN

```python
# core/registry.py
from typing import Protocol
import pandas as pd, numpy as np

class AttackGenerator(Protocol):
    attack_id: str                       # "A1" ... "A4"

    def default_params(self) -> dict:
        """Round-0 parameter values."""

    def param_space(self) -> dict[str, tuple[float, float]]:
        """{param_name: (low, high)} — the red agent searches inside this."""

    def generate(self, params: dict, population, seed: int) -> pd.DataFrame:
        """Must return rows matching REQUIRED_COLUMNS. Deterministic given seed."""

    def utility(self, df: pd.DataFrame) -> float:
        """0..1 — is the crime still worth committing at these params?
        Used by the utility budget. E.g. A3 = cards validated per hour, normalised."""


class FeatureBlock(Protocol):
    attack_id: str

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        """Returns a frame indexed by event_id.
        EVERY column name MUST start with the attack_id in lowercase: a1_, a2_, ...
        """


class DetectorHead(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None: ...

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns float array in [0, 1], same length as X."""
```

### 5.4 The column prefix rule

> **Every feature column you create must be prefixed with your attack id in lowercase.**
> `a1_component_size`, `a2_intent_cart_cosine`, `a3_bin_cooccurrence`, `a4_ramp_convexity`.

This single rule is why four people can write features independently and have them concatenate without a single collision. There is no exception to it.

### 5.5 Base population — built in Phase 0

`core/population.py` produces the honest traffic that everything is injected into.

- 20,000 customers, 2,000 merchants, 5,000 VPAs, 90 days of history
- Per-customer spend profile: log-normal amounts, personal merchant affinity
- Diurnal curve (evening peak), weekly cycle (weekend lift)
- Realistic MCC mix, realistic decline rate (~4%)
- **Marginals fitted to a real public dataset** — use Sparkov or IEEE-CIS from Kaggle. This is what makes our fidelity claims survive scrutiny. Do not invent distributions.

Target: roughly 3–4 million legit events. Fraud will be well under 1% of rows, as in reality.

### 5.6 Core features — built in Phase 0

`core/features.py` gives everyone, for free:

- rolling customer aggregates over 1h / 24h / 7d / 30d: count, sum, mean, std, distinct counterparties, distinct devices
- counterparty rolling aggregates over the same windows
- amount z-score vs the customer's own 30-day history
- hour-of-day, day-of-week, time-since-previous-transaction

You build attack-specific features **on top of** these. Do not reimplement them.

### 5.7 The contract test — your ticket to merge

`tests/test_contract.py` is written in Phase 0 and is parameterised over all four attacks. It asserts:

1. `generate()` returns every column in `REQUIRED_COLUMNS` with correct dtypes
2. `label` is only 0 or 1, and `attack_id` matches your own id on fraud rows
3. every column from your `FeatureBlock.build()` starts with your prefix
4. `generate()` with the same seed twice produces byte-identical output
5. `param_space()` keys exactly match the keys in your `card.json`
6. `score()` returns a float array in `[0, 1]` of the right length
7. `utility()` returns a float in `[0, 1]`

```bash
pytest tests/test_contract.py -k a1   # run yours before every push
```

**If this test passes, your work will integrate. If it fails, it will not. There is no third outcome.**

### 5.8 Git rules

- Branch naming: `feat/a1`, `feat/a2`, `feat/a3`, `feat/a4`
- You touch **only** `attacks/a{N}_*/`. Nothing else.
- Need a change in `core/`? Post in the group chat and let the owner make it. Do not edit it yourself, do not work around it locally.
- Push at least every 4 hours so integration problems surface early instead of at hour 40.

### 5.9 Round output format — FROZEN

```
results/round_{r}/events.parquet          # all attacks + legit for that round
results/round_{r}/features.parquet
results/round_{r}/scores.parquet
results/round_{r}/metrics.json
```

`metrics.json` shape:

```json
{
  "round": 1,
  "per_attack": {
    "A1": {"recall": 0.41, "precision": 0.88, "auc_pr": 0.52,
           "recall_at_fpr_0.001": 0.33, "utility": 0.71}
  },
  "signal_auc": {
    "a1_benef_account_age_days": 0.55,
    "a1_fanin_24h": 0.58
  },
  "overall": {"auc_pr": 0.61, "fpr_at_recall_0.85": 0.021}
}
```

---

## 6. Timeline (48 hours)

| Window | What happens | Who |
|---|---|---|
| **H0 – H4** | Phase 0: skeleton, schema, population, core features, contract test | all four, together |
| **H4 – H28** | Individual attack builds | each person, alone |
| **H28 – H36** | Integration, meta-learner, run all three rounds | A + D |
| **H36 – H44** | Dashboard, fidelity metrics, LOAO evaluation | B + C |
| **H44 – H48** | Writeup, slides, dry run | all four |

**Checkpoint at H16 (mandatory):** every person demos their round-0 generator producing valid rows that pass the contract test. Anyone not there gets help immediately, not at hour 30.

---

## 7. POST-REQUISITE — Phase 3, after the four branches merge

Do these **in this order**. Each depends on the one before.

### Step 1 — Merge and verify
```bash
pytest tests/test_contract.py      # all four attacks must pass
python -m loop.run_rounds --round 0 --dry-run
```

### Step 2 — Fidelity proof *(owner: Person C)*
Compare the base population against the real reference dataset:
- Kolmogorov–Smirnov statistic per feature (target: most below 0.1)
- correlation matrix distance between synthetic and real
- **discriminator test:** train LightGBM to tell synthetic from real. Target AUC ≈ 0.5, meaning it cannot.

Write to `results/fidelity.json`. This screen goes *first* in the demo — establish trust before showing anything else.

### Step 3 — Run the loop *(owner: Person A)*
```bash
python -m loop.run_rounds --rounds 0 1 2 --seed 42
```
Round 0 uses `default_params()`. Rounds 1 and 2 use CEM-mutated params, subject to realism and utility budgets. Each person's round-1 replacement features are already in the codebase, so the recovery is real, not staged.

### Step 4 — Meta-learner *(owner: Person D)*
Logistic regression stacking all detector heads plus the IsolationForest novelty score into one risk score. Report `fpr_at_recall_0.85` — how many honest customers we would block. A fraud system that annoys real people is useless, and judges know it.

### Step 5 — Leave-one-attack-out *(owner: Person D)*
Four folds. Train on three attacks, detect the fourth, never seen. This is the single result that proves we catch *novel* fraud rather than memorising our own. Also run it on round-2 evolved variants.

### Step 6 — Dashboard *(owner: Person B)*
Five screens, reading only precomputed Parquet:
1. **Attack Library** — the four cards with their JSON params
2. **Fidelity Proof** — distribution overlays, KS table, discriminator AUC
3. **Live Arena** — pick attack, replay a round, watch scores stream
4. **Signal Mortality** — features × rounds, cells going red as AUC decays toward 0.5, replacement signals appearing below
5. **Scoreboard** — recall curves, attacker cost frontier, LOAO results

Screen 4 is the money shot. Give it the most polish.

### Step 7 — Writeup *(all four)*
Must include:
- the taxonomy of 40+ attack vectors on paper, with the four we implemented marked
- explicit framing: *four rails, four detection mechanisms, one engine that breaks all four and forces the defence to grow*
- the two honest caveats below

### The two caveats — state them yourselves before a judge does

**Self-play risk.** If the attacker and the generator share assumptions, we improve against our own imagination and learn nothing about reality. Mitigated by the realism budget and by fitting the base population to a real public dataset.

**Overfitting to our own red agent.** Mitigated by the leave-one-attack-out test on evolved variants.

Raising these unprompted reads as rigour. Being caught out on them reads as naivety.

---

## 8. Definition of done, per person

You are done when all seven are true:

1. `attacks/a{N}_*/card.json` exists with round-0 params and full param ranges
2. `generator.py` produces schema-valid, seed-deterministic rows
3. `features.py` produces round-0 **and** round-1 features, all correctly prefixed
4. `detector.py` fits and scores through the `DetectorHead` interface
5. `pytest tests/test_contract.py -k a{N}` passes
6. `README.md` in your folder contains your **signal mortality table** with real measured AUCs
7. You can explain your attack out loud in 60 seconds to a non-technical judge

Number 7 is not optional. Any of us might be the one standing in front of the judges.

---

## 9. If we run out of time

Cut in this order:

1. Round 2 (ship a two-round loop)
2. Person D's GRU (fall back to hand-engineered trajectory features)
3. Person B's live LLM content (use pre-generated fixtures)
4. One entire attack

**Never cut the loop.** Two attacks with a complete, working loop beats four attacks with no loop. The loop is what is being judged.
