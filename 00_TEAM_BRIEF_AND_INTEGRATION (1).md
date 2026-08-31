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
        │ RED AGENT — two stages      │───────────────────┘
        │  1. Codex reasons in words  │   next round
        │  2. CEM refines the numbers │
        └─────────────────────────────┘
```

The red agent only ever sees **approve or decline** plus its own attack's performance — exactly like a real fraudster probing a live system. It never sees model internals or weights.

### The red agent uses Codex as its LLM reasoner, not just an optimiser

This is what makes our system genuinely *AI* red teaming rather than parameter tuning, so understand it properly.

**Stage 1 — Reasoner (Codex).** After each round we hand Codex a **detector report**: which of our signals are working (per-feature AUC), what each attack parameter physically means, the current values, and a handful of caught-versus-missed examples. Codex writes back, in plain English, *why* the detector is catching the attack and *what a real fraud operator would change* — then proposes concrete parameter moves.

Codex runs locally through non-interactive `codex exec`, authenticated with the operator's ChatGPT subscription. We do **not** use OpenAI API keys or API credits. The Python loop sends the prompt on stdin, requests schema-constrained JSON, captures the final response, and caches it. OpenAI documents both [ChatGPT subscription sign-in for Codex](https://learn.chatgpt.com/docs/auth) and [`codex exec` for scripts](https://learn.chatgpt.com/docs/non-interactive-mode).

**Stage 2 — Refiner (CEM).** Cross-entropy method search tunes the numbers inside the region the reasoner proposed. Codex supplies the hypothesis; the optimiser finds the exact values.

Why both: Codex alone can give plausible-sounding but unmeasured guesses. CEM alone gives numbers with no explanation — it will discover that `mule_age_days = 45` works, but it will never say *why*, and it can only search directions somebody already parameterised. Together you get reasoned, measured, explainable attack evolution.

**The reasoning is a demo artifact.** Every round's output is saved to `results/round_{r}/redteam_reasoning.json` and gets its own dashboard screen. Judges watch the attacker explain its plan in English, then watch our detection collapse. That is far more compelling than a table of parameter deltas.

**Framing note:** this is an authorised red-team simulation operating on fully synthetic data, and Codex proposes bounded parameter values inside a pre-defined space. It is not generating attack tooling. Write the prompts that way, and say so if a judge asks.

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
| Red agent — reasoner | Codex CLI via `codex exec`, ChatGPT-authenticated, **cached** | proposes evasions in words without API credits |
| Red agent — refiner | Cross-Entropy Method, ~80 lines NumPy | no RL framework |
| Dashboard | Streamlit + Plotly | |
| LLM content | Codex CLI, **cached to JSON** | generated once during precomputation, never during judging |

**Explicitly skipped:** Kafka, GNNs, Docker, cloud deploy, RL frameworks. These earn zero marks and eat hours. The streaming architecture goes in the slides as a diagram, not in the repo.

**Hard rule:** no Codex invocation and no model training happens during the judging demo. Everything is precomputed to Parquet/JSON. The dashboard only replays and requires no credentials or network access.

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
│   ├── llm_agent.py       # STAGE 1 — Codex CLI reasoner adapter
│   ├── proposal.schema.json # strict JSON schema passed to codex exec
│   ├── prompts/
│   │   └── evasion.txt    # system + user prompt template
│   ├── report.py          # builds the detector report fed to Codex
│   ├── cem.py             # STAGE 2 — cross-entropy method refiner
│   └── realism.py         # realism + utility budget
├── models/
│   └── meta.py            # meta-learner that fuses heads
├── eval/
│   └── metrics.py         # AUC-PR, recall@FPR, per-feature AUC, LOAO
├── loop/
│   └── run_rounds.py      # orchestrator
├── app/
│   └── dashboard.py       # Streamlit
├── data/                  # reference dataset, cached model-generated content
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

    def describe_state(self, params: dict) -> dict:
        """Plain-English state, read by the Codex reasoner. Returns:
        {
          "attack_summary":  "one paragraph on how this attack works",
          "param_semantics": {param: "what it means physically, and what
                                      raising it costs the criminal"},
          "defender_signals": {feature: "what this feature measures"},
          "current_params":  params,
        }
        Codex cannot reason about `split_factor` unless you tell it that
        split_factor means 'how many mule accounts the payout is divided across'.
        """


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

### 5.4 The Codex red agent contract — built in Phase 0

Stubbed in Phase 0, completed by Person A in Phase 3. Everyone needs to know its two touchpoints.

**Touchpoint 1 — you must fill in `param_semantics` and `defender_signals` in your `card.json`.** Codex reads these. Without them it is reasoning about meaningless variable names.

```json
"param_semantics": {
  "mule_age_days": "How long the receiving accounts have existed before use. Higher = harder to spot as new, but pre-aged accounts cost more to buy.",
  "split_factor": "How many accounts the stolen money is divided across. Higher = lower per-account fan-in, but more mules to acquire and slower cash-out."
},
"defender_signals": {
  "a1_benef_account_age_days": "Age of the account receiving the money.",
  "a1_benef_fanin_24h": "How many distinct people paid this account in 24 hours."
}
```

**Touchpoint 2 — implement `describe_state()`** as shown in the interface above. Ten minutes of work, and it is checked by the contract test.

#### How Codex is invoked

Before precomputing rounds, install the Codex CLI and run `codex login`, choosing ChatGPT sign-in. `codex exec` reuses that saved subscription authentication. Do not create or request an API key for this project.

The adapter invokes Codex as a subprocess with argument-list APIs, never a shell string. Conceptually:

```bash
codex exec --ephemeral --sandbox read-only \
  --output-schema redteam/proposal.schema.json -
```

The rendered detector report is supplied on stdin. Progress goes to stderr and the final schema-constrained JSON response is captured from stdout. `--ephemeral` avoids persisting a session rollout, and `--sandbox read-only` prevents the reasoner from modifying the repository. The adapter records the Codex CLI version and selected model with the cached exchange so results can be audited.

This is a **precomputation dependency only**. A teammate without the authenticated Codex CLI can still run the numeric CEM fallback and replay committed/cached reasoning.

#### What the loop does with it

```python
report   = build_detector_report(attack, round_metrics, sample_events)
proposal = llm_agent.propose(report)      # stage 1 — codex exec + validated JSON
params   = cem.refine(proposal, attack, score_candidate)  # stage 2 — numeric search
events   = attack.generate(params, population, seed)
events   = realism.enforce(attack, events, reference_events)
```

Here `score_candidate(params) -> float` is supplied by the round orchestrator and rewards
detector evasion while keeping higher scores better. Calling `cem.refine()` without an objective
only returns the bounded proposal, which is useful for dry-run integration checks but does not
claim to have optimised it.

`llm_agent.propose()` returns strict JSON regardless of whether Codex or the fallback produced it:

```json
{
  "reasoning": "The detector is leaning on beneficiary account age (AUC 0.81) and 24-hour fan-in (0.78). Both describe the mule accounts, not the victim. A real network would buy pre-aged accounts and split the payout across more of them. Neither change reduces the money moved, so the crime stays worth doing.",
  "hypotheses": [
    {
      "signal_targeted": "a1_benef_account_age_days",
      "action": "raise mule_age_days substantially",
      "cost_to_attacker": "pre-aged mule accounts cost more to acquire"
    }
  ],
  "proposed_params": {"mule_age_days": 45, "split_factor": 15}
}
```

#### Guardrails — non-negotiable, Person A implements

1. Clamp every proposed value into `param_space`; silently drop unknown keys
2. If Codex is unavailable, exits non-zero, times out, or returns invalid JSON, log it and **fall back to pure CEM** — the loop never breaks because the reasoner failed
3. Run realism and utility budgets on the result; reject and re-prompt up to 3 times
4. **Cache the full exchange** to `results/round_{r}/redteam_reasoning.json`, including runner/model metadata. The demo replays this file. No live Codex invocation during judging, ever.

### 5.5 The column prefix rule

> **Every feature column you create must be prefixed with your attack id in lowercase.**
> `a1_component_size`, `a2_intent_cart_cosine`, `a3_bin_cooccurrence`, `a4_ramp_convexity`.

This single rule is why four people can write features independently and have them concatenate without a single collision. There is no exception to it.

### 5.6 Base population — built in Phase 0

`core/population.py` produces the honest traffic that everything is injected into.

- 20,000 customers, 2,000 merchants, 5,000 VPAs, 90 days of history
- Per-customer spend profile: log-normal amounts, personal merchant affinity
- Diurnal curve (evening peak), weekly cycle (weekend lift)
- Realistic MCC mix, realistic decline rate (~4%)
- **Marginals fitted to a real public dataset** — use Sparkov or IEEE-CIS from Kaggle. This is what makes our fidelity claims survive scrutiny. Do not invent distributions.

Target: roughly 3–4 million legit events. Fraud will be well under 1% of rows, as in reality.

### 5.7 Core features — built in Phase 0

`core/features.py` gives everyone, for free:

- rolling customer aggregates over 1h / 24h / 7d / 30d: count, sum, mean, std, distinct counterparties, distinct devices
- counterparty rolling aggregates over the same windows
- amount z-score vs the customer's own 30-day history
- hour-of-day, day-of-week, time-since-previous-transaction

You build attack-specific features **on top of** these. Do not reimplement them.

### 5.8 The contract test — your ticket to merge

`tests/test_contract.py` is written in Phase 0 and is parameterised over all four attacks. It asserts:

1. `generate()` returns every column in `REQUIRED_COLUMNS` with correct dtypes
2. `label` is only 0 or 1, and `attack_id` matches your own id on fraud rows
3. every column from your `FeatureBlock.build()` starts with your prefix
4. `generate()` with the same seed twice produces byte-identical output
5. `param_space()` keys exactly match the keys in your `card.json`
6. `score()` returns a float array in `[0, 1]` of the right length
7. `utility()` returns a float in `[0, 1]`
8. `describe_state()` returns all four required keys
9. `param_semantics` covers **every** key in `param_space` — no undocumented parameters
10. `defender_signals` covers every feature your `FeatureBlock` emits

```bash
pytest tests/test_contract.py -k a1   # run yours before every push
```

**If this test passes, your work will integrate. If it fails, it will not. There is no third outcome.**

### 5.9 Git rules

- Branch naming: `feat/a1`, `feat/a2`, `feat/a3`, `feat/a4`
- You touch **only** `attacks/a{N}_*/`. Nothing else.
- Need a change in `core/`? Post in the group chat and let the owner make it. Do not edit it yourself, do not work around it locally.
- Push at least every 4 hours so integration problems surface early instead of at hour 40.

### 5.10 Round output format — FROZEN

```
results/round_{r}/events.parquet          # all attacks + legit for that round
results/round_{r}/features.parquet
results/round_{r}/scores.parquet
results/round_{r}/metrics.json
results/round_{r}/redteam_reasoning.json   # Codex's written evasion plan
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
| **H0 – H5** | Phase 0: skeleton, schema, population, core features, contract test, red agent stub | all four, together |
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
Round 0 uses `default_params()`. Rounds 1 and 2 run the **two-stage red agent**: the ChatGPT-authenticated Codex reasoner reads each attack's detector report and writes an evasion plan, then CEM refines the numbers inside that region, subject to realism and utility budgets. Each person's round-1 replacement features are already in the codebase, so the recovery is real, not staged.

Person A also finishes `redteam/llm_agent.py` (~3 hours): a safe `codex exec` subprocess adapter, JSON Schema output, timeout/error handling, clamping, retry, CEM fallback, and caching to `redteam_reasoning.json`. Build the fallback path first — a working pure-CEM loop that you then upgrade is much safer than a Codex-backed loop that might fail at hour 40. Do not add the OpenAI SDK or request an API key.

### Step 4 — Meta-learner *(owner: Person D)*
Logistic regression stacking all detector heads plus the IsolationForest novelty score into one risk score. Report `fpr_at_recall_0.85` — how many honest customers we would block. A fraud system that annoys real people is useless, and judges know it.

### Step 5 — Leave-one-attack-out *(owner: Person D)*
Four folds. Train on three attacks, detect the fourth, never seen. This is the single result that proves we catch *novel* fraud rather than memorising our own. Also run it on round-2 evolved variants.

### Step 6 — Dashboard *(owner: Person B)*
Five screens, reading only precomputed Parquet:
1. **Attack Library** — the four cards with their JSON params
2. **Fidelity Proof** — distribution overlays, KS table, discriminator AUC
3. **Live Arena** — pick attack, replay a round, watch scores stream
4. **Red Agent Reasoning** — Codex's evasion plan for each attack, in its own words, rendered from `redteam_reasoning.json`. Show it *next to* the recall drop it caused.
5. **Signal Mortality** — features × rounds, cells going red as AUC decays toward 0.5, replacement signals appearing below
6. **Scoreboard** — recall curves, attacker cost frontier, LOAO results

Screens 4 and 5 are the money shot, and they work as a pair: screen 4 is the attacker saying what it will do, screen 5 is the damage. Give them the most polish.

### Step 7 — Writeup *(all four)*
Must include:
- the taxonomy of 40+ attack vectors on paper, with the four we implemented marked
- explicit framing: *four rails, four detection mechanisms, one engine that breaks all four and forces the defence to grow*
- the two honest caveats below

**Where GenAI sits in our system** — have this answer ready, someone will ask:

> "GenAI is the threat model in all four attacks — deepfake coercion, LLM-written prompt injection, agent-orchestrated carding, generated KYC documents. It also drives our red-teaming agent: Codex, invoked offline through the CLI with ChatGPT subscription access, reasons in natural language about how to defeat our own detector. What we deliberately do *not* use it for is generating the transaction rows themselves, because parametric simulation gives us verifiable distributional fidelity and reproducibility that LLM-generated tabular data cannot. We use GenAI where it is the right tool, not where it is decorative."

### The two caveats — state them yourselves before a judge does

**Self-play risk.** If the attacker and the generator share assumptions, we improve against our own imagination and learn nothing about reality. Mitigated by the realism budget and by fitting the base population to a real public dataset.

**Overfitting to our own red agent.** Mitigated by the leave-one-attack-out test on evolved variants.

Raising these unprompted reads as rigour. Being caught out on them reads as naivety.

---

## 8. Definition of done, per person

You are done when all seven are true:

1. `attacks/a{N}_*/card.json` exists with round-0 params, full param ranges, **`param_semantics` and `defender_signals` written in plain English**, and `describe_state()` implemented
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
3. One entire attack

**Do not cut the Codex reasoner.** If automated `codex exec` is failing at hour 40, run the same prompt once interactively in the authenticated Codex client for each attack, validate and paste the output into `redteam_reasoning.json`, and let the loop read the cached file. The reasoning is real either way — you just ran it manually instead of in-loop. Say that plainly if asked; nobody penalises a cached run, and the screen is worth far more than the automation.

**Never cut the loop.** Two attacks with a complete, working loop beats four attacks with no loop. The loop is what is being judged.
