# PERSON A — Attack A1: Deepfake Coercion → UPI Mule Cash-Out

**Folder you own:** `attacks/a1_upi_mule/`
**Feature prefix:** `a1_`
**Branch:** `feat/a1`
**Read `00_TEAM_BRIEF_AND_INTEGRATION (1).md` first.**

---

## 1. Context in 60 seconds

We are in the Mastercard Innovation Challenge at Global Fintech Fest 2026 (Mumbai, 9–11 September). The challenge asks us to play both sides: invent GenAI-powered payment fraud, simulate it realistically, then detect it.

The winning idea is a **closed loop**. We build attacks, train a detector, then an automated red agent mutates the attacks until our detector stops working. We then build better signals. Three rounds of that, plotted on a chart, is our submission.

You own one attack end to end: generator, detector, and the story of which signals died and what replaced them.

---

## 2. Your attack, in plain language

A victim gets a video call from someone who looks and sounds like a police officer. It is a deepfake. The caller says the victim's Aadhaar or bank account is linked to a criminal investigation and that they must transfer money immediately to a "verification account" or face arrest.

The victim panics and transfers the money themselves, from their own phone, using their own PIN and their own biometrics.

The money lands in a **mule account** — a bank account or UPI ID controlled by the fraud network. Within minutes it is split across more mule accounts, then withdrawn.

### Why this is a great pick for these judges

This is happening at enormous scale in India right now. Over **92,000 "digital arrest" cases** have been reported. In 2025, Indians lost roughly **₹22,495 crore across 2.81 million complaints**, a 24% year-on-year jump in volume. On the cash-out side, **524,121 suspected mule accounts and identities were flagged in March 2026 alone**, of which 520,559 were UPI VPAs — and payments banks accounted for about 41% of suspicious activity because of their low-friction onboarding.

Every person in that hall in Mumbai knows someone affected by this. **Your attack is the emotional anchor of our demo — it goes first.**

### Why it is technically hard, and why that's the point

Look at what the payment system actually sees:

- Real customer ✓
- Their own registered device ✓
- Correct PIN and biometric ✓
- Normal location ✓
- Transaction willingly initiated ✓

**Every payer-side authentication signal is clean.** The customer really did authorise it. This is why traditional fraud detection fails here completely.

So the detection has to happen on the **receiving side**. Who is getting this money, and does that account look like a real person's account?

---

## 3. What you generate

Channel: `UPI_P2P`.

### The structure

```
  VICTIM                 LAYER 1 MULES        LAYER 2 MULES       CASH-OUT
                        ┌──────────┐         ┌──────────┐
  ┌────────┐    ₹2.4L   │ mule_01  │  ₹80k   │ mule_11  │
  │victim_1├───────────▶│          ├────────▶│          ├───┐
  └────────┘            └──────────┘         └──────────┘   │   ┌──────────┐
                        ┌──────────┐         ┌──────────┐   ├──▶│cashout_1 │
  ┌────────┐    ₹1.8L   │ mule_02  │  ₹60k   │ mule_12  │   │   └──────────┘
  │victim_2├───────────▶│          ├────────▶│          ├───┘
  └────────┘            └──────────┘         └──────────┘
```

### Generation steps

1. Sample `n_victims` customers from the base population.
2. Each victim makes 1–3 transfers to **newly added payees** (the mule VPAs), for amounts far above their own normal spending (`amount_multiplier_vs_baseline` × their 90-day mean).
3. Layer-1 mules receive, wait `dwell_time_hours`, then fan out to layer-2 mules, split `split_factor` ways.
4. Layer-2 mules converge on `n_cashout_points` cash-out accounts.
5. Mule accounts have age `mule_age_days` — brand new in round 0.

### Fields you put in `meta`

```python
meta = {
    "a1_call_active": True,             # inbound call from unknown number during transfer
    "a1_payee_added_at": timestamp,     # when the payee was added
    "a1_mule_layer": 1,                 # 0 = victim, 1 = layer1, 2 = layer2, 3 = cashout
    "a1_cashout_node": "cashout_003",
    "a1_campaign_id": "camp_007",
}
```

`a1_call_active` matters. A real bank can get this from the mobile SDK — whether a call is in progress while a payment is being made. **The fraudster cannot control this, because the coercion call is the attack.** That property makes it a durable signal, and it's the reason you're modelling it.

### `card.json`

```json
{
  "attack_id": "A1",
  "name": "Deepfake coercion to UPI mule cash-out",
  "channel": "UPI_P2P",
  "default_params": {
    "n_victims": 400,
    "mule_age_days": 2,
    "split_factor": 3,
    "dwell_time_hours": 0.2,
    "amount_multiplier_vs_baseline": 8.0,
    "n_cashout_points": 5,
    "n_mules_layer1": 60
  },
  "param_space": {
    "mule_age_days": [1, 120],
    "split_factor": [2, 25],
    "dwell_time_hours": [0.05, 48.0],
    "amount_multiplier_vs_baseline": [2.0, 15.0],
    "n_cashout_points": [3, 60],
    "n_mules_layer1": [20, 400]
  },
  "param_semantics": {
    "mule_age_days": "How long the receiving accounts have existed before being used. Higher looks less like a throwaway account, but pre-aged accounts cost more to acquire and risk being frozen before use.",
    "split_factor": "How many accounts the stolen money is divided across at each hop. Higher lowers per-account fan-in, but needs more mules and slows cash-out.",
    "dwell_time_hours": "How long money sits in a mule account before moving on. Longer looks less like instant laundering, but gives the bank more time to freeze it.",
    "amount_multiplier_vs_baseline": "How large the coerced transfer is relative to the victim's normal spending. Higher is more profitable but more anomalous.",
    "n_cashout_points": "Number of distinct final withdrawal accounts. More points reduce convergence, but cash-out infrastructure is the most expensive part of the operation.",
    "n_mules_layer1": "Size of the first mule layer. More mules means more spread and more acquisition cost."
  },
  "defender_signals": {
    "a1_benef_account_age_days": "Age of the account receiving the money.",
    "a1_benef_fanin_24h": "How many distinct people paid this account in 24 hours.",
    "a1_dissipation_ratio": "Fraction of money leaving the account within an hour of arriving.",
    "a1_payer_amount_zscore": "How unusual this amount is for this specific victim.",
    "a1_payee_age_seconds": "Seconds between the payee being added and money being sent."
  }
}
```

Implement `describe_state()` to return these two blocks plus a one-paragraph `attack_summary`. The Codex reasoner reads them. Without the semantics it is guessing about variable names, and its proposals will be useless.

### `utility()`

The crime's payoff is money successfully moved and cashed out per unit of infrastructure:

```python
def utility(self, df):
    moved   = df[df.meta.str["a1_mule_layer"] == 0].amount.sum()
    n_mules = df.counterparty_id.nunique()
    return normalise(moved / (n_mules * MULE_ACQUISITION_COST))
```

The red agent cannot buy evasion by using 10,000 mules — the utility budget rejects it. Mules cost money to acquire.

---

## 4. Round 0 — your first detector

Build these features in `features.py`:

| Feature | What it measures |
|---|---|
| `a1_benef_account_age_days` | how new is the receiving account |
| `a1_benef_fanin_24h` | distinct senders paying this account in 24h |
| `a1_dissipation_ratio` | money out within 1h ÷ money in |
| `a1_payer_amount_zscore` | amount vs the victim's own 90-day history |
| `a1_payee_age_seconds` | seconds between payee being added and money sent |

Detector: LightGBM on `a1_*` plus the core rolling features.

**Expected round-0 result: recall ≈ 0.89.** Easy. Brand-new accounts receiving money from many strangers and emptying instantly is a screaming signal.

---

## 5. The adaptation — where it gets interesting

### How the adaptation is produced

The mutations below are **not hardcoded by you**. After round 0, our red agent runs in two stages:

1. **Codex reads your detector report** through ChatGPT-authenticated `codex exec` — your per-feature AUCs, your `param_semantics`, your current values, and a few caught-versus-missed examples — and writes a schema-constrained evasion plan in plain English.
2. **CEM search** then tunes the exact numbers inside the region Codex proposed.

For your attack, the reasoner should produce something close to this. Save a real example — it goes on the dashboard:

```json
{
  "reasoning": "Detection is concentrated in three signals, all describing the
    receiving accounts rather than the victim: account age (AUC 0.81), 24-hour
    fan-in (0.78) and dissipation ratio (0.84). The payer side contributes almost
    nothing, which makes sense - the victim genuinely authorised the payment. A
    real operator would therefore invest on the receiving side. Pre-aged accounts
    are purchasable, splitting the payout wider lowers per-account fan-in below any
    plausible threshold, and holding funds a few hours defeats an instant-drain
    rule. Crucially none of these reduce the amount extracted from the victim, so
    the campaign stays worth running.",
  "hypotheses": [
    {"signal_targeted": "a1_benef_account_age_days",
     "action": "raise mule_age_days from 2 to roughly 45",
     "cost_to_attacker": "aged mule accounts cost more and may be frozen first"},
    {"signal_targeted": "a1_benef_fanin_24h",
     "action": "raise split_factor from 3 to roughly 15",
     "cost_to_attacker": "five times more mule accounts to acquire and manage"},
    {"signal_targeted": "a1_dissipation_ratio",
     "action": "raise dwell_time_hours from 0.2 to roughly 6",
     "cost_to_attacker": "longer exposure to account freezes"}
  ],
  "proposed_params": {"mule_age_days": 45, "split_factor": 15, "dwell_time_hours": 6.0}
}
```

Notice the reasoner is doing something CEM alone cannot: it explains *why* the payer side is uninformative, and it names the cost of each move. That cost commentary is what feeds our attacker-cost-frontier chart.

**Your job is not to write this.** Your job is to write `param_semantics` well enough that Codex can produce it. Test it early — run one reasoning pass through the shared `codex exec` adapter at around hour 12 and check that the output is sensible. If Codex proposes nonsense, your semantics are too thin.

### The mutations it lands on

| Param | Round 0 | Round 1 | Why it helps the criminal |
|---|---|---|---|
| `mule_age_days` | 2 | 45 | accounts are pre-aged before use |
| `split_factor` | 3 | 15 | per-account fan-in drops below threshold |
| `dwell_time_hours` | 0.2 | 6 | dissipation no longer looks instant |

Result:

```
a1_benef_account_age_days  AUC 0.81 → 0.55   DEAD
a1_benef_fanin_24h         AUC 0.78 → 0.58   DEAD
a1_dissipation_ratio       AUC 0.84 → 0.61   DEGRADED

Recall 0.89 → 0.41
Retraining on the new data alone recovers only to ~0.52.
```

That last line is the important one. **Retraining does not save you.** The features are genuinely uninformative now. You must build new ones.

---

## 6. Round 1 — the replacement signals

### Move from node-level to component-level

Fifteen mules each look innocent individually. But they still **reconverge** at shared cash-out points, because cash-out infrastructure (agents, ATMs, crypto ramps) is the expensive part and gets reused.

Build a NetworkX directed graph of transfers over a rolling 48h window, then compute:

| Feature | What it measures |
|---|---|
| `a1_component_size` | size of the 2-hop connected component containing the beneficiary |
| `a1_component_dissipation` | outflow ÷ inflow at the **component** level, not the node |
| `a1_cashout_convergence` | distinct layer-1 mules reaching the same cash-out node |
| `a1_component_age_variance` | do all accounts in this component have suspiciously similar ages |

### Add payer-side signals the attacker cannot touch

| Feature | What it measures |
|---|---|
| `a1_call_active` | was a call in progress during the transfer |
| `a1_payee_added_to_transfer_seconds` | payee added 40 seconds before a ₹2.4L transfer is not normal |
| `a1_session_duration_anomaly` | coerced sessions are long and hesitant |

**Expected recovery: recall ≈ 0.86.**

### The economics to quote in the pitch

To break second-hop convergence, the attacker needs **disjoint cash-out infrastructure per mule**. Mule acquisition cost rises roughly 5x and campaign yield drops about 60%.

> **Your durable insight, one sentence:** the attacker can reshape the money's path, but not the victim's behaviour while being coerced. That asymmetry is permanent.

---

## 7. What your finished work looks like

### Files

```
attacks/a1_upi_mule/
├── card.json
├── generator.py       # A1Generator(AttackGenerator)
├── features.py        # A1Features(FeatureBlock)
├── detector.py        # A1Detector(DetectorHead)
├── README.md          # your signal mortality table
└── tests/test_a1.py
```

### Sample generated rows

| event_id | timestamp | customer_id | counterparty_id | amount | is_first_time_payee | attack_id | label |
|---|---|---|---|---|---|---|---|
| a1_00001 | 2026-06-14 21:14:02 | cust_08812 | vpa_mule_017 | 240000 | 1 | A1 | 1 |
| a1_00002 | 2026-06-14 21:26:41 | vpa_mule_017 | vpa_mule_112 | 80000 | 1 | A1 | 1 |
| a1_00003 | 2026-06-14 21:27:09 | vpa_mule_017 | vpa_mule_118 | 79500 | 1 | A1 | 1 |
| a1_00004 | 2026-06-14 22:03:55 | vpa_mule_112 | vpa_cashout_03 | 78000 | 1 | A1 | 1 |

### Your README.md — the signal mortality table

```markdown
## A1 — Signal Mortality

| Signal                       | R0 AUC | R1 AUC | Verdict           |
|------------------------------|--------|--------|-------------------|
| a1_benef_account_age_days    | 0.81   | 0.55   | DEAD              |
| a1_benef_fanin_24h           | 0.78   | 0.58   | DEAD              |
| a1_dissipation_ratio         | 0.84   | 0.61   | DEGRADED          |
| a1_component_size            |   —    | 0.79   | NEW — replacement |
| a1_cashout_convergence       |   —    | 0.83   | NEW — replacement |
| a1_call_active               |   —    | 0.88   | NEW — durable     |

Recall: 0.89 (R0) → 0.41 (R1 pre-fix) → 0.86 (R1 post-fix)
Attacker cost: mule acquisition +5x, campaign yield -60%
```

*(These numbers are the shape to expect. Replace with your real measurements.)*

### metrics.json contribution

```json
"A1": {
  "recall": 0.86, "precision": 0.91, "auc_pr": 0.88,
  "recall_at_fpr_0.001": 0.74, "utility": 0.38
}
```

Note `utility` falling from ~1.0 to 0.38. That is the attacker paying for evasion, and it's a chart on our scoreboard.

---

## 8. Your 60-second pitch

> "A fake police officer video-calls a victim and frightens them into transferring money. The customer's own phone, own PIN, own fingerprint — every authentication check passes, because the customer really did authorise it. So we stopped looking at the sender and started looking at the receiver. When we did that, the fraudsters aged their mule accounts and split the money fifteen ways, and our detection collapsed from 89% to 41%. So we stopped looking at accounts and started looking at *networks* — because however many mules they use, the cash-out points are expensive and get reused. We got back to 86%, and we made each campaign cost them five times more to run."

---

## 9. Time budget

| Hours | Task |
|---|---|
| 0–3 | Phase 0 with the team |
| 3–8 | Generator + `card.json`, passing the contract test |
| 8–12 | Round-0 features + LightGBM detector, measure baseline |
| 12–16 | **H16 checkpoint — demo valid rows to the team** |
| 16–22 | NetworkX component features + payer-side signals |
| 22–26 | Measure round-1 AUCs, write README signal mortality table |
| 26+ | Phase 3: you own `loop/run_rounds.py` and the **two-stage red agent** |

**Your Phase 3 job grew.** You now build both stages: `redteam/llm_agent.py` (read-only `codex exec` subprocess adapter, JSON Schema output, timeout/error handling, clamping to `param_space`, retry, and caching) and `redteam/cem.py`. It must reuse ChatGPT subscription authentication from `codex login`; do not add an OpenAI API key or SDK. Roughly 3 extra hours.

**Build order matters.** Get the pure-CEM loop running end to end first, then bolt the Codex reasoner on top of it. A working numeric loop you upgrade is far safer than a Codex subprocess that fails at hour 40. The Codex path must always fall back to CEM on a missing CLI, authentication failure, timeout, non-zero exit, or invalid response.

## 10. Do not

- Edit anything outside `attacks/a1_upi_mule/`
- Create a feature column without the `a1_` prefix
- Add a column to the shared schema — use `meta`
- Rebuild core rolling aggregates; they already exist in `core/features.py`
- Make `generate()` non-deterministic — always take `seed` and use `np.random.default_rng(seed)`
