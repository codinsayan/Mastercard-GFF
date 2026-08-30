# PERSON D — Attack A4: Synthetic Identity Bust-Out

**Folder you own:** `attacks/a4_synth_identity/`
**Feature prefix:** `a4_`
**Branch:** `feat/a4`
**Read `00_TEAM_BRIEF_AND_INTEGRATION (1).md` first.**

---

## 1. Context in 60 seconds

We are in the Mastercard Innovation Challenge at Global Fintech Fest 2026 (Mumbai, 9–11 September). The challenge asks us to play both sides: invent GenAI-powered payment fraud, simulate it realistically, then detect it.

The winning idea is a **closed loop**. We build attacks, train a detector, then an automated red agent mutates the attacks until our detector stops working. We then build better signals. Three rounds of that, plotted on a chart, is our submission.

You own one attack end to end: generator, detector, and the story of which signals died and what replaced them.

---

## 2. Your attack, in plain language

GenAI can now produce fake identity documents that pass inspection — fake driving licences and passports with realistic shadows, textures and holographic effects. Institutions are reporting a surge in suspicious activity involving AI-generated documents used to get past KYC checks, and synthetic identity fraud is the fastest-growing financial crime.

So a criminal creates a thousand people who do not exist. Each gets an account.

Then comes the patient part. **These fake customers behave perfectly.** They make small purchases, pay their bills on time, never miss a payment. Over two or three months the bank's models learn to trust them and raise their credit limits.

Then one day, all of them max out simultaneously and vanish. That final coordinated spike is called the **bust-out**.

### Why this attack is different from the other three

Attacks A1, A2 and A3 all happen in minutes or hours. Yours unfolds over months.

Look at any single transaction from a synthetic identity during the build phase: a ₹800 grocery purchase, paid off on time. It is completely, genuinely normal. **A per-transaction model cannot detect this attack at all**, no matter how good it is, because there is nothing anomalous in any individual event.

The fraud is in the **trajectory** — the shape of behaviour over time and the fact that a group of supposedly unrelated customers is following the same shape in sync.

**Your attack is our technical anchor.** It is the reason our ensemble contains a sequence model. Without you, a judge could reasonably ask "why isn't this just one XGBoost?" With you, the answer is concrete.

---

## 3. What you generate

This is the most complex generator on the team. Budget time accordingly.

Channels: `CNP` and `CARD_PRESENT`, over a compressed 120-day timeline.

### The structure

```
  IDENTITY CLUSTER (one GenAI "playbook")
  ┌────────────────────────────────────────────────┐
  │ ident_001  ident_002  ident_003 ... ident_040  │
  └────────┬───────────────────────────────────────┘
           │
    DAY 0  │  onboarding — KYC passed with generated documents
           ▼
    DAY 1–75   build phase
           │   small purchases, on-time payments
           │   utilisation creeping up: 8% → 15% → 28% → 44%
           │   credit limit raised by the issuer
           ▼
    DAY 76–80  BUST-OUT
           │   every identity in the cluster maxes out
           │   within a 4-day window
           ▼
           ✕   accounts abandoned
```

### Generation steps

1. Create `n_clusters` identity clusters, each with `identities_per_cluster` fake customers.
2. In round 0, identities within a cluster share faint artifacts at rate `attribute_overlap_rate`: address tokens ("Flat 4B, Sunrise Apartments"), device fingerprint prefixes, phone number blocks.
3. Onboarding event for each identity.
4. Build phase of `build_days`: regular small transactions, on-time payments, utilisation rising along a curve governed by `utilization_curve_exponent`.
5. Credit limit increases as the issuer's own logic would grant them.
6. Bust-out: all identities in a cluster spike to near-limit within `bustout_sync_window_days`.

### The utilisation curve — get this right

Organic credit growth is roughly **linear or concave** — people settle into a stable spending level.

A synthetic bust-out ramp is **convex** — deliberately accelerating, because the operator wants the limit raised as fast as trust allows.

```python
utilization(t) = (t / build_days) ** utilization_curve_exponent
# exponent > 1  → convex  → synthetic
# exponent ≈ 1  → linear  → organic
```

That single parameter is the heart of your round-1 detection. Make sure your legit population also has customers with rising utilisation, or the signal is trivially easy and the judges will smell it.

### Fields you put in `meta`

```python
meta = {
    "a4_identity_cluster_id": "cluster_007",
    "a4_playbook_id": "pb_02",
    "a4_days_since_onboarding": 62,
    "a4_credit_limit": 180000,
    "a4_utilization": 0.44,
    "a4_payment_ratio": 1.0,        # fraction of statement paid off
    "a4_phase": "BUILD",            # ONBOARD | BUILD | BUSTOUT
}
```

### `card.json`

```json
{
  "attack_id": "A4",
  "name": "Synthetic identity bust-out",
  "channel": "CNP",
  "default_params": {
    "n_clusters": 12,
    "identities_per_cluster": 40,
    "attribute_overlap_rate": 0.70,
    "build_days": 75,
    "utilization_curve_exponent": 2.4,
    "bustout_sync_window_days": 4,
    "playbook_variance": 0.05
  },
  "param_space": {
    "attribute_overlap_rate": [0.0, 0.9],
    "build_days": [30, 220],
    "utilization_curve_exponent": [1.0, 3.5],
    "bustout_sync_window_days": [1, 45],
    "playbook_variance": [0.0, 0.8],
    "identities_per_cluster": [10, 80]
  },
  "param_semantics": {
    "attribute_overlap_rate": "How much identities within a cluster share address fragments, device prefixes and phone blocks. Lowering it is essentially free, because generating fully distinct identity attributes is exactly what generative AI is good at.",
    "build_days": "How long fake identities behave well before busting out. Longer earns higher credit limits and looks more organic, but ties up capital and delays payout.",
    "utilization_curve_exponent": "Shape of the credit-usage ramp. Above 1 is convex, meaning deliberately accelerating. Near 1 is linear, which resembles organic growth but earns limits more slowly.",
    "bustout_sync_window_days": "How tightly the cluster's final maxing-out is synchronised. Wider is less detectable as coordination, but gives the issuer time to freeze later accounts once the first ones default.",
    "playbook_variance": "How much the operating strategy differs between identities. Raising it breaks behavioural correlation, but destroys the automation that makes the scheme scalable - each identity then needs individual handling.",
    "identities_per_cluster": "Number of fake identities run under one playbook. More is more profitable per campaign but easier to spot as a group."
  },
  "defender_signals": {
    "a4_shared_address_token_count": "How many other accounts share an address fragment with this one.",
    "a4_shared_device_prefix_count": "How many accounts share a device fingerprint prefix.",
    "a4_phone_block_density": "Density of accounts in the same phone number block.",
    "a4_trajectory_cluster_density": "How many other accounts follow a near-identical behavioural path over time.",
    "a4_ramp_convexity": "Whether credit usage accelerates rather than growing steadily.",
    "a4_bustout_sync_score": "How many behaviourally similar accounts spike in the same window."
  }
}
```

Implement `describe_state()` to return these two blocks plus a one-paragraph `attack_summary`. The cost clauses matter most in your case — see section 5.

### `utility()`

The criminal's payoff is credit extracted per unit of time and effort invested:

```python
def utility(self, df):
    extracted = total bust-out spend
    invested  = build_days * n_identities   # each identity costs time to nurture
    return normalise(extracted / invested)
```

This is what makes the round-2 economics bite. Read section 6.

---

## 4. Round 0 — your first detector

| Feature | What it measures |
|---|---|
| `a4_shared_address_token_count` | how many other accounts share an address fragment |
| `a4_shared_device_prefix_count` | how many share a device fingerprint prefix |
| `a4_phone_block_density` | density of accounts in the same phone number block |
| `a4_account_age_at_first_large_txn` | account age when the first big spend appears |

Cluster identities on shared attributes and the rings pop out immediately.

**Expected round-0 result: recall ≈ 0.85.**

---

## 5. The adaptation — attribute clusters simply vanish

### How the adaptation is produced

You do not hardcode the mutation. After round 0 our red agent runs in two stages: **Codex reads your detector report through ChatGPT-authenticated `codex exec`** (per-feature AUCs, your `param_semantics`, current values, caught-versus-missed examples) and writes a schema-constrained evasion plan in plain English, then **CEM search** tunes the numbers inside that region.

Expected output for your attack:

```json
{
  "reasoning": "All three working signals - shared address tokens (0.83), device
    prefixes (0.79), phone block density (0.71) - measure the same thing: identities
    within a cluster resemble each other on static attributes. This is the cheapest
    possible thing for me to fix. Generating a thousand fully distinct addresses,
    names, phone numbers and device profiles is precisely what generative models do
    well, and it costs nothing and changes no behaviour. Dropping overlap to near
    zero should eliminate all three signals at once with no impact on payout. I am
    deliberately not touching playbook_variance: randomising strategy per identity
    would break the automation that makes this scheme scalable at all, and there is
    no evidence yet that the defence is looking at behaviour.",
  "hypotheses": [
    {"signal_targeted": "a4_shared_address_token_count",
     "action": "drop attribute_overlap_rate from 0.70 to ~0.02",
     "cost_to_attacker": "none - this is free for a generative model"}
  ],
  "proposed_params": {"attribute_overlap_rate": 0.02}
}
```

**Two things to draw out for judges.** First, the reasoner is explicit that this evasion is *free* — that is precisely why the round-0 signals were fragile, and it says so unprompted. Second, its last sentence is a prediction that fails: it declines to randomise the playbook because it sees no evidence the defence is watching behaviour. In round 2, once your trajectory features are live, it will be forced to reconsider — and it will discover that the fix is expensive. **That two-round arc, in the attacker's own words, is the single best narrative on the team.** Make sure round 2 gets run for your attack even if it gets cut elsewhere.

**Your job is not to write this.** Your job is to write `param_semantics` well enough that Codex produces it — particularly the cost clause on `playbook_variance`, which is what drives the round-2 reversal. Run one reasoning pass through the shared adapter around hour 12 and check.

### The mutation it lands on

| Param | Round 0 | Round 1 |
|---|---|---|
| `attribute_overlap_rate` | 0.70 | 0.02 |

This is what GenAI is genuinely good at. Generating a thousand fully distinct addresses, names, phone numbers and device profiles is trivial and free. There is no cost to the attacker at all.

```
a4_shared_address_token_count  AUC 0.83 → 0.51   DEAD
a4_shared_device_prefix_count  AUC 0.79 → 0.50   DEAD
a4_phone_block_density         AUC 0.71 → 0.52   DEAD

Recall 0.85 → 0.29
```

Your entire round-0 detector is gone. Note that this adaptation cost the attacker **nothing** — which is exactly what makes those signals fragile.

---

## 6. Round 1 — the replacement signals

Here is the insight: the attacker changed **who these identities look like**. They did not change **what these identities are trying to do**.

All 480 fake customers are following the same operating playbook, because that playbook is what makes the scheme profitable. So even with completely distinct attributes, their **behaviour over time stays correlated**.

### Trajectory embeddings — your main deliverable

Train a small GRU (2 layers, hidden 32) over each account's first 90 days of activity. Feed it a weekly summary: transaction count, mean amount, utilisation, payment ratio, distinct merchants. Take the final hidden state as a **trajectory embedding** for that account.

Then cluster the embeddings (DBSCAN or k-means).

Accounts produced by one playbook land close together in embedding space even when every attribute differs.

| Feature | What it measures |
|---|---|
| `a4_trajectory_cluster_density` | local density around this account's embedding |
| `a4_trajectory_dist_to_centroid` | distance to nearest cluster centroid |
| `a4_trajectory_cluster_size` | size of the embedding cluster this account sits in |

### Curve shape

| Feature | What it measures |
|---|---|
| `a4_ramp_convexity` | fitted convexity of the utilisation curve |
| `a4_payment_ratio_regularity` | synthetic accounts pay suspiciously exactly |

### Synchronisation

| Feature | What it measures |
|---|---|
| `a4_bustout_sync_score` | how many similar-trajectory accounts spike in the same window |

**Expected recovery: recall ≈ 0.82.**

### The economics to quote in the pitch

To break behavioural correlation, the attacker must **randomise the playbook per identity**. That means:

- build time extends from about 6 weeks to roughly 5 months
- achieved credit limits fall, because slower, noisier build-up earns less trust
- the automation that made the scheme scalable is destroyed — each identity now needs individual handling

> **Your durable insight, one sentence:** attributes are cheap for GenAI to diversify, but strategy is expensive, because randomising strategy degrades the attack's own payoff.

---

## 7. What your finished work looks like

### Files

```
attacks/a4_synth_identity/
├── card.json
├── generator.py       # A4Generator(AttackGenerator)
├── features.py        # A4Features(FeatureBlock)
├── trajectory.py      # GRU encoder + clustering
├── detector.py        # A4Detector(DetectorHead)
├── README.md          # your signal mortality table
└── tests/test_a4.py
```

### Sample generated rows

| event_id | timestamp | customer_id | day | phase | utilization | amount | label |
|---|---|---|---|---|---|---|---|
| a4_00112 | 2026-03-04 | ident_0031 | 12 | BUILD | 0.09 | 840 | 1 |
| a4_00389 | 2026-04-18 | ident_0031 | 57 | BUILD | 0.31 | 2,150 | 1 |
| a4_00744 | 2026-05-19 | ident_0031 | 88 | BUILD | 0.58 | 4,900 | 1 |
| a4_00902 | 2026-05-27 | ident_0031 | 96 | BUSTOUT | 0.97 | 71,400 | 1 |

Look at row 1. A ₹840 purchase, 9% utilisation, paid on time. **That row is labelled fraud and there is nothing wrong with it.** Show exactly this table to judges — it is the clearest possible argument for why sequence modelling is required.

### Your README.md — the signal mortality table

```markdown
## A4 — Signal Mortality

| Signal                          | R0 AUC | R1 AUC | Verdict           |
|---------------------------------|--------|--------|-------------------|
| a4_shared_address_token_count   | 0.83   | 0.51   | DEAD              |
| a4_shared_device_prefix_count   | 0.79   | 0.50   | DEAD              |
| a4_phone_block_density          | 0.71   | 0.52   | DEAD              |
| a4_trajectory_cluster_density   |   —    | 0.80   | NEW — replacement |
| a4_ramp_convexity               |   —    | 0.76   | NEW — durable     |
| a4_bustout_sync_score           |   —    | 0.84   | NEW — durable     |

Recall: 0.85 (R0) → 0.29 (R1 pre-fix) → 0.82 (R1 post-fix)
Attacker cost: build time 6 weeks → ~5 months, limits achieved lower
```

*(These numbers are the shape to expect. Replace with your real measurements.)*

---

## 8. Your 60-second pitch

> "AI now makes fake ID documents good enough to pass KYC, so criminals open accounts for a thousand people who don't exist. Then they wait. These fake customers behave perfectly for two months — small purchases, bills paid on time — until the bank trusts them and raises their limits. Then they all max out on the same day. We first caught them because the fake identities shared address fragments and device prefixes. But diversifying those costs the attacker nothing, so they did, and we fell from 85% to 29%. So we stopped looking at who they *look* like and started looking at how they *behave over time*. A thousand identities following the same playbook produce the same behavioural shape, whatever their names are. We got back to 82%, and breaking that forces them to slow from six weeks to five months per campaign."

---

## 9. Time budget

Your generator is the most complex. Start early and keep the GRU simple.

| Hours | Task |
|---|---|
| 0–3 | Phase 0 with the team |
| 3–10 | Generator + `card.json` — the 120-day timeline is the hard part |
| 10–13 | Round-0 attribute features + LightGBM, measure baseline |
| 13–16 | **H16 checkpoint — demo valid rows to the team** |
| 16–20 | `trajectory.py`: GRU encoder + clustering |
| 20–24 | Convexity and sync features, measure round-1 AUCs |
| 24–26 | README signal mortality table |
| 26+ | Phase 3: you own the **meta-learner and leave-one-attack-out** — see section 7 of the team brief |

**Fallback if the GRU eats your time:** replace trajectory embeddings with hand-engineered trajectory statistics — weekly utilisation deltas, spend growth rate, payment regularity variance — clustered with k-means. This gets most of the signal. A working fallback beats a broken GRU, and you can say in the pitch that the GRU is the production path.

## 10. Do not

- Edit anything outside `attacks/a4_synth_identity/`
- Create a feature column without the `a4_` prefix
- Add a column to the shared schema — use `meta`
- Simulate real calendar time — compress 120 days into generated timestamps
- Forget that **build-phase rows are labelled fraud but look completely normal**. This is intentional and it is your whole point. Do not "fix" it by making them subtly weird
- Make `generate()` non-deterministic — always take `seed` and use `np.random.default_rng(seed)`
