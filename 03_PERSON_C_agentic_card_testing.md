# PERSON C — Attack A3: Agentic Card Testing

**Folder you own:** `attacks/a3_card_testing/`
**Feature prefix:** `a3_`
**Branch:** `feat/a3`
**Read `00_TEAM_BRIEF_AND_INTEGRATION (1).md` first.**

---

## 1. Context in 60 seconds

We are in the Mastercard Innovation Challenge at Global Fintech Fest 2026 (Mumbai, 9–11 September). The challenge asks us to play both sides: invent GenAI-powered payment fraud, simulate it realistically, then detect it.

The winning idea is a **closed loop**. We build attacks, train a detector, then an automated red agent mutates the attacks until our detector stops working. We then build better signals. Three rounds of that, plotted on a chart, is our submission.

You own one attack end to end: generator, detector, and the story of which signals died and what replaced them.

---

## 2. Your attack, in plain language

A criminal buys a dump of 50,000 stolen card numbers from a breach. Most are already cancelled or expired. Before they can be sold on or used for real purchases, the criminal needs to know **which ones still work**.

So they test them. Each card gets charged a tiny amount — ₹40 at some small online shop. If it goes through, the card is live and valuable. If it declines, discard it.

The GenAI part is that this is no longer a dumb script hammering one merchant. An AI agent runs the campaign: it picks which merchants to hit, varies amounts to look organic, rotates device fingerprints, and paces itself to stay under whatever velocity thresholds it infers from the pattern of declines.

### Why this matters to these judges

This is **Mastercard's home turf** — card-not-present fraud on the card rail. And it is surging: AI-powered carding attacks now validate stolen credentials in moments, and **card-testing attacks rose 175% year-on-year between January and April 2026**.

**Your attack is our credibility anchor.** It proves we understand the fundamentals of card fraud before we get clever with agentic commerce. A judge from Mastercard's fraud team will evaluate this one hardest, because they know it best.

You also have the fastest attack to build, which is why you own the fidelity proof in Phase 3.

---

## 3. What you generate

Channel: `CNP`.

### The structure

```
  BREACH DUMPS              BOT DEVICES           LOW-FRICTION MERCHANTS
  ┌──────────────┐          ┌─────────┐          ┌──────────────┐
  │ dump_A       │          │ dev_001 ├─────────▶│ merch_0182   │  ₹42
  │ BIN 522xxx   ├─────────▶│         ├─────────▶│ merch_0447   │  ₹38
  │ 18,000 cards │          └─────────┘          │ merch_0891   │  ₹51
  └──────────────┘          ┌─────────┐          │ merch_1203   │  ₹45
  ┌──────────────┐          │ dev_002 ├─────────▶│              │
  │ dump_B       ├─────────▶│         │          │ (weak AVS,   │
  │ BIN 411xxx   │          └─────────┘          │  no 3DS)     │
  └──────────────┘                               └──────────────┘
```

### Generation steps

1. Create `n_bins` card dumps. Each dump draws cards from a narrow BIN range — this reflects reality, because a breach at one issuer or one processor yields cards sharing BIN prefixes.
2. Spin up bot devices. Each device handles `cards_per_device` cards before rotating.
3. The bot picks merchants from a low-friction pool of size `n_merchants` — merchants with weak address verification and no 3-D Secure.
4. Each attempt is a micro-transaction with amount drawn from `amount_range`.
5. Apply a realistic `decline_rate` — most tested cards are dead. This matters, because declines are how the bot learns.
6. Pace at `cards_per_hour`.

### Fields you put in `meta`

```python
meta = {
    "a3_bin_prefix": "522145",
    "a3_dump_id": "dump_A",
    "a3_bot_session_id": "bot_0031",
    "a3_merchant_friction_tier": "LOW",   # LOW | MED | HIGH
    "a3_attempt_index": 17,
}
```

### `card.json`

```json
{
  "attack_id": "A3",
  "name": "Agentic card testing",
  "channel": "CNP",
  "default_params": {
    "cards_per_hour": 40,
    "amount_min": 30,
    "amount_max": 60,
    "n_merchants": 6,
    "cards_per_device": 40,
    "n_bins": 2,
    "decline_rate": 0.78,
    "merchant_friction_mix": 0.05
  },
  "param_space": {
    "cards_per_hour": [2, 60],
    "amount_min": [10, 400],
    "amount_max": [40, 900],
    "n_merchants": [3, 120],
    "cards_per_device": [1, 60],
    "n_bins": [1, 20],
    "merchant_friction_mix": [0.0, 0.8]
  },
  "param_semantics": {
    "cards_per_hour": "Testing throughput. This IS the crime - lowering it evades velocity rules but directly reduces how many live cards get validated per campaign.",
    "cards_per_device": "How many cards share a device fingerprint before rotation. Lower breaks device-based graph clustering; fingerprint rotation is essentially free for a bot.",
    "n_merchants": "How many merchants the campaign is spread across. More merchants dilute per-merchant spikes but require finding more soft targets.",
    "n_bins": "How many separate breach dumps the cards are drawn from. More dumps break BIN clustering, but each dump costs money and mixing them shrinks usable volume per campaign.",
    "merchant_friction_mix": "Fraction of merchants used that have strong verification such as 3DS. Raising it blends in with normal traffic, but strong-verification merchants produce declines that carry no information about whether the card is live, so it wastes attempts.",
    "amount_min": "Lower bound of the test charge. Larger amounts look less like micro-testing but risk alerting the cardholder.",
    "amount_max": "Upper bound of the test charge."
  },
  "defender_signals": {
    "a3_device_card_fanin_1h": "How many distinct cards were used on one device in an hour.",
    "a3_merchant_micro_txn_rate_1h": "Rate of very small transactions at this merchant.",
    "a3_amount_percentile": "How far below normal spending this amount sits.",
    "a3_merchant_set_jaccard": "How similar the set of merchants touched is across different cards.",
    "a3_bin_cooccurrence_score": "Whether cards appearing together cluster on issuer BIN prefix more than chance would predict."
  }
}
```

Implement `describe_state()` to return these two blocks plus a one-paragraph `attack_summary`. The Codex reasoner reads them — it cannot reason about `merchant_friction_mix` unless you explain the decline trade-off.

`merchant_friction_mix` is the fraction of merchants used that have strong verification. Round 0 sits near zero — the bot only hits soft targets. Raising it helps the bot blend in, but it also raises declines, which is itself a signal. That trade-off is your economic story.

### `utility()`

The criminal's payoff is validated live cards per hour:

```python
def utility(self, df):
    approved = (df.auth_result == "APPROVED").sum()
    hours    = (df.timestamp.max() - df.timestamp.min()).total_seconds() / 3600
    return normalise(approved / max(hours, 1))
```

This is what stops the red agent from "evading" by simply testing three cards a day. **Throughput is the crime.**

---

## 4. Round 0 — your first detector

| Feature | What it measures |
|---|---|
| `a3_device_card_fanin_1h` | distinct cards used on one device in an hour |
| `a3_merchant_micro_txn_rate_1h` | rate of tiny transactions at this merchant |
| `a3_amount_percentile` | how far below normal this amount sits |
| `a3_card_first_seen_recency` | is this card brand new to the system |

Forty unrelated cards on one device in twenty minutes is impossible to miss.

**Expected round-0 result: recall ≈ 0.91.**

---

## 5. The adaptation

### How the adaptation is produced

You do not hardcode the mutation. After round 0 our red agent runs in two stages: **Codex reads your detector report through ChatGPT-authenticated `codex exec`** (per-feature AUCs, your `param_semantics`, current values, caught-versus-missed examples) and writes a schema-constrained evasion plan in plain English, then **CEM search** tunes the numbers inside that region.

Expected output for your attack:

```json
{
  "reasoning": "Device fan-in is the dominant signal at AUC 0.88, with merchant
    micro-transaction rate behind it at 0.74. Both are cheap to defeat. Device
    fingerprints cost a bot nothing to rotate, so dropping to three cards per device
    removes the clustering outright. Widening from six merchants to around forty
    flattens the per-merchant spike. Slowing from forty to twelve cards an hour
    should clear any velocity threshold. The last one is the only move with a real
    price: throughput is the entire point of card testing, so I am trading validated
    cards per hour for stealth and I should not slow further than necessary.",
  "hypotheses": [
    {"signal_targeted": "a3_device_card_fanin_1h",
     "action": "drop cards_per_device from 40 to 3",
     "cost_to_attacker": "none - fingerprint rotation is free"},
    {"signal_targeted": "a3_merchant_micro_txn_rate_1h",
     "action": "raise n_merchants from 6 to ~40",
     "cost_to_attacker": "must source more low-friction merchants"},
    {"signal_targeted": "velocity thresholds generally",
     "action": "lower cards_per_hour from 40 to ~12",
     "cost_to_attacker": "70% reduction in validation throughput"}
  ],
  "proposed_params": {"cards_per_device": 3, "n_merchants": 40, "cards_per_hour": 12}
}
```

**Watch what the reasoner did not propose:** it left `n_bins` alone. That is correct behaviour and it is the cleanest possible demonstration of your durable signal — Codex had BIN mixing available in the parameter space, read that mixing dumps shrinks usable volume, and judged it not worth the cost. **Quote that in the pitch.** Even an adaptive attacker reasoning explicitly about the trade-off declines to attack BIN co-occurrence, because the cost is structural.

**Your job is not to write this.** Your job is to write `param_semantics` well enough that Codex produces it — especially the cost clauses, since those are what make it decline the `n_bins` move. Run one reasoning pass through the shared adapter around hour 12 and check.

### The mutations it lands on

| Param | Round 0 | Round 1 | Why it helps the criminal |
|---|---|---|---|
| `cards_per_device` | 40 | 3 | device graph convergence disappears |
| `n_merchants` | 6 | 40 | no single merchant sees a spike |
| `cards_per_hour` | 40 | 12 | slides under velocity thresholds |

Result:

```
a3_device_card_fanin_1h        AUC 0.88 → 0.53   DEAD
a3_merchant_micro_txn_rate_1h  AUC 0.74 → 0.59   DEGRADED

Recall 0.91 → 0.44
```

---

## 6. Round 1 — the replacement signals

Your two new signals are the best on the team, because they attack things the criminal **structurally cannot change**.

### Merchant-set similarity

Devices rotate freely and cost nothing. But the bot still needs merchants with weak verification, because that is the entire point — testing against a strong-3DS merchant produces declines that tell you nothing about whether the card is live.

So even with 40 merchants instead of 6, the **merchant sets touched by different bot devices remain highly correlated**, while genuine customers have idiosyncratic, personal merchant sets.

| Feature | What it measures |
|---|---|
| `a3_merchant_set_jaccard` | Jaccard similarity of merchant sets across cards in a window |
| `a3_merchant_friction_score` | average verification strength of merchants touched |

Build this with a NetworkX **bipartite card ↔ merchant graph** and compute similarity over a rolling window.

### BIN co-occurrence — your best signal

This is the sharpest idea in the whole project, so understand why it works.

The cards come from the same breach. That means they share BIN prefixes. That clustering exists in **how the cards were acquired**, not in how they are used.

> No change in bot behaviour can remove it. Timing, devices, merchants, amounts — none of it touches where the cards came from.

| Feature | What it measures |
|---|---|
| `a3_bin_cooccurrence_score` | do cards seen in this window cluster on BIN prefix beyond chance |
| `a3_bin_entropy_1h` | entropy of BIN prefixes in the window (low = suspicious) |

### Also add

| Feature | What it measures |
|---|---|
| `a3_decline_retry_pattern` | decline followed by a slightly perturbed retry — the signature of a probing agent |

**Expected recovery: recall ≈ 0.87.**

### The economics to quote in the pitch

- Breaking BIN clustering means **mixing dumps**, which shrinks the card supply per campaign
- Breaking merchant correlation means including high-friction merchants, which **raises declines**, which is itself a detection signal

Throughput collapses: **40 → 12 → 4 cards per hour.**

> **Your durable insight, one sentence:** BIN co-occurrence is structurally unremovable, because it is a property of how the cards were stolen, not how they are spent.

That sentence should go on a slide. It is the clearest illustration of the fragile-versus-durable distinction our entire pitch rests on.

---

## 7. What your finished work looks like

### Files

```
attacks/a3_card_testing/
├── card.json
├── generator.py       # A3Generator(AttackGenerator)
├── features.py        # A3Features(FeatureBlock)
├── detector.py        # A3Detector(DetectorHead)
├── README.md          # your signal mortality table
└── tests/test_a3.py
```

### Sample generated rows

| event_id | timestamp | card_id | card_bin | device_id | counterparty_id | amount | auth_result | label |
|---|---|---|---|---|---|---|---|---|
| a3_00001 | 2026-06-14 03:12:04 | card_44192 | 522145 | dev_001 | merch_0182 | 42 | DECLINED | 1 |
| a3_00002 | 2026-06-14 03:12:51 | card_44193 | 522145 | dev_001 | merch_0447 | 38 | DECLINED | 1 |
| a3_00003 | 2026-06-14 03:13:33 | card_44194 | 522146 | dev_001 | merch_0182 | 51 | APPROVED | 1 |

Round 1 rows look the same but with `device_id` changing every third row and `counterparty_id` spread across forty merchants. **The BIN column stays clustered in both rounds — show that side by side, it makes the point instantly.**

### Your README.md — the signal mortality table

```markdown
## A3 — Signal Mortality

| Signal                        | R0 AUC | R1 AUC | Verdict           |
|-------------------------------|--------|--------|-------------------|
| a3_device_card_fanin_1h       | 0.88   | 0.53   | DEAD              |
| a3_merchant_micro_txn_rate_1h | 0.74   | 0.59   | DEGRADED          |
| a3_merchant_set_jaccard       |   —    | 0.80   | NEW — replacement |
| a3_bin_cooccurrence_score     |   —    | 0.85   | NEW — durable     |
| a3_decline_retry_pattern      |   —    | 0.72   | NEW — replacement |

Recall: 0.91 (R0) → 0.44 (R1 pre-fix) → 0.87 (R1 post-fix)
Attacker throughput: 40 → 12 → 4 cards/hour
```

*(These numbers are the shape to expect. Replace with your real measurements.)*

---

## 8. Your 60-second pitch

> "Criminals buy fifty thousand stolen cards and need to know which still work, so an AI agent charges each one forty rupees at a small shop. We caught it by spotting forty cards on one device. So the bot rotated devices every third card and spread across forty merchants, and we dropped from 91% to 44%. Then we asked a different question: where did these cards *come from*? They all came from the same breach, so they share BIN prefixes — and no amount of clever bot behaviour changes that, because it's a fact about the theft, not the spending. We got back to 87%, and we cut their testing rate from forty cards an hour to four."

---

## 9. Time budget

| Hours | Task |
|---|---|
| 0–3 | Phase 0 with the team |
| 3–7 | Generator + `card.json`, passing the contract test |
| 7–11 | Round-0 features + LightGBM detector, measure baseline |
| 11–16 | **H16 checkpoint — demo valid rows to the team** |
| 16–21 | Bipartite graph, Jaccard and BIN co-occurrence features |
| 21–25 | Measure round-1 AUCs, write README signal mortality table |
| 25+ | Phase 3: you own the **fidelity proof** — see section 7 of the team brief |

**Your Phase 3 job is fidelity**, and it is the first screen judges see. You will compare our synthetic base population against a real public dataset using KS statistics, correlation matrix distance, and a discriminator test that should score AUC ≈ 0.5. Download Sparkov or IEEE-CIS from Kaggle early so it is not a last-minute scramble.

## 10. Do not

- Edit anything outside `attacks/a3_card_testing/`
- Create a feature column without the `a3_` prefix
- Add a column to the shared schema — use `meta`
- Forget to generate **declines**. Most tested cards are dead, and the decline pattern is both realism and signal
- Make `generate()` non-deterministic — always take `seed` and use `np.random.default_rng(seed)`
