# A2 — Prompt-Injected AI Shopping Agent

This package simulates an authorised red-team scenario on fully synthetic agentic-commerce
events. A compromised merchant page causes an AI shopping agent to add an attacker-selected
product while the payment token, merchant, category consent, and spending ceiling remain
valid. The attack is detectable only by comparing the signed intent with the resulting cart
and, after adaptation, by checking the cart's economics and provenance.

## Signal mortality

The table below is measured by `python -m attacks.a2_agent_inject.evaluate` on independent
deterministic train/test seeds (600 matched sessions per dataset). Signal AUC is reported in
the predictive direction, so `0.5` means no information regardless of whether high or low
raw values indicate fraud.

| Signal | Round 0 AUC | Evolved AUC | Verdict |
|---|---:|---:|---|
| `a2_intent_cart_cosine` | 0.9998 | 0.5098 | **DEAD** |
| `a2_amount_vs_cap_ratio` | 0.6645 | 0.6562 | weak residual |
| `a2_item_count` | 0.5388 | 0.5066 | never useful |
| `a2_price_zscore_in_category` | — | 1.0000 | **NEW — economic** |
| `a2_max_item_price_share` | — | 0.9450 | **NEW — economic** |
| `a2_cart_mutated_post_load` | — | 1.0000 | **NEW — provenance** |
| `a2_consent_age_hours` | — | 0.6900 | supporting provenance |
| `a2_cart_coherence` | — | 0.5141 | dies with semantic adaptation |
| `a2_min_item_cosine_to_cart` | — | 0.5243 | dies with semantic adaptation |

At the round-0 operating threshold, held-out recall changes as follows:

```text
Round 0 detector                     91.67%
Evolved attack, before defence fix   0.33%
Evolved attack, durable detector    100.00%
```

The adapted generator lowers `injection_semantic_distance` from `0.80` to `0.25`: instead
of inserting an unrelated luxury product, it inserts a product whose category already
matches the signed intent. This collapses the semantic distributions rather than merely
crossing a threshold. The controlled recovery is perfect because the default attack still
uses a 40× price multiplier and post-load mutation; these figures must be recalibrated on
the integrated reference-fitted population rather than presented as production performance.

To evade both replacement controls, the constrained variant lowers the price multiplier
from 40× to 3× and requires the product to be preloaded in merchant source. Measured attacker
utility falls from `0.7976` to `0.0409`, a **94.88% reduction**.

> Durable insight: the injection remains technically possible, but defeating economic and
> provenance controls makes it substantially less profitable and requires deeper merchant
> compromise.

## Package contents

- `card.json` defines the bounded red-agent parameter space and full signal semantics.
- `generator.py` produces deterministic schema-valid legitimate and attacked sessions.
- `features.py` loads the committed MiniLM cache and never encodes text during replay.
- `detector.py` implements the frozen LightGBM detector-head interface.
- `evaluate.py` reproduces the signal-mortality measurements above.
- `build_assets.py` is the one-time precomputation entry point for `data/embeddings.npz`.

## Verification

```powershell
python -m pytest tests/test_contract.py -k a2
python -m pytest attacks/a2_agent_inject/tests/test_a2.py
python -m attacks.a2_agent_inject.evaluate
```

No Codex or sentence-transformer call occurs in generation, feature building, detector
scoring, or dashboard replay. The shared red-agent reasoner consumes `describe_state()`
offline and its output is cached by the integration layer.
