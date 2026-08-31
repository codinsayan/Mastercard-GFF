# A3 — Agentic Card Testing

A3 simulates a deterministic, one-hour card-testing campaign against card-not-present
merchants. The detector combines four initial velocity/amount signals with five causal
replacement signals based on card–merchant overlap, merchant friction, BIN concentration,
and decline retries.

## Feature semantics

All rolling features use event time, expire observations at exactly one hour, and never read
future rows. Equal-timestamp rows are ordered deterministically by `event_id`. The card–merchant
feature is the maximum Jaccard similarity between the current card's merchant set and another
active card's set in the rolling NetworkX bipartite graph. Friction is attributed to the card
from `meta.a3_merchant_friction_tier` (`LOW=0`, `MED=0.5`, `HIGH=1`).

BIN entropy is normalized Shannon entropy over distinct active cards. BIN co-occurrence is the
positive excess of the current BIN's active-window share over its strictly prior distinct-card
background share. A decline retry is attributed only to the retry row when the same card returns
within five minutes with a non-zero amount change no greater than 10%.

Missing identifiers produce zero counts/similarities. Missing amount percentile and merchant
friction use the neutral value `0.5`; missing BIN entropy uses the neutral maximum `1.0`.

## Measured functional benchmark

These are measured synthetic integration results, not external-fidelity claims. The benchmark
used 40 disjoint campaign seeds: 28 for training and 12 for testing. Each campaign contained 500
legitimate events. Round 0 used `default_params`; adapted Round 1 used
`cards_per_hour=12`, `cards_per_device=3`, and `n_merchants=40`, with all other parameters left at
their defaults. The recall threshold was the 99th percentile of legitimate training scores.

| Detector evaluation | Test rows | Fraud rows | ROC AUC | Recall at training 1% FPR threshold |
|---|---:|---:|---:|---:|
| Round-0 head on Round 0 | 6,480 | 480 | 0.999687 | 0.989583 |
| Round-0 head on adapted Round 1 | 6,148 | 148 | 0.990851 | 0.804054 |
| Nine-feature head on adapted Round 1 | 6,148 | 148 | 1.000000 | 1.000000 |

Per-feature values below are direction-normalized ROC AUC (`max(AUC, 1-AUC)`) on the same held-out
campaigns. This reports signal separation regardless of whether low or high values indicate risk.

| Signal | Round-0 AUC | Adapted Round-1 AUC | Measured observation |
|---|---:|---:|---|
| `a3_device_card_fanin_1h` | 0.987500 | 0.834459 | weakened after device rotation |
| `a3_merchant_micro_txn_rate_1h` | 0.965221 | 0.794279 | weakened after merchant spreading |
| `a3_amount_percentile` | 0.630672 | 0.636921 | broadly stable, modest separation |
| `a3_card_first_seen_recency` | 0.906083 | 0.903545 | stable in this simulation |
| `a3_merchant_set_jaccard` | 0.915101 | 0.553910 | weak under the measured adaptation |
| `a3_merchant_friction_score` | 1.000000 | 0.959459 | strongest measured replacement |
| `a3_bin_cooccurrence_score` | 0.624153 | 0.615315 | stable but modest separation |
| `a3_bin_entropy_1h` | 0.673410 | 0.593651 | weakened, above chance |
| `a3_decline_retry_pattern` | 0.500000 | 0.513514 | sparse at the default friction mix |

The benchmark is deliberately limited: it uses the repository's synthetic base population, a
small functional sample, and one specified adaptation. It is not the deferred fidelity proof,
does not use an external dataset, and should not be presented as production detection efficacy.
