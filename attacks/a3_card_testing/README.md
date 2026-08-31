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

## Reproducible adaptive benchmark

These are measured synthetic integration results, not external-fidelity claims. Run them with:

```powershell
python -m attacks.a3_card_testing.evaluation
```

The committed experiment used 16 training campaigns and eight disjoint test campaigns per round,
with 500 legitimate events per campaign. CEM used three fixed search campaigns, 16 candidates per
iteration, and three iterations. Detector thresholds are the 99th percentile of legitimate
training scores. The authenticated Codex CLI produced both proposals without fallback; CEM then
evaluated complete parameter vectors under fixed throughput, minimum-utility, and amount-KS
constraints. Full reports, proposals, candidate counts, and metrics are under `results/`.

### Throughput and detector mortality

| Round | Target | Actual cards/hour | Utility | Active detector ROC AUC | Active recall | Frozen previous-head recall |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 40.0 | 0.147917 | 0.999755 | 0.981250 | — |
| 1 | 12 | 12.0 | 0.085417 | 0.999996 | 1.000000 | 0.247934 |
| 2 | 4 | 4.0 | 0.035417 | 1.000000 | 1.000000 | 1.000000 |

Round 1 demonstrates the intended first-head mortality: recall falls from 0.981250 on its native
Round-0 traffic to 0.247934 on the Codex/CEM-adapted traffic. Retraining on all nine signals
recovers recall to 1.000000 in this synthetic holdout. The second adaptation reaches four
cards/hour, but does not defeat the retrained nine-feature head in this measurement.

The measured parameter sequence was:

| Parameter | Round 0 | Round 1 | Round 2 |
|---|---:|---:|---:|
| `cards_per_hour` | 40 | 12 | 4 |
| `cards_per_device` | 40 | 1 | 1 |
| `n_merchants` | 6 | 16 | 24 |
| `n_bins` | 2 | 12 | 4 |
| `amount_min` | 30 | 10 | 150 |
| `amount_max` | 60 | 76 | 300 |
| `decline_rate` | 0.78 | 0.555111 | 0.50 |
| `merchant_friction_mix` | 0.05 | 0.285461 | 0.15 |

### Signal mortality

Values are direction-normalized ROC AUC (`max(AUC, 1-AUC)`) on the held-out campaigns.

| Signal | Round 0 | Round 1 | Round 2 | Measured verdict |
|---|---:|---:|---:|---|
| `a3_device_card_fanin_1h` | 0.987500 | 0.500000 | 0.500000 | dead after adaptation |
| `a3_merchant_micro_txn_rate_1h` | 0.967230 | 0.854721 | 0.719375 | degraded |
| `a3_amount_percentile` | 0.631732 | 0.673572 | 0.727079 | modestly stronger |
| `a3_card_first_seen_recency` | 0.908000 | 0.888992 | 0.893474 | durable in this simulation |
| `a3_merchant_set_jaccard` | 0.915624 | 0.674757 | 0.500414 | degraded, then dead |
| `a3_merchant_friction_score` | 1.000000 | 0.690083 | 0.763158 | degraded replacement |
| `a3_bin_cooccurrence_score` | 0.618620 | 0.787084 | 0.732964 | new durable replacement |
| `a3_bin_entropy_1h` | 0.617129 | 0.684733 | 0.560059 | weak replacement |
| `a3_decline_retry_pattern` | 0.500000 | 0.603306 | 0.578947 | weak replacement |

The measured reasoning differs from the MD's illustrative story: Codex did choose to increase
`n_bins` in both adaptations after weighing its stated cost. The committed artifacts preserve that
result rather than replacing it with the expected narrative.

This remains a small synthetic benchmark. It is not the Phase-3 fidelity proof, uses no external
dataset, and should not be presented as production detection efficacy.

## Presentation artifacts

Generate four deterministic, offline SVG charts plus their machine-readable source data with:

```powershell
python -m attacks.a3_card_testing.presentation
```

Outputs are under `results/presentation/`:

- `round_metrics.svg` — active recall/ROC AUC and frozen previous-head recall;
- `throughput.svg` — measured 40 → 12 → 4 cards/hour;
- `signal_mortality.svg` — all nine feature AUCs across three rounds;
- `bin_clustering.svg` — configured dominant-BIN share, normalized BIN entropy, and measured BIN
  co-occurrence AUC; and
- `presentation.json` — the exact values used by every chart.

The BIN chart is intentionally honest: the adaptations acquired more dumps and reduced raw BIN
concentration, while `a3_bin_cooccurrence_score` retained direction-normalized AUC 0.787084 in
Round 1 and 0.732964 in Round 2.

## Phase-3 fidelity methodology

The A3 fidelity runner accepts a local Sparkov or IEEE-CIS transaction file, filters labeled fraud
rows when the source provides a label, takes a deterministic bounded sample, fits
`BasePopulation` to its amount/category reference, and compares card-channel synthetic rows using:

- exact two-sample KS statistics for amount, log amount, hour, weekday, and category frequency;
- root-mean-square distance between the two Pearson correlation matrices; and
- a deterministic LightGBM discriminator with a stratified holdout, where synthetic is the
  positive class.

No real dataset currently exists under `data/`. Consequently, root `results/fidelity.json` records
`status: blocked_no_dataset` and null metrics—no external result has been invented. Supplying a
dataset replaces that status with measured statistics, dataset SHA-256, sample counts, correlation
matrices, discriminator AUC, and feature importance.

Sparkov:

```powershell
python -m attacks.a3_card_testing.fidelity --dataset "data/raw/fraudTrain.csv" --format sparkov --output "results/fidelity.json"
```

IEEE-CIS:

```powershell
python -m attacks.a3_card_testing.fidelity --dataset "data/raw/train_transaction.csv" --format ieee-cis --output "results/fidelity.json"
```

Either command also accepts a directory containing the conventional filename. Use `--max-rows`
to change the default deterministic 50,000-row cap and `--seed` to change the default seed of 42.

## Shared orchestrator status

The shared `loop/run_rounds.py` remains an intentional Phase-0 stub and has no attack execution or
artifact-writing hook. A3 does not redesign it. The exact integration boundary and future steps are
documented in `ORCHESTRATOR_INTEGRATION.md`; all A3 evaluation, presentation, and fidelity commands
run independently today.
