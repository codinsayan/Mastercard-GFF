# A1 — Deepfake Coercion to UPI Mule Cash-Out

## Attack summary

A victim receives a deepfake video or voice call from someone impersonating an authority.
Under coercion, the victim sends money through UPI from their genuine account, device,
location, PIN, and biometric session. Authentication succeeds because the customer truly
authorized the payment. Detection must therefore examine the beneficiary, subsequent
money movement, and observable signs of coercion.

This package is an authorized simulation using synthetic payment events. It contains no
victim-contact, credential-access, or real payment functionality.

## Generated topology

```text
Victim → layer-1 mule → layer-2 mules → shared cash-out points
```

The generator creates one to three victim payments, delays layer-1 movement using
`dwell_time_hours`, splits funds using `split_factor`, and reconverges layer-2 transfers
on a bounded cash-out pool. It retains 2% at layer 1 and 3% at layer 2, ensuring causal
money conservation at every hop.

## Attack parameters

| Parameter | Default | Search range | Economic trade-off |
|---|---:|---:|---|
| `n_victims` | 400 | 20–2000 | More yield, but more coercion operations and exposure |
| `mule_age_days` | 2 | 1–120 | Older accounts blend in but cost more to source |
| `split_factor` | 3 | 2–25 | Lower node fan-in but more mule infrastructure |
| `dwell_time_hours` | 0.2 | 0.05–48 | Less instant movement but more freeze exposure |
| `amount_multiplier_vs_baseline` | 8.0 | 2–15 | More yield but a larger payer anomaly |
| `n_cashout_points` | 5 | 3–60 | Less convergence but costly withdrawal infrastructure |
| `n_mules_layer1` | 60 | 20–400 | More distribution but greater acquisition cost |

## Utility

Utility measures successful approved cash-out value per unit of infrastructure cost.
Unique mule cost increases with account age; cash-out points have a higher fixed cost.
Raw payoff is normalized with a stable saturating function into `[0,1]`.

The development experiment measured:

| Scenario | Mules | Cash-out points | Cash-out value | Utility |
|---|---:|---:|---:|---:|
| Round 0 | 80 | 5 | ₹63,928.25 | 0.194 |
| Adapted | 320 | 20 | ₹65,968.07 | 0.047 |

The adapted campaign moved a similar amount but required four times as many mule accounts
and cash-out points, reducing normalized utility by approximately 76%.

## Features

### Round-0 signals

| Feature | Meaning | Availability |
|---|---|---|
| `a1_benef_account_age_days` | Receiving-account age | Authorization time |
| `a1_benef_fanin_24h` | Distinct prior payers in 24 hours | Authorization time |
| `a1_dissipation_ratio` | Outflow within one hour divided by inflow | Delayed monitoring |
| `a1_payer_amount_zscore` | Amount relative to prior payer behavior | Authorization time |
| `a1_payee_added_to_transfer_seconds` | Delay between payee addition and transfer | Authorization time |

### Durable and component signals

| Feature | Meaning | Availability |
|---|---|---|
| `a1_component_size` | Beneficiary's two-hop transfer neighborhood | Delayed monitoring |
| `a1_component_dissipation` | Component outflow divided by component inflow | Delayed monitoring |
| `a1_cashout_convergence` | Distinct upstream paths reaching a shared terminal node | Delayed monitoring |
| `a1_component_age_variance` | Account-age variance in the local component | Delayed monitoring |
| `a1_call_active` | Call active during the payment | Authorization time |
| `a1_session_duration_anomaly` | Session duration versus prior payer history | Authorization time |

Graph signals use an indexed local neighborhood within ±48 hours. The future portion of
that window is explicitly a delayed investigation/intervention signal and is not claimed
as authorization-time information.

## Detector

`A1Detector` uses deterministic LightGBM with balanced classes. It stores its training
feature schema, aligns reordered input, fills missing trained columns, ignores extras,
and returns finite fraud probabilities. Tiny or single-class inputs use a constant-prior
fallback so the integration loop cannot fail on a small fold.

## Evaluation method

Run:

```powershell
python -m attacks.a1_upi_mule.experiment --population-events 6000
```

The runner uses:

- separate seeds for default train/test and adapted train/test campaigns;
- earliest 70% of legitimate events for training and latest 30% for testing;
- a threshold learned from training legitimate scores at target FPR 0.001;
- a frozen Round-0 detector against adapted campaigns;
- old-feature retraining on adapted campaigns;
- durable-feature retraining on adapted campaigns;
- per-feature AUC and attacker-infrastructure measurement.

The full measured artifact is stored in
[`results/iteration8_metrics.json`](results/iteration8_metrics.json).

## Development measurement — important limitation

The current run uses a small development amount/MCC fixture because Sparkov or IEEE-CIS
has not yet been placed under `data/raw/`. These figures validate the experiment pipeline;
they are **not final fidelity or judging claims**.

| Experiment | Recall | Precision | AUC-PR | Recall @ FPR 0.001 |
|---|---:|---:|---:|---:|
| Round 0 | 1.000 | 0.988 | 1.000 | 1.000 |
| Adapted, frozen detector | 1.000 | 0.997 | 1.000 | 1.000 |
| Adapted, old features retrained | 1.000 | 0.992 | 1.000 | 1.000 |
| Adapted, durable features | 1.000 | 0.992 | 1.000 | 1.000 |

These perfect results are a diagnostic warning, not a success claim. The expected signal
mortality did not occur. The development data contains two shortcuts:

1. legitimate rows have no payee-addition metadata, while A1 rows always have it, making
   `a1_payee_added_to_transfer_seconds` perfectly discriminative;
2. the shared development population gives beneficiaries broadly uniform ages, making a
   45-day adapted mule still unusually young.

Before final measurement, legitimate traffic must receive realistic payee-addition
history and beneficiary-age marginals must be fitted to Sparkov/IEEE-CIS or another
defensible public source. The frozen-detector experiment must then be rerun. Do not use
the current perfect metrics in the pitch as evidence of adaptation or recovery.

## Current signal-mortality table

| Signal | Round-0 AUC | Adapted AUC | Current verdict |
|---|---:|---:|---|
| `a1_benef_account_age_days` | 0.999 | 0.993 | NOT DEAD — fixture shortcut remains |
| `a1_benef_fanin_24h` | 0.725 | 0.723 | ALIVE |
| `a1_dissipation_ratio` | 0.816 | 0.734 | DEGRADED |
| `a1_payer_amount_zscore` | 0.692 | 0.612 | DEGRADED |
| `a1_payee_added_to_transfer_seconds` | 1.000 | 1.000 | INVALID SHORTCUT in development fixture |
| `a1_component_size` | — | 1.000 | NEW, requires realistic rerun |
| `a1_cashout_convergence` | — | 1.000 | NEW, requires realistic rerun |
| `a1_call_active` | — | 0.533 | Weak when all laundering hops are scored |

This table intentionally reports what was measured instead of copying illustrative AUCs
from the attack brief.

## Known limitations and required final rerun

- Replace the development fixture with real reference-calibrated traffic.
- Populate legitimate payee-addition metadata from a realistic distribution.
- Decide and report whether authorization-time and delayed-monitoring rows are evaluated
  separately; combining every laundering hop dilutes the victim-only call signal.
- Validate the graph implementation on production-scale samples and record memory/time.
- Rerun all four experiments using held-out campaigns.
- Update this README and the demo only from the regenerated metrics artifact.

## Reproduction

```powershell
.venv\Scripts\Activate.ps1
python -m pytest attacks/a1_upi_mule/tests/test_a1.py -q -p no:cacheprovider
python -m pytest tests/test_contract.py -k a1 -v -p no:cacheprovider
python -m ruff check attacks/a1_upi_mule
python -m attacks.a1_upi_mule.experiment --population-events 6000
```

## 60-second pitch

> A fake police officer video-calls a victim and frightens them into transferring money.
> The customer's own phone, PIN, fingerprint, and authorization all pass because the
> customer truly initiated the payment. We therefore look beyond authentication to the
> receiving network: new beneficiaries, multi-sender fan-in, rapid movement, and shared
> cash-out infrastructure. When the attacker ages accounts and spreads funds wider, the
> infrastructure cost rises sharply. Our durable design combines network convergence with
> payer-side coercion signals such as an active call. The final claim will use real
> reference-calibrated traffic and held-out campaigns; our current development run has
> already identified and documented metadata shortcuts that must be removed before we
> present performance numbers.

