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

### External-reference validation

For showcase metrics, place an authorized Sparkov or IEEE-CIS transaction file under
`data/raw/` and run:

```powershell
python -m attacks.a1_upi_mule.experiment `
  --reference data/raw/fraudTrain.csv `
  --population-events 6000 `
  --output attacks/a1_upi_mule/results/iteration8_metrics_external.json
```

Use `data/raw/train_transaction.csv` for IEEE-CIS. The loader recognizes both schemas,
filters source fraud rows from legitimate baseline calibration, deterministically caps
very large references at 500,000 rows, and records a SHA-256 digest plus row counts. The
Iteration 11 demo should display an **External reference** badge only when the selected
artifact has `methodology.reference.kind == "external_reference"`.

## Corrected development measurement — important limitation

The completed rerun uses a small development amount/MCC fixture because Sparkov or
IEEE-CIS has not been placed under `data/raw/`. The fixture now gives legitimate
first-time UPI payees plausible payee-addition timestamps and uses a mixed
new/established/old beneficiary-age distribution. This removes the two known label
shortcuts from the original development run. These figures validate the experiment
pipeline and provide cached inputs for the replay demo; they are **not external-dataset
fidelity or production claims**.

| Experiment | Recall | Precision | AUC-PR | Recall @ FPR 0.001 |
|---|---:|---:|---:|---:|
| Round 0 | 1.000 | 0.994 | 1.000 | 1.000 |
| Adapted, frozen detector | 0.000 | 0.000 | 0.406 | 0.000 |
| Adapted, old features retrained | 1.000 | 0.995 | 1.000 | 0.995 |
| Adapted, durable features | 1.000 | 0.994 | 1.000 | 1.000 |

The frozen Round-0 detector collapses on the adapted campaign at its original threshold,
while both retrained variants recover on held-out adapted campaigns. Old-feature recovery
is nearly perfect in this synthetic fixture, so the result does not prove the durable
features are uniquely responsible for recovery. The durable model nevertheless restores
recall and provides graph/coercion signals for when node-level signals weaken.

## Current signal-mortality table

| Signal | Round-0 AUC | Adapted AUC | Current verdict |
|---|---:|---:|---|
| `a1_benef_account_age_days` | 0.989 | 0.962 | ALIVE, weaker after adaptation |
| `a1_benef_fanin_24h` | 0.725 | 0.723 | ALIVE |
| `a1_dissipation_ratio` | 0.816 | 0.734 | DEGRADED |
| `a1_payer_amount_zscore` | 0.692 | 0.612 | DEGRADED |
| `a1_payee_added_to_transfer_seconds` | 0.512 | 0.512 | DEAD / near random |
| `a1_component_size` | — | 1.000 | NEW, strong in synthetic fixture |
| `a1_cashout_convergence` | — | 1.000 | NEW, strong in synthetic fixture |
| `a1_call_active` | — | 0.533 | Weak when all laundering hops are scored |

This table intentionally reports what was measured instead of copying illustrative AUCs
from the attack brief.

## Known limitations

- Replace the development fixture with real reference-calibrated traffic.
- Validate the synthetic payee-addition and beneficiary-age assumptions against that
  reference source.
- Decide and report whether authorization-time and delayed-monitoring rows are evaluated
  separately; combining every laundering hop dilutes the victim-only call signal.
- Validate the graph implementation on production-scale samples and record memory/time.
- Keep the demo tied to the regenerated JSON artifact rather than hard-coded metrics.

## Reproduction

```powershell
.venv\Scripts\Activate.ps1
python -m pytest attacks/a1_upi_mule/tests/test_a1.py -q -p no:cacheprovider
python -m pytest tests/test_contract.py -k a1 -v -p no:cacheprovider
python -m ruff check attacks/a1_upi_mule
python -m attacks.a1_upi_mule.experiment --population-events 6000
```

## Adaptive three-round loop

The shared orchestrator runs offline pure CEM by default and writes the frozen output
contract for every round:

```powershell
python -m loop.run_rounds --rounds 0 1 2 --seed 42
```

Each `results/round_N/` directory contains `events.parquet`, `features.parquet`,
`scores.parquet`, `metrics.json`, and `redteam_reasoning.json`. Offline mode deliberately
records a Codex-disabled failure and continues through deterministic CEM, proving that a
missing CLI, authentication problem, timeout, invalid response, or rejected proposal
cannot stop the loop.

The opt-in authenticated reasoner is enabled with `--use-codex`. It sends a bounded
detector report, parameter semantics, metrics, and a small set of synthetic examples to
the configured Codex service. Run that mode only after explicit approval for this data
egress. A one-time approved audit completed successfully with Codex CLI
`0.151.0-alpha.7.2`; the schema-valid response was accepted with `fallback_used: false`.
The judging dashboard never enables it and only replays cached output.

## 60-second pitch

> A fake police officer video-calls a victim and frightens them into transferring money.
> The customer's own phone, PIN, fingerprint, and authorization all pass because the
> customer truly initiated the payment. We therefore look beyond authentication to the
> receiving network: new beneficiaries, multi-sender fan-in, rapid movement, and shared
> cash-out infrastructure. When the attacker ages accounts and spreads funds wider, the
> infrastructure cost rises sharply. Our durable design combines network convergence with
> payer-side coercion signals such as an active call. The final claim will use real
> held-out campaigns. In our corrected synthetic development run, adaptation drops the
> frozen detector's recall from 100% to 0%; retraining with durable graph and coercion
> features restores it to 100%. We label these as development results until they are
> validated against an external reference dataset.
