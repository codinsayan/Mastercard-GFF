# A4 — Synthetic identity bust-out

This package simulates generated identities that establish normal-looking credit histories,
receive higher limits, and then draw those limits down in a coordinated window. It includes the
deterministic generator, static and behavioral features, a two-layer GRU trajectory encoder,
DBSCAN clustering, detector heads, Codex-plus-CEM adaptation, three-round evaluation, and a
Streamlit test interface.

## Manual test

From the repository root:

```bash
.venv/bin/streamlit run attacks/a4_synth_identity/manual_app.py
```

Use the sidebar to compare a shared-attribute Round 0 campaign with a diversified Round 1
campaign. The interface shows an individual 120-day trajectory, group signals, account scores,
generated rows, and the precomputed full-round results. Its interactive scores are fitted
in-sample for visual inspection; only the independent train/test experiment below supplies report
metrics.

## Reproduce the experiment

The quick command is an offline smoke test and uses the deterministic CEM fallback:

```bash
.venv/bin/python -m attacks.a4_synth_identity.experiment --quick --output /tmp/a4-quick
```

The full command uses the locally authenticated Codex CLI through the ChatGPT subscription. It
does not use OpenAI API credits:

```bash
codex login status
.venv/bin/python -m attacks.a4_synth_identity.experiment --codex --seed 42
```

Each round writes `events.parquet`, `features.parquet`, `scores.parquet`, and `metrics.json` under
`results/a4/round_N/`. A4 writes under its own subdirectory because `results/round_N/` is reserved
by the frozen round format for the merged all-attack run. Rounds 1 and 2 also cache the full reasoner exchange in
`redteam_reasoning.json`, so a demo can replay it without a live model or credentials.

## Reference data

Population marginals are fitted to the public **Sparkov** transaction set (1,289,169 legitimate
rows, 14 merchant categories), committed at `data/processed/sparkov_reference.parquet` and
attributed in `data/README.md`. It needs no setup and is shared by all four attacks, not just
A4. Fitting to real marginals rather than an inline table moves the generated amount
distribution from KS 0.351 to **KS 0.132** against the real data.

Should the file be missing, the experiment falls back to a small inline stand-in and prints a
warning; every `a4_summary.json` records which reference produced it under the `reference` key,
so results are auditable. Only runs with `"is_real_reference_data": true` are reportable.

## Measured result

Independent train/test measurements from the full seed-42 run on 31 August 2026, fitted to the
Sparkov reference. Both reasoning calls used Codex CLI 0.144.5 with the configured ChatGPT model;
neither used the fallback. Outputs are written under `results/a4/`.

| Stage | Recall | Precision | AUC-PR | FPR at 85% recall |
|---|---:|---:|---:|---:|
| Round 0 static detector | 0.971 | 1.000 | 0.981 | 0.000 |
| Round 1 static detector, before fix | 0.000 | 0.000 | 0.334 | 1.000 |
| Round 1 behavioral detector, after fix | 1.000 | 1.000 | 1.000 | 0.000 |
| Round 2 behavioral detector, before recovery | 0.162 | 1.000 | 0.998 | 0.001 |
| Round 2 full detector, after recovery | 0.995 | 1.000 | 0.997 | 0.000 |

Round 2's "before recovery" row is a threshold effect, not a ranking collapse: AUC-PR stays at
0.998, so the detector still ranks the evolved campaign correctly and only the 0.5 cut-point
moves. Round 1 is the genuine collapse, at AUC-PR 0.334.

### Signal mortality

| Signal | Round 0 AUC | Round 1 AUC | Round 2 AUC | Verdict |
|---|---:|---:|---:|---|
| `a4_shared_address_token_count` | 0.843 | 0.500 | 0.500 | dead |
| `a4_shared_device_prefix_count` | 0.847 | 0.500 | 0.500 | dead |
| `a4_phone_block_density` | 0.854 | 0.500 | 0.500 | dead |
| `a4_account_age_at_first_large_txn` | 0.586 | 0.535 | 0.558 | never discriminative |
| `a4_trajectory_cluster_density` | 0.590 | 0.711 | 0.775 | replacement |
| `a4_trajectory_dist_to_centroid` | 0.590 | 0.606 | 0.717 | replacement |
| `a4_trajectory_cluster_size` | 0.590 | 0.711 | 0.775 | replacement |
| `a4_ramp_convexity` | 0.991 | 0.687 | 0.845 | durable |
| `a4_payment_ratio_regularity` | 0.996 | 1.000 | 1.000 | **leaking — see below** |
| `a4_bustout_sync_score` | 1.000 | 1.000 | 0.997 | **suspect — see below** |

Round-0 AUCs are shown for every signal, including behavioral ones deliberately excluded from
the round-0 detector. The mortality claim concerns the three static signals, which fall to
exactly 0.500 once attribute overlap reaches zero.

The first adaptation drove `attribute_overlap_rate` to 0.0 and `playbook_variance` to its 0.8
ceiling, killing all three static signals. The trajectory features are the genuine replacement:
they *strengthen* across rounds (0.590 to 0.775) as the campaign is forced to spread out, which
is the intended behaviour of a behavioral signal against an attacker abandoning shared
attributes.

**Round 2 did not extract a cost from the attacker.** Utility stayed at 0.738 (round 1: 0.781;
round 0: 0.732), because the two leaking features below could not be broken by any parameter
move, so the reasoner had no incentive gradient to trade against. The earlier claim that
evasion forces build time up and payoff down is **not supported by this run** and must not be
quoted until the leak is fixed. The evolved campaign passed the declared budgets: amount KS
0.093 (limit 0.350), utility above 0.10.

### Known leak — do not quote these two signals

`a4_payment_ratio_regularity` is not measuring repayment regularity. `features.py` defines the
build window as `utilization < 0.85`, but the first bust-out burst is emitted at utilization
0.52 with `payment_ratio = 0.0`, so **exactly one bust-out row per synthetic account leaks into
its build statistics**. That single zero dominates the standard deviation:

| One synthetic account | Payment std | Regularity |
|---|---:|---:|
| Build window as currently defined | 0.1906 | 0.0499 |
| True build rows only | 0.0079 | 0.5602 |

The feature therefore detects *a missed payment* — the bust-out itself — rather than regularity,
which is why it separates in the opposite direction to its stated rationale and scores ~1.000.
The fix is to define the build window by phase or by the first utilization spike rather than by
a fixed 0.85 threshold.

`a4_bustout_sync_score` remains near 1.000. The structural defect is fixed — organic utilization
is no longer clipped below the 0.85 test line, and 13.3% of legitimate accounts now genuinely
cross it — but synthetic clusters still spike far more tightly than any organic cohort, so
ranking separation stays almost total. Treat it as unproven until it has been measured against
an organic population with correlated seasonal spending.

### What the population fix did change

Fitting the organic population properly (per-customer utilization peaks, heterogeneous repayment
behaviour, and a history depth comparable to a synthetic account: median 19 events per organic
account against 28 per synthetic one) removed a third artifact outright.
`a4_account_age_at_first_large_txn` fell from 0.807/0.972 to 0.586/0.535 — it had been separating
on the organic population's shallow history, and it is now correctly shown as non-discriminative.
The trajectory and convexity signals are the ones that survived on merit.

## Verification

```bash
PYTHONPATH=. .venv/bin/pytest -q attacks/a4_synth_identity/tests/test_a4.py
```

The tests cover schema validity, seeded determinism, synchronized bust-out timing, static-signal
mortality, the exact two-layer/32-hidden-unit GRU architecture, deterministic embeddings,
detector bounds, and both Streamlit presets.
