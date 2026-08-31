# A4 — Synthetic identity bust-out

This package simulates generated identities that establish normal-looking credit histories,
receive higher limits, and then draw those limits down in a coordinated window. It includes the
deterministic generator, static and behavioral features, a two-layer GRU trajectory encoder,
DBSCAN clustering, detector heads, Codex-plus-CEM adaptation, three-round evaluation, and a
Streamlit test interface.

## The two apps

There are two Streamlit interfaces and they are not interchangeable.

### `demo_app.py` - for judges

```bash
.venv/bin/streamlit run attacks/a4_synth_identity/demo_app.py
```

Five screens, replay only. It reads `results/a4/` and nothing else: no model is trained, no
events are generated, Codex is never called and no network is touched, which is what the team
brief requires of the judging demo. A test enforces this structurally by parsing its imports.

1. **The innocent transaction** - one build-phase row displayed large, beside the same account's
   120-day arc. The argument for why a sequence model is required.
2. **Moving in step** - one synthetic cluster's utilization curves against organic customers,
   plus when each account first crosses 85%.
3. **Signal mortality** - the 10 signals by 3 rounds heatmap. The money shot.
4. **The attacker's words** - Codex's verbatim plan beside the damage it caused and the cost it
   paid.
5. **Scoreboard and caveats** - operating points, score distributions, and the four things we
   explicitly do not claim.

Each screen carries a presenter note in grey suggesting what to say while it is on screen.

### `manual_app.py` - for development

```bash
.venv/bin/streamlit run attacks/a4_synth_identity/manual_app.py
```

Sliders that regenerate and re-score a campaign live, for eyeballing generator changes. Its
scores are fitted **in-sample** and are not evaluation metrics; it carries a banner saying so.
Useful for showing attribute overlap collapsing in real time, but do not put it in front of
judges.

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

After any run, refresh the committed demo assets so the judge demo reflects it:

```bash
.venv/bin/python -m attacks.a4_synth_identity.build_demo_assets
```

That distils the round outputs into `results/a4/demo_assets.json` (~43 KB, committed), because
the `events.parquet` files are git-ignored and would leave the demo empty on a fresh clone.

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

Independent train/test measurements from the full seed-42 run, fitted to the Sparkov reference.
Both reasoning calls used Codex CLI 0.144.5 with the configured ChatGPT model; neither used the
fallback. Outputs are written under `results/a4/`.

| Stage | AUC-PR | Recall at FPR 0.001 | FPR at 85% recall | Recall at 0.5 |
|---|---:|---:|---:|---:|
| Round 0 static detector | 0.981 | 0.971 | 0.000 | 0.971 |
| Round 1 static detector, before fix | **0.334** | 0.000 | 1.000 | 0.000 |
| Round 1 behavioral detector, after fix | 1.000 | 1.000 | 0.000 | 1.000 |
| Round 2 behavioral detector, before recovery | 0.936 | 0.000 | 0.034 | 1.000 |
| Round 2 retrained detector, after recovery | 0.943 | 0.408 | 0.041 | 0.408 |

**Read the operating points, not the 0.5 column.** A fixed 0.5 threshold is misleading here: at
round 2 the pre-recovery detector scores recall 1.000 at 0.5 but **0.000** at FPR 0.001 -- it is
uncalibrated and simply flags almost everyone. The retrained detector scores 0.408 at that same
strict operating point, which is the better model despite the worse-looking headline. AUC-PR and
FPR-at-recall are the honest comparisons and are what the team brief asks for.

### Signal mortality

| Signal | Round 0 AUC | Round 1 AUC | Round 2 AUC | Verdict |
|---|---:|---:|---:|---|
| `a4_shared_address_token_count` | 0.843 | 0.505 | 0.510 | dead |
| `a4_shared_device_prefix_count` | 0.847 | 0.502 | 0.504 | dead |
| `a4_phone_block_density` | 0.854 | 0.506 | 0.500 | dead |
| `a4_account_age_at_first_large_txn` | 0.586 | 0.616 | 0.828 | strengthens |
| `a4_trajectory_cluster_density` | 0.590 | 0.629 | 0.734 | replacement |
| `a4_trajectory_dist_to_centroid` | 0.590 | 0.644 | 0.610 | replacement |
| `a4_trajectory_cluster_size` | 0.590 | 0.629 | 0.734 | replacement |
| `a4_ramp_convexity` | 0.986 | 0.984 | 0.986 | durable |
| `a4_payment_ratio_regularity` | 0.967 | 0.971 | **0.600** | attacked successfully |
| `a4_bustout_sync_score` | 1.000 | 1.000 | 0.978 | durable, see caveat |

Round-0 AUCs are shown for every signal, including behavioral ones deliberately excluded from
the round-0 detector. The mortality claim concerns the three static signals, which fall to
chance.

### The two-round arc

**Round 1 -- evasion is free.** The reasoner drove `attribute_overlap_rate` to ~0.0 and killed
all three static signals at once. Attacker utility was *unchanged* (0.732 to 0.723): diversifying
generated identity attributes costs a generative model nothing. Detection collapsed to AUC-PR
0.334.

**Round 2 -- evasion has a price.** With only behavioral signals left, the reasoner had to attack
behaviour itself: `playbook_variance` to its 0.8 ceiling and a flattened utilization ramp. It
worked -- `a4_payment_ratio_regularity` fell from 0.971 to 0.600, and holding 85% recall now
costs 4.1% false positives instead of 0%. But it **cost the attacker 14% of its payoff**
(utility 0.723 to 0.621), because randomising the playbook is the same thing as giving up the
automation that makes the scheme scalable.

> Identity attributes are cheap for GenAI to diversify. Operating strategy is expensive, because
> randomising strategy degrades the attack's own payoff.

That contrast -- a free evasion in round 1, a paid one in round 2 -- is the result this attack
exists to produce. The evolved campaign stayed inside the declared budgets: amount KS 0.033
against a 0.350 limit, utility well above the 0.10 floor.

### Fixed: the build-window leak

Earlier runs reported `a4_payment_ratio_regularity` at ~1.000 and showed no attacker cost in
round 2. The cause was in `features.py`: the build window was defined as `utilization < 0.85`,
but a draw-down opens *below* that line before crossing it, so the first bust-out row -- carrying
a zero repayment -- fell back inside each account's build statistics. The feature was detecting a
missed payment, and because no parameter move can hide a missed payment, the red agent had no
gradient to trade against.

The window is now every row before an account's first utilization spike, derived from observed
utilization only and never from the generator's phase label. The feature now separates in the
direction its description claims -- synthetic accounts repay *more* consistently than organic
ones (0.55 against 0.17) -- and round 2 extracts a real cost.
`tests/test_a4.py::test_build_window_excludes_every_bustout_row` is the regression guard.

### Remaining caveat

`a4_bustout_sync_score` still measures 0.978-1.000. The structural defect is gone -- organic
utilization is no longer clipped below the test line, and 13.3% of legitimate accounts genuinely
cross it -- but synthetic clusters still spike far more tightly than any organic cohort. Treat it
as unproven until measured against an organic population with correlated seasonal spending.

Fraud prevalence in the evaluation set is also far above the real ~0.6%, so precision is
optimistic. Both caveats are stated on the demo's final screen.

## Verification

```bash
PYTHONPATH=. .venv/bin/pytest -q attacks/a4_synth_identity/tests/test_a4.py
```

The tests cover schema validity, seeded determinism, synchronized bust-out timing, static-signal
mortality, the exact two-layer/32-hidden-unit GRU architecture, deterministic embeddings,
detector bounds, and both Streamlit presets.
