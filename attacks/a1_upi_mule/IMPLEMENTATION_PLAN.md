# A1 — Deepfake Coercion to UPI Mule Cash-Out

## Detailed Iterative Implementation Plan

This document is the working implementation guide for Person A. It translates
`01_PERSON_A_deepfake_upi_mule (1).md` and the shared integration contract into a
sequence of independently verifiable iterations.

The implementation owned by Person A is restricted to:

```text
attacks/a1_upi_mule/
├── __init__.py
├── card.json
├── generator.py
├── features.py
├── detector.py
├── README.md
├── IMPLEMENTATION_PLAN.md
└── tests/
    └── test_a1.py
```

Do not modify `core/`, `redteam/`, `loop/`, shared tests, or another person's attack
folder during the A1 implementation. Shared Phase 3 work must be coordinated after the
individual attack passes its merge gate.

---

## 1. Final objective

Build a deterministic simulation of a victim-authorized UPI fraud campaign in which:

1. A victim is coerced during a deepfake call.
2. The victim uses their genuine device and credentials.
3. The victim sends an unusually large payment to a newly added layer-1 mule.
4. Layer-1 mules wait, then distribute funds across layer-2 mules.
5. Layer-2 mules reconverge on shared cash-out infrastructure.
6. A Round-0 detector catches obvious new-account, fan-in, and rapid-dissipation signals.
7. The red agent ages and spreads the mule network, causing those signals to die.
8. Component-level graph signals and durable payer-behavior signals restore detection.

The completed package must provide three shared-interface objects:

```python
GENERATOR
FEATURE_BLOCK
DETECTOR
```

The final A1 merge gate is:

```powershell
.venv\Scripts\Activate.ps1
python -m pytest tests/test_contract.py -k a1 -v -p no:cacheprovider
```

The A1 test must pass. A skipped test means the package exports have not been wired and
the implementation is not merge-ready.

---

## 2. Decisions that resolve ambiguities in the specification

### 2.1 Add `n_victims` to `param_space`

The attack specification includes `n_victims` in `default_params` but omits it from
`param_space`. The shared contract requires exact key equality between:

```text
default_params()
param_space()
card.json.default_params
card.json.param_space
card.json.param_semantics
```

Add this range:

```json
"n_victims": [20, 2000]
```

Its semantic description should explain that more victims increase campaign yield and
exposure while requiring more concurrent coercion operations.

### 2.2 Use one canonical payee-delay feature

The specification uses both `a1_payee_age_seconds` and
`a1_payee_added_to_transfer_seconds` for the same concept. Use only:

```text
a1_payee_added_to_transfer_seconds
```

Use this exact name in the feature builder, detector, card, state description, tests,
metrics, and README.

### 2.3 Define `a1_mule_layer` as the destination stage

Use these values:

| Value | Transfer |
|---:|---|
| `1` | victim → layer-1 mule |
| `2` | layer-1 mule → layer-2 mule |
| `3` | layer-2 mule → cash-out point |

Victim-originated value is therefore the sum of rows where `a1_mule_layer == 1`.
Cashed-out value is the sum of rows where `a1_mule_layer == 3`.

### 2.4 Separate real-time and delayed signals

`a1_dissipation_ratio` observes what happens after funds arrive. It is not available at
the exact instant of the victim's first payment. Treat it as a near-real-time mule
monitoring signal and state this limitation in the README.

The following are available at payment authorization time:

- beneficiary account age;
- prior beneficiary fan-in;
- payer amount anomaly;
- payee-addition delay;
- call-active state;
- session-duration anomaly.

Graph convergence and dissipation become available after subsequent movement occurs.

### 2.5 Treat `population` as a DataFrame

The shared contract passes a legitimate-event `pandas.DataFrame` into `generate()`.
Do not require a custom population class. The generator should derive customer history
and normal attributes directly from this DataFrame.

---

## 3. Canonical parameters

### 3.1 Defaults

```json
{
  "n_victims": 400,
  "mule_age_days": 2,
  "split_factor": 3,
  "dwell_time_hours": 0.2,
  "amount_multiplier_vs_baseline": 8.0,
  "n_cashout_points": 5,
  "n_mules_layer1": 60
}
```

### 3.2 Search space

```json
{
  "n_victims": [20, 2000],
  "mule_age_days": [1, 120],
  "split_factor": [2, 25],
  "dwell_time_hours": [0.05, 48.0],
  "amount_multiplier_vs_baseline": [2.0, 15.0],
  "n_cashout_points": [3, 60],
  "n_mules_layer1": [20, 400]
}
```

### 3.3 Parameter type policy

Treat these as integers:

- `n_victims`;
- `mule_age_days`;
- `split_factor`;
- `n_cashout_points`;
- `n_mules_layer1`.

Treat these as floats:

- `dwell_time_hours`;
- `amount_multiplier_vs_baseline`.

Validate parameter types and bounds at the beginning of `generate()`. Reject booleans,
even though Python considers `bool` a subclass of `int`.

---

# Iteration 0 — Package and configuration contract

## Goal

Create an importable A1 package with a single canonical configuration source.

## Work items

### Populate `card.json`

Include:

- `attack_id`;
- `name`;
- `channel`;
- `attack_summary`;
- `default_params`;
- `param_space`;
- `param_semantics`;
- `defender_signals`.

Initially list every planned feature in `defender_signals`, because the shared contract
requires exact coverage of all emitted feature columns.

Planned canonical features:

```text
a1_benef_account_age_days
a1_benef_fanin_24h
a1_dissipation_ratio
a1_payer_amount_zscore
a1_payee_added_to_transfer_seconds
a1_component_size
a1_component_dissipation
a1_cashout_convergence
a1_component_age_variance
a1_call_active
a1_session_duration_anomaly
```

### Configuration loading

Load the card relative to the module path:

```python
CARD_PATH = Path(__file__).with_name("card.json")
```

Return copies from configuration methods so callers cannot mutate shared state.

### Package exports

Do not export instantiated components until `generator.py`, `features.py`, and
`detector.py` all import successfully. Once ready, `__init__.py` should contain:

```python
from .detector import A1Detector
from .features import A1Features
from .generator import A1Generator

GENERATOR = A1Generator()
FEATURE_BLOCK = A1Features()
DETECTOR = A1Detector()
```

## Tests

- `card.json` parses successfully.
- Default and range keys match exactly.
- Every parameter has semantics.
- Every planned feature has a defender description.
- `attack_id == "A1"` and `channel == "UPI_P2P"`.

## Exit criteria

- Package imports without side effects.
- Configuration has no duplicated source of truth.
- All key-set assertions pass.

---

# Iteration 1 — Minimal deterministic generator

**Status: COMPLETE** — implemented and verified with deterministic schema tests.

## Goal

Generate a small, schema-valid victim-to-layer-1 flow before implementing the complete
network. This establishes the hardest integration requirement early.

## Class contract

```python
class A1Generator:
    attack_id = "A1"

    def default_params(self) -> dict: ...
    def param_space(self) -> dict[str, tuple[float, float]]: ...
    def generate(self, params: dict, population: pd.DataFrame, seed: int) -> pd.DataFrame: ...
    def utility(self, df: pd.DataFrame) -> float: ...
    def describe_state(self, params: dict) -> dict: ...
```

## Input validation

Validate:

1. `population` is a non-empty DataFrame.
2. It has the fields needed to copy legitimate customer characteristics.
3. Parameter keys exactly match the expected set.
4. Numeric values have the correct type and lie inside their ranges.
5. Counts are positive and structurally valid.
6. There are enough eligible customers, or a documented deterministic cap is applied.

Use only:

```python
rng = np.random.default_rng(seed)
```

Do not use `random`, `np.random.seed`, current time, UUID4, process IDs, or unordered set
iteration when creating output.

## Customer-history preparation

From approved legitimate transactions, derive one deterministic record per customer:

- prior mean and median amount;
- most recent known device;
- most recent ASN;
- most recent VPA;
- account age;
- latest historical timestamp;
- normal session-duration median and dispersion.

Sort before grouping so “most recent” is stable. Exclude customers without usable
amount history.

## Victim selection

Sample without replacement from sorted eligible customer IDs. When a tiny contract-test
population contains fewer customers than `n_victims`, cap the effective victim count to
the eligible count. This keeps the default configuration compatible with the shared
200-event population fixture.

Document the requested and effective victim counts in internal variables, but do not add
shared schema columns.

## Minimal transfer generation

For each victim:

1. Choose 1–3 payments deterministically from the RNG.
2. Assign a layer-1 mule.
3. Calculate a campaign total using the customer's historical baseline.
4. Split the total across the chosen number of payments.
5. Use the victim's legitimate device and network attributes.
6. Create a new beneficiary VPA.
7. Add the payee shortly before the transaction.
8. Simulate a long coercion session and active call.

Suggested campaign amount:

```python
baseline = max(customer_mean_amount, minimum_baseline_amount)
campaign_total = baseline * amount_multiplier_vs_baseline
```

Use bounded random proportions that sum to one. Round money only after allocation and
correct the final payment for rounding drift.

## Frozen-schema mapping

Every generated row must provide all required columns. Typical constants:

```text
attack_id = A1
label = 1
round = 0
channel = UPI_P2P
counterparty_type = BENEFICIARY
counterparty_mcc = ""
card_id = ""
card_bin = ""
token_id = ""
auth_result = APPROVED
decline_code = ""
is_first_time_payee = 1
```

At the end:

```python
result = coerce(result)
result = result.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
validate(result)
return result
```

## Metadata

Every row must contain a fresh dictionary:

```python
{
    "a1_call_active": True,
    "a1_payee_added_at": "2026-...",
    "a1_mule_layer": 1,
    "a1_cashout_node": "a1_cashout_0003",
    "a1_campaign_id": "a1_camp_000007",
    "a1_source_role": "VICTIM",
    "a1_destination_role": "MULE_L1"
}
```

Use ISO-8601 strings for timestamps inside `meta` to remain JSON-serializable.

## Tests

- Frozen schema validation passes.
- Same seed and input produce exactly equal DataFrames.
- Different seeds alter generated data.
- IDs are unique.
- All rows have `attack_id == "A1"` and `label == 1`.
- All rows use `UPI_P2P`.
- Every metadata value is a dictionary.
- Metadata dictionaries are not shared between rows.
- Amounts are finite and positive.
- Tiny populations work through deterministic capping.

## Exit criteria

A deterministic victim-to-layer-1 generator passes local tests and shared schema
validation.

---

# Iteration 2 — Complete layered mule topology

**Status: COMPLETE** — implemented with causal three-stage flow, bounded infrastructure,
retained fees, parameter-effect tests, and campaign money-conservation tests.

## Goal

Extend the minimal generator into the full victim → L1 → L2 → cash-out flow while
preserving time and money.

## Infrastructure creation

Generate deterministic identifiers for:

```text
a1_mule_l1_0001
a1_mule_l2_0001
a1_cashout_0001
a1_vpa_l1_0001@bank
a1_vpa_l2_0001@bank
a1_cashout_0001@bank
```

Map victims to layer-1 mules in a balanced but seeded manner.

## Layer-1 fan-out

For each layer-1 mule:

1. Sum received value.
2. Wait `dwell_time_hours` after the last relevant inflow.
3. Retain or lose a small bounded fee.
4. Split the remainder across `split_factor` layer-2 destinations.
5. Ensure outgoing value never exceeds incoming value.

Use a deterministic split function that handles rounding without creating or destroying
material value.

## Layer-2 convergence

For every layer-2 mule:

1. Sum received value.
2. Wait a short deterministic delay.
3. Send most of the value to one of `n_cashout_points` shared destinations.
4. Assign cash-out points so multiple upstream paths reconverge.

The generator must preserve the structural tension needed for Round 1: spreading makes
individual mules less suspicious, but shared cash-out infrastructure creates a component
signal.

## Account attributes

For synthetic mule accounts:

- set `counterparty_account_age_days` from `mule_age_days`, optionally with small seeded
  variation;
- use non-victim customer IDs for mule-originated transfers;
- create deterministic mule devices, sessions, ASNs, and VPAs;
- use `is_first_time_payee = 1` for newly connected laundering hops;
- set `a1_call_active = False` after the victim stage.

## Transaction time invariants

Assert:

```text
victim payment time < layer-1 outgoing time < layer-2 outgoing time
```

All generated times should remain inside a reasonable extension of the population's
historical time range.

## Conservation invariants

At each mule:

```text
sum(outgoing) <= sum(incoming)
```

For the campaign:

```text
cashed_out <= victim_originated
```

Track a small loss/fee ratio explicitly in code comments and tests.

## Scale protection

Extreme values can create very large graphs. Define a documented maximum number of
layer-2 nodes or reuse layer-2 nodes when necessary. The cap must not be random or depend
on machine resources.

## Tests

Construct a small deterministic campaign and verify:

- all three destination layers exist;
- each campaign has a cash-out node;
- layer transitions are valid;
- timestamps are causally ordered;
- node and campaign value is conserved;
- adapted `split_factor` increases infrastructure usage;
- adapted `dwell_time_hours` increases transfer delay;
- adapted `mule_age_days` changes beneficiary age.

## Exit criteria

The complete graph is structurally correct, deterministic, schema-valid, and bounded at
parameter extremes.

---

# Iteration 3 — Utility and red-agent state

**Status: COMPLETE** — implemented with normalized cash-out yield, age-adjusted mule
cost, cash-out infrastructure cost, safe malformed-data handling, exact state-contract
validation, and economic monotonicity tests.

## Goal

Expose attack economics and plain-English semantics to the adaptive red agent.

## Utility design

Calculate:

```python
victim_value = value entering layer 1
cashed_out_value = value entering layer 3
mule_count = unique synthetic layer-1 and layer-2 accounts used
cashout_count = unique cash-out accounts used
```

Define infrastructure cost:

```python
infrastructure_cost = (
    mule_count * MULE_ACQUISITION_COST
    + cashout_count * CASHOUT_POINT_COST
)
```

Then derive raw payoff:

```python
raw_payoff = cashed_out_value / max(infrastructure_cost, epsilon)
```

Normalize with a stable saturating function:

```python
utility = raw_payoff / (raw_payoff + REFERENCE_PAYOFF)
```

Return a native Python `float` in `[0, 1]`.

Do not use campaign metadata access patterns such as `df.meta.str[...]`; extract from
dictionaries safely so legitimate/missing metadata cannot crash the method.

## Utility behavior tests

- More cashed-out value with unchanged infrastructure increases utility.
- More mules for unchanged value decreases utility.
- More cash-out points for unchanged value decreases utility.
- Empty input returns a documented safe value, normally `0.0`.
- Result is always finite and bounded.

## `describe_state()`

Return exactly:

```python
{
    "attack_summary": str,
    "param_semantics": dict,
    "defender_signals": dict,
    "current_params": dict,
}
```

Do not add a fifth key because the shared contract compares key sets exactly.

Load semantics and signal descriptions from `card.json`, and return the supplied params
as `current_params`.

## Exit criteria

- Utility communicates the cost of excessive mule spreading.
- State output contains every parameter and every emitted feature.
- All shared reasoning-contract checks pass.

---

# Iteration 4 — Round-0 feature block

## Goal

Build leakage-aware features that detect the obvious default campaign.

## FeatureBlock contract

```python
class A1Features:
    attack_id = "A1"

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        ...
```

The result must:

- be indexed by `event_id`;
- retain one feature row per input event;
- preserve input event order;
- use only columns beginning with `a1_`;
- contain finite numeric values;
- be deterministic.

## 4.1 Beneficiary account age

```text
a1_benef_account_age_days
```

Copy and numerically sanitize `counterparty_account_age_days`.

## 4.2 Prior beneficiary fan-in

```text
a1_benef_fanin_24h
```

For each event, count distinct prior `customer_id` values that paid the same
`counterparty_id` during:

```text
[timestamp - 24 hours, timestamp)
```

Exclude the current event and every future event. Resolve equal timestamps using
`event_id` for deterministic ordering, while ensuring same-time future rows are not
accidentally exposed.

## 4.3 Dissipation ratio

```text
a1_dissipation_ratio
```

For the beneficiary account, measure value leaving within one hour relative to value
received. Define the numerator and denominator carefully so internal transfers are not
double-counted.

This feature intentionally observes post-arrival behavior. Mark it as a delayed mule
monitoring signal in the README and evaluation output.

## 4.4 Payer amount anomaly

```text
a1_payer_amount_zscore
```

Calculate the current amount against prior events for the same payer. Do not include the
current event. Prefer a 90-day history where available.

When standard deviation is zero or history is insufficient, use `0.0` rather than
infinity.

## 4.5 Payee-addition delay

```text
a1_payee_added_to_transfer_seconds
```

Safely extract `a1_payee_added_at` from `meta`, parse it, and subtract it from the event
timestamp. Missing, malformed, or future payee-addition values should use a documented
neutral/sentinel value.

## Implementation strategy

Use vectorized pandas or DuckDB for time-window operations. Avoid a Python loop over
every transaction in the full population. Small loops over campaigns or graph windows
are acceptable after filtering relevant events.

## Tests

Use hand-authored timelines to verify:

- current and future senders are excluded from fan-in;
- a sender is counted once even after multiple payments;
- the 24-hour boundary is handled consistently;
- amount z-score uses prior history only;
- malformed metadata does not crash feature building;
- payee delay is calculated exactly;
- output index and prefix contract are correct.

## Exit criteria

All five Round-0 features are correct on hand-calculated examples and work on mixed
legitimate/A1 events.

---

# Iteration 5 — Baseline LightGBM detector

## Goal

Provide a robust `DetectorHead` implementation and measure the default campaign.

## Class contract

```python
class A1Detector:
    name = "a1_lightgbm"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None: ...
    def score(self, X: pd.DataFrame) -> np.ndarray: ...
```

## Suggested initial model

```python
LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
```

Treat this as a baseline, not a final tuning prescription.

## Fit behavior

1. Validate row count against `y`.
2. Select numeric columns.
3. Store the exact training-column order.
4. Replace infinities with missing values.
5. Fill missing values consistently.
6. Convert labels to a one-dimensional NumPy array.
7. Handle tiny test frames and single-class labels gracefully.

The shared contract provides a small alternating-label frame, but local unit tests should
also verify a safe fallback for single-class input. A constant-prior scorer is acceptable
when a real classifier cannot be fitted.

## Score behavior

1. Require that `fit()` has been called.
2. Add missing trained columns with zeros.
3. Ignore unexpected columns.
4. Reorder to the training schema.
5. Return positive-class probabilities.
6. Return a floating NumPy array of shape `(len(X),)`.
7. Clip defensive numerical drift into `[0, 1]`.

## Evaluation design

Use a time-ordered and campaign-grouped split:

- earliest 70% of campaigns for training;
- next 15% for validation;
- latest 15% for testing.

Do not split individual rows from one campaign across train and test.

Measure:

- precision;
- recall;
- F1;
- AUC-PR;
- ROC-AUC;
- recall at FPR `0.001`;
- per-feature AUC.

The brief's expected recall is a target, not a value to hardcode or report as measured.

## Tests

- scoring before fitting raises a clear error;
- input column order does not change scores;
- unexpected columns are ignored;
- missing trained columns are handled consistently;
- output is finite, floating, bounded, and correctly shaped;
- repeated fits with the same data are reproducible.

## Exit criteria

The detector passes both local tests and the detector section of the shared contract.

---

# Iteration 6 — Round-1 graph and durable features

## Goal

Recover detection after node-level features lose information under adapted parameters.

## Adapted scenario

Start with approximately:

```json
{
  "mule_age_days": 45,
  "split_factor": 15,
  "dwell_time_hours": 6.0
}
```

These values are an initial manual validation case. The final round parameters must come
from the Codex-reasoner-plus-CEM loop.

## Graph definition

Build a directed transfer graph:

```text
customer_id → counterparty_id
```

Edges should carry:

- event ID;
- timestamp;
- amount;
- campaign ID when present.

Use a rolling 48-hour observation horizon.

Do not rebuild the entire full-population graph once per event. Recommended options:

1. group events into 48-hour or smaller time batches;
2. maintain a sliding active-edge structure;
3. compute graph features only for relevant beneficiary/campaign neighborhoods;
4. cache repeated component calculations within a snapshot.

First implement correctness on tiny graphs, then optimize based on profiling.

## 6.1 Component size

```text
a1_component_size
```

For the beneficiary, convert the relevant local graph to undirected and count unique
nodes within shortest-path distance at most two. Document this precise interpretation of
“2-hop connected component.”

## 6.2 Component dissipation

```text
a1_component_dissipation
```

For the beneficiary's local component:

- inflow is value entering the component from outside;
- outflow is value leaving the component to outside;
- internal edges are not counted as external flow.

Compute:

```text
external outflow / max(external inflow, epsilon)
```

## 6.3 Cash-out convergence

```text
a1_cashout_convergence
```

Count distinct upstream layer-1 sources that can reach a common downstream cash-out node
within the active graph. Metadata may be used in tests to confirm ground truth, but the
production feature should be based primarily on topology rather than simply reading
`a1_cashout_node`.

## 6.4 Component age variance

```text
a1_component_age_variance
```

Calculate account-age variance across known accounts in the local component. Define and
test behavior when fewer than two ages are available.

## 6.5 Call-active signal

```text
a1_call_active
```

Safely extract the boolean from metadata and encode it as `0.0` or `1.0`. Missing values
must map to `0.0`.

This is a durable signal because the social-engineering call is part of the attack's
mechanism and is not controlled by the payment-network mutation parameters.

## 6.6 Session-duration anomaly

```text
a1_session_duration_anomaly
```

Compare current session duration with the payer's prior history. Prefer a robust measure:

```text
(current - prior median) / max(prior MAD, minimum scale)
```

Use prior events only. Fall back to `0.0` when history is insufficient.

## Graph test fixtures

Create tiny graphs with known answers:

1. one victim, one mule, one cash-out;
2. several victims converging on one mule;
3. wide layer-1 fan-out reconverging on one cash-out;
4. two completely disjoint campaigns;
5. a transfer just outside the 48-hour window;
6. cycles and repeated edges;
7. nodes with missing age information.

For each graph, hand-calculate expected component size, convergence, flow ratio, and age
variance.

## Exit criteria

All eleven canonical features are emitted together, correctly prefixed, numerically
finite, documented in `card.json`, and verified on hand-built graphs.

---

# Iteration 7 — Shared contract activation

## Goal

Turn the A1 shared test from skipped into passed.

## Steps

1. Export the three instantiated components from `__init__.py`.
2. Run local tests.
3. Run the A1 contract test.
4. Run linting on the owned folder.
5. Run the full repository suite to detect integration regressions.

Commands:

```powershell
.venv\Scripts\Activate.ps1
python -m pytest attacks/a1_upi_mule/tests/test_a1.py -v -p no:cacheprovider
python -m pytest tests/test_contract.py -k a1 -v -p no:cacheprovider
python -m ruff check attacks/a1_upi_mule
python -m pytest -q -p no:cacheprovider
```

## Contract checklist

- Generator returns every frozen schema column with the exact dtype.
- Labels are binary and fraud rows use `A1`.
- Every feature starts with `a1_`.
- Same seed produces an exactly equal DataFrame.
- Default, range, and card parameter keys match.
- Every parameter has semantics.
- Every emitted feature has a defender description.
- Detector scores are finite floating values in `[0, 1]`.
- Utility is a native float in `[0, 1]`.
- `describe_state()` returns exactly four required keys.

## Exit criteria

`tests/test_contract.py -k a1` reports one passed A1 test and no A1 skips.

---

# Iteration 8 — Signal-mortality experiment

## Goal

Measure the complete attack-adaptation-defense story using held-out campaigns.

## Dataset construction

Build a mixed event set containing:

- legitimate reference-calibrated population events;
- Round-0 A1 campaigns;
- separately seeded held-out Round-0 campaigns;
- separately seeded adapted campaigns.

Maintain fraud prevalence below 1% for final evaluation when practical. During rapid
development, a balanced subset may be used for debugging but never for the final reported
false-positive results.

## Experiment A — Default attack

1. Generate default campaigns.
2. Build core and A1 features.
3. Train the baseline detector.
4. Evaluate on held-out campaigns.
5. Record per-feature AUC and detector metrics.

## Experiment B — Adapted attack, frozen detector

1. Generate held-out campaigns with aged mules, wider splitting, and longer dwell.
2. Keep the Round-0 detector unchanged.
3. Score adapted events.
4. Measure feature mortality and recall collapse.

Expected qualitative behavior:

- beneficiary age approaches random discrimination;
- per-node fan-in loses strength;
- rapid dissipation weakens;
- original-detector recall declines materially.

Do not force the expected numeric values from the brief.

## Experiment C — Retrain without durable features

Retrain using only the original node-level features on adapted data. This establishes
whether retraining alone can restore performance. The desired story is that genuinely
dead signals remain weak even after retraining.

## Experiment D — Durable-feature recovery

Add graph and payer-behavior features, retrain, and evaluate on held-out adapted
campaigns. Record recovery and attacker utility.

## Leakage controls

- Hold out complete campaign IDs.
- Preserve temporal order.
- Fit models only on training partitions.
- Calculate metrics only on validation/test partitions.
- Do not describe delayed graph/dissipation features as authorization-time signals.
- Do not use attack metadata as a direct model label or shortcut.

## Required metrics

- recall;
- precision;
- F1;
- AUC-PR;
- recall at FPR `0.001`;
- FPR at recall `0.85` where applicable;
- per-feature AUC;
- attacker utility;
- infrastructure count and cash-out yield.

## Exit criteria

Real measured values exist for:

```text
Round 0
Round 1 before recovery
Round 1 after retraining on old signals
Round 1 after durable-feature recovery
```

---

# Iteration 9 — Documentation and pitch

## Goal

Make the A1 result reproducible and understandable to technical and non-technical judges.

## `README.md` structure

1. One-paragraph attack explanation.
2. Why payer-side authentication passes.
3. Generated transaction topology.
4. Parameter table and attacker costs.
5. Utility formula and normalization.
6. Round-0 feature definitions.
7. Round-1 durable feature definitions.
8. Real-time versus delayed feature availability.
9. Evaluation methodology and leakage controls.
10. Signal-mortality table with measured values.
11. Recall and attacker-utility progression.
12. Known limitations.
13. Reproduction commands.
14. The 60-second pitch.

## Signal-mortality table

Use measured values only:

```markdown
| Signal | R0 AUC | Adapted AUC | Verdict |
|---|---:|---:|---|
| a1_benef_account_age_days | measured | measured | DEAD/ALIVE |
| a1_benef_fanin_24h | measured | measured | DEAD/ALIVE |
| a1_dissipation_ratio | measured | measured | DEGRADED/ALIVE |
| a1_component_size | — | measured | NEW |
| a1_cashout_convergence | — | measured | NEW |
| a1_call_active | — | measured | DURABLE |
```

Do not present the brief's illustrative values as experimental results.

## Durable insight

The central conclusion should remain:

> The attacker can reshape the money's path, but cannot remove the victim's behavior
> while the coercion is happening without changing the attack itself.

## Exit criteria

A teammate can reproduce the A1 tests and explain the attack in 60 seconds using only
the package README.

---

# Iteration 10 — Person A Phase 3 responsibilities

This iteration begins only after A1 is merge-ready and the team authorizes changes to
shared files.

## Goal

Own the three-round orchestrator and the two-stage red agent:

```text
Codex reasoner → CEM numeric refinement → realism/utility budgets → next round
```

## Required build order

### Stage A — Pure CEM first

1. Complete CEM search.
2. Generate candidates inside `param_space`.
3. Evaluate detector evasion.
4. Reject candidates outside realism or utility budgets.
5. Run an end-to-end numeric loop without Codex.

### Stage B — Codex reasoner

Integrate the shared adapter using ChatGPT-authenticated `codex exec`. Do not add an API
key or OpenAI SDK dependency.

Handle:

- missing Codex CLI;
- missing authentication;
- subprocess timeout;
- non-zero exit code;
- empty output;
- malformed JSON;
- JSON Schema violation;
- unknown parameter keys;
- out-of-range values;
- budget rejection.

Every failure must fall back to CEM without breaking the loop.

### Stage C — Reasoning validation

Validate output against:

```text
redteam/proposal.schema.json
```

Clamp known numeric values to `param_space` and silently drop unknown parameters.

### Stage D — Caching

Cache the complete report, response, failures, and final accepted proposal at:

```text
results/round_N/redteam_reasoning.json
```

The judging dashboard must replay this file. It must never invoke Codex or train a model
live.

### Stage E — Three-round run

Run:

```powershell
python -m loop.run_rounds --rounds 0 1 2 --seed 42
```

Verify all required Parquet and JSON artifacts exist for every round.

## Exit criteria

- Pure CEM succeeds independently.
- Codex reasoning is schema-valid and bounded.
- Every Codex failure mode falls back safely.
- Reasoning exchanges are cached.
- Three rounds complete deterministically.
- No model training or external call occurs during dashboard replay.

---

## 4. Suggested local test organization

Organize `tests/test_a1.py` into clear sections or classes:

```text
TestCardContract
TestParameterValidation
TestGeneratorDeterminism
TestGeneratorSchema
TestTopology
TestMoneyConservation
TestUtility
TestDescribeState
TestRound0Features
TestGraphFeatures
TestDetector
TestRegressionCases
```

Prefer small hand-constructed event frames for feature tests. Full population generation
is unsuitable for ordinary unit tests.

Recommended reusable fixtures:

- `small_legit_population`;
- `default_params`;
- `small_a1_campaign`;
- `adapted_params`;
- `wide_reconverging_campaign`;
- `disjoint_campaigns`.

---

## 5. Performance expectations

The final base population contains millions of rows, so the implementation must avoid:

- `DataFrame.apply(axis=1)` for core feature calculations;
- scanning the full event table once per event;
- rebuilding a complete NetworkX graph for every transaction;
- repeated parsing of the same metadata;
- repeated loading of `card.json` inside inner loops;
- unconstrained expansion from `split_factor` at maximum parameter values.

Development order:

1. establish correctness on tiny data;
2. profile on approximately 10,000 events;
3. optimize actual bottlenecks;
4. test on approximately 100,000 events;
5. run full scale only after correctness and memory behavior are known.

Record benchmark size, elapsed time, and peak memory in the README when possible.

---

## 6. Safety and realism checks

The generator represents an authorized synthetic simulation. It must not contain tooling
for contacting victims, accessing accounts, bypassing authentication, or operating real
mule infrastructure.

Realism checks should include:

- amounts remain plausible for the victim's observed profile;
- no negative or impossible account ages;
- timestamps are causal;
- money is conserved through the graph;
- fraud remains below realistic prevalence in final evaluation;
- extreme parameter values do not create impossible transaction volumes;
- adapted evasion still achieves meaningful cash-out utility.

---

## 7. Definition of done

A1 is complete only when all items are true:

- [ ] `card.json` contains defaults, ranges, semantics, and all defender signals.
- [x] `generator.py` creates all three transfer stages.
- [x] Generation is deterministic for a fixed input and seed.
- [x] Generated rows pass the frozen schema validator.
- [x] Causal time and money-conservation tests pass.
- [x] `utility()` is finite, bounded, and penalizes infrastructure expansion.
- [x] `describe_state()` returns exactly the four required keys.
- [ ] All eleven canonical A1 features are implemented and prefixed.
- [ ] Feature calculations have hand-verified unit tests.
- [ ] `A1Detector` fits and scores through the frozen interface.
- [ ] Package exports activate the shared contract.
- [ ] `pytest tests/test_contract.py -k a1` passes rather than skips.
- [ ] Full repository tests and A1 lint checks pass.
- [ ] Round-0 metrics are measured on held-out campaigns.
- [ ] Adapted-attack recall collapse is measured.
- [ ] Old-feature-only retraining is measured.
- [ ] Durable-feature recovery is measured.
- [ ] README contains real signal-mortality numbers.
- [ ] Attacker utility and infrastructure cost are reported.
- [ ] The 60-second non-technical explanation is ready.

---

## 8. Recommended execution order

Use this order without skipping ahead:

```text
Iteration 0  Card and package configuration
Iteration 1  Minimal deterministic generator
Iteration 2  Full layered topology
Iteration 3  Utility and describe_state
Iteration 4  Round-0 features
Iteration 5  Baseline detector
Iteration 6  Graph and durable features
Iteration 7  Shared contract activation
Iteration 8  Signal-mortality experiments
Iteration 9  README and pitch
Iteration 10 Shared Phase 3 loop and red agent
```

The first implementation milestone should be a tiny deterministic campaign that passes
`core.schema.validate()`. Do not begin with NetworkX, model tuning, or Codex integration;
all of those depend on trustworthy generated transaction rows.
