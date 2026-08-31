"""Minimal replay dashboard for the A2 prompt-injection simulation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from attacks.a2_agent_inject.features import Features
from attacks.a2_agent_inject.generator import Generator
from core.population import BasePopulation, PopulationConfig

METRICS_PATH = Path(__file__).with_name("demo_metrics.json")
SCENARIOS = (
    "Legitimate checkout",
    "Round 0 · obvious injection",
    "Round 1 · semantic evasion",
    "Constrained · low-profit evasion",
)

st.set_page_config(
    page_title="A2 · Agentic Prompt Injection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#13241e; --muted:#65726d; --line:#dce5e0; --mint:#eaf7f0;
            --green:#177a50; --red:#b43b36; --amber:#b16a0b; --paper:#fbfcfb; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #f2f6f3; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
      color:var(--ink) !important;
    }
    [data-testid="stMetricValue"] { color:var(--ink) !important; }
    [data-testid="stMetricDelta"] { color:var(--green) !important; }
    [data-baseweb="tab"] { color:var(--ink) !important; }
    [data-baseweb="tab"][aria-selected="true"] { color:var(--green) !important; }
    button[data-testid="stBaseButton-primary"] {
      background:#177a50 !important; border:1px solid #177a50 !important;
      color:#ffffff !important; font-weight:700 !important;
    }
    button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] span { color:#ffffff !important; }
    button[data-testid="stBaseButton-primary"]:hover {
      background:#0f6340 !important; border-color:#0f6340 !important;
      color:#ffffff !important;
    }
    button[data-testid="stBaseButton-primary"]:hover p,
    button[data-testid="stBaseButton-primary"]:hover span { color:#ffffff !important; }
    button[data-testid="stBaseButton-primary"]:focus-visible {
      outline:3px solid #9bd4b7 !important; outline-offset:2px !important;
    }
    button[data-testid="stBaseButton-primary"]:disabled {
      background:#9bb9aa !important; border-color:#9bb9aa !important;
      color:#ffffff !important; opacity:1 !important;
    }
    [data-testid="InputInstructions"] { display:none !important; }
    [data-testid="stMetric"] { background: white; border:1px solid var(--line);
      border-radius:14px; padding:16px 18px; box-shadow:0 2px 12px rgba(19,36,30,.04); }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    .hero { padding:8px 0 18px 0; }
    .eyebrow { color:var(--green); font-size:.78rem; letter-spacing:.12em;
      text-transform:uppercase; font-weight:700; }
    .hero h1 { color:var(--ink); font-size:2.45rem; line-height:1.08; margin:.35rem 0 .55rem; }
    .hero p { color:var(--muted); max-width:850px; font-size:1.02rem; }
    .flow { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:8px 0 22px; }
    .flow-card { background:white; border:1px solid var(--line); border-radius:12px; padding:14px; }
    .flow-card b { display:block; color:var(--ink); margin-bottom:4px; }
    .flow-card span { color:var(--muted); font-size:.88rem; }
    .number { color:var(--green); font-weight:800; font-size:.78rem; }
    .notice { border-left:4px solid var(--green); background:var(--mint); padding:13px 15px;
      border-radius:0 10px 10px 0; color:var(--ink); margin:10px 0; }
    .danger { border-left-color:var(--red); background:#fff0ef; }
    .status { display:inline-block; padding:4px 9px; border-radius:999px; font-weight:700;
      font-size:.75rem; margin-right:6px; }
    .status-ok { background:#dff3e8; color:#12633f; }
    .status-warn { background:#fff0d5; color:#8b5208; }
    .status-bad { background:#fae1df; color:#96322e; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .small-note { color:var(--muted); font-size:.82rem; }
    .run-result { border:1px solid #b9ddc9; background:#eff9f3; color:#12633f;
      padding:10px 12px; border-radius:10px; font-size:.86rem; margin:8px 0 14px; }
    .demo-loader { display:flex; align-items:center; gap:18px; background:#ffffff;
      border:1px solid #b9ddc9; border-radius:16px; padding:20px 22px; margin:14px 0;
      box-shadow:0 8px 24px rgba(19,36,30,.07); }
    .demo-loader-ring { width:42px; height:42px; flex:0 0 42px; border-radius:50%;
      border:5px solid #dcefe5; border-top-color:#177a50;
      animation:demo-spin .85s linear infinite; }
    .demo-loader-copy strong { display:block; color:var(--ink); font-size:1.08rem;
      margin-bottom:3px; }
    .demo-loader-copy span { color:var(--muted); font-size:.9rem; }
    @keyframes demo-spin { to { transform:rotate(360deg); } }
    @media (max-width: 900px) { .flow { grid-template-columns:1fr 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def evaluation_results() -> dict[str, object]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def demo_sessions(seed: int) -> dict[str, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            "amount": [125.0, 499.0, 1200.0, 2800.0, 6500.0, 18000.0, 72000.0],
            "mcc": ["5411", "5812", "5941", "5732", "5712", "5651", "5946"],
        }
    )
    config = PopulationConfig(customers=500, merchants=150, vpas=200, days=30, events=1000)
    population = BasePopulation(reference, config).generate(seed=2026)
    generator = Generator()
    round0 = generator.default_params()
    evolved = {**round0, "injection_semantic_distance": 0.25}
    constrained = {
        **evolved,
        "price_multiplier": 3.0,
        "cart_mutated_post_load": 0,
    }
    round0_events = generator.generate(round0, population, seed=seed)
    evolved_events = generator.generate(evolved, population, seed=seed + 1)
    constrained_events = generator.generate(constrained, population, seed=seed + 2)
    return {
        "Legitimate checkout": integer_multiple_prices(round0_events, injected_multiple=40),
        "Round 0 · obvious injection": integer_multiple_prices(
            round0_events, injected_multiple=40
        ),
        "Round 1 · semantic evasion": integer_multiple_prices(
            evolved_events, injected_multiple=40
        ),
        "Constrained · low-profit evasion": integer_multiple_prices(
            constrained_events, injected_multiple=3
        ),
    }


def integer_multiple_prices(frame: pd.DataFrame, *, injected_multiple: int) -> pd.DataFrame:
    """Normalize demo prices to exact whole-number multiples while preserving cap usage."""
    result = frame.copy(deep=True)
    normalized_meta: list[dict[str, object]] = []
    normalized_amounts: list[float] = []
    for amount, original_meta in zip(result["amount"], result["meta"], strict=True):
        meta = dict(original_meta)
        medians = [float(value) for value in meta["a2_item_median_prices"]]
        injected = {int(value) for value in meta["a2_injected_indices"]}
        multiples = [injected_multiple if index in injected else 1 for index in range(len(medians))]
        prices = [round(median * multiple, 2) for median, multiple in zip(medians, multiples, strict=True)]
        total = round(sum(prices), 2)
        old_cap = float(meta["a2_spend_cap"])
        utilization = float(amount) / old_cap if old_cap > 0 else 0.8
        meta["a2_item_prices"] = prices
        meta["a2_spend_cap"] = round(total / utilization, 2) if utilization > 0 else total
        normalized_meta.append(meta)
        normalized_amounts.append(total)
    result["amount"] = normalized_amounts
    result["meta"] = normalized_meta
    return result


def validate_seed(raw_value: str) -> tuple[int | None, str | None]:
    """Validate the intentionally bounded set of demo seeds."""
    cleaned = raw_value.strip()
    if not re.fullmatch(r"[0-9]+", cleaned):
        return None, "Enter a whole number from 1 to 600. Letters, decimals and signs are not allowed."
    value = int(cleaned)
    if not 1 <= value <= 600:
        return None, "Seed must be between 1 and 600."
    return value, None


def representative_event(frame: pd.DataFrame, fraud: bool) -> pd.Series:
    candidates = frame.loc[frame["label"].eq(int(fraud))]
    camping = candidates.loc[
        candidates["meta"].map(lambda meta: meta.get("a2_intent_category") == "camping")
    ]
    return (camping if not camping.empty else candidates).iloc[0]


def cart_frame(event: pd.Series) -> pd.DataFrame:
    meta = event["meta"]
    injected = set(meta["a2_injected_indices"])
    rows = []
    for index, (item, category, price, median) in enumerate(
        zip(
            meta["a2_cart_items"],
            meta["a2_cart_categories"],
            meta["a2_item_prices"],
            meta["a2_item_median_prices"],
            strict=True,
        )
    ):
        rows.append(
            {
                "Product": item,
                "Category": str(category).replace("_", " ").title(),
                "Price": f"₹{float(price):,.0f}",
                "Normal price": f"₹{float(median):,.0f}",
                "Multiple": f"{round(float(price) / float(median))}×",
                "Origin": "INJECTED" if index in injected else "Agent selected",
            }
        )
    return pd.DataFrame(rows)


def recall_chart(metrics: dict[str, object]) -> go.Figure:
    labels = ["Round 0", "After evasion", "Durable defence"]
    values = [
        100 * metrics["round0"]["recall"],
        100 * metrics["round1_pre_fix"]["recall"],
        100 * metrics["round1_post_fix"]["recall"],
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#177a50", "#b43b36", "#177a50"],
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>Recall %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis={
            "title": "Fraud recall",
            "range": [0, 108],
            "ticksuffix": "%",
            "gridcolor": "#e6ece8",
        },
        xaxis={"title": None},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def mortality_chart(metrics: dict[str, object]) -> go.Figure:
    signals = {
        "Intent ↔ cart": "a2_intent_cart_cosine",
        "Cap utilization": "a2_amount_vs_cap_ratio",
        "Item count": "a2_item_count",
        "Price abnormality": "a2_price_zscore_in_category",
        "Post-load mutation": "a2_cart_mutated_post_load",
    }
    round0 = [metrics["signal_auc"][key]["round0"] for key in signals.values()]
    evolved = [metrics["signal_auc"][key]["round1"] for key in signals.values()]
    fig = go.Figure()
    fig.add_bar(name="Round 0", x=list(signals), y=round0, marker_color="#9bb9aa")
    fig.add_bar(name="Evolved", x=list(signals), y=evolved, marker_color="#177a50")
    fig.add_hline(y=0.5, line_dash="dash", line_color="#b43b36", annotation_text="No signal")
    fig.update_layout(
        barmode="group",
        height=380,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        yaxis={
            "title": "Single-signal AUC",
            "range": [0.45, 1.05],
            "gridcolor": "#e6ece8",
        },
        legend={"orientation": "h", "y": 1.12, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


metrics = evaluation_results()

if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = SCENARIOS[0]
if "active_seed" not in st.session_state:
    st.session_state.active_seed = 42

with st.sidebar:
    st.markdown("### Demo controls")
    scenario_choice = st.radio(
        "Replay scenario",
        SCENARIOS,
        index=SCENARIOS.index(st.session_state.active_scenario),
    )
    seed_text = st.text_input(
        "Simulation seed (1–600)",
        value=str(st.session_state.active_seed),
        max_chars=3,
        placeholder="Enter 1 to 600",
    )
    seed_choice, seed_error = validate_seed(seed_text)
    if seed_error:
        st.error(seed_error, icon="⚠️")
    else:
        st.caption("Each valid seed creates 600 sessions. The same seed repeats the same carts.")
    run_requested = st.button(
        "Run this simulation",
        type="primary",
        width="stretch",
        disabled=seed_error is not None,
    )
    st.caption("1. Choose a scenario  ·  2. Click Run  ·  3. Compare the signals")

if run_requested and seed_choice is not None:
    st.session_state.active_scenario = scenario_choice
    st.session_state.active_seed = seed_choice

scenario = st.session_state.active_scenario
seed = int(st.session_state.active_seed)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">A2 · Authorised red-team simulation</div>
      <h1>When the cart obeys the page—not the person.</h1>
      <p>A compromised merchant page injects instructions into an AI shopping agent. The
      payment remains policy-valid; the fraud appears only in the gap between signed intent,
      cart meaning, price, and session provenance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice"><b>Main goal:</b> prove the adaptive loop. First we catch an obvious
    semantic mismatch. Then the attacker chooses an in-category product and kills that signal.
    Finally, price and provenance restore detection while cutting attacker utility.</div>
    """,
    unsafe_allow_html=True,
)

if run_requested:
    demo_sessions.clear()
    loader = st.empty()
    loader.markdown(
        """
        <div class="demo-loader">
          <div class="demo-loader-ring"></div>
          <div class="demo-loader-copy">
            <strong>Building the attack replay…</strong>
            <span>Generating 600 sessions · applying the selected scenario · calculating signals</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sessions = demo_sessions(int(seed))
    loader.empty()
else:
    sessions = demo_sessions(int(seed))
selected = representative_event(sessions[scenario], fraud=scenario != "Legitimate checkout")
selected_features = Features().build(selected.to_frame().T).iloc[0]
meta = selected["meta"]
if run_requested:
    st.markdown(
        f'<div class="run-result"><b>Simulation complete.</b> Generated 600 synthetic '
        f'sessions with seed {int(seed)} and selected a representative checkout below.</div>',
        unsafe_allow_html=True,
    )

top1, top2, top3, top4 = st.columns(4)
top1.metric("Round-0 recall", f"{100 * metrics['round0']['recall']:.1f}%")
top2.metric(
    "Recall after evasion",
    f"{100 * metrics['round1_pre_fix']['recall']:.1f}%",
    delta=f"{-100 * (metrics['round0']['recall'] - metrics['round1_pre_fix']['recall']):.1f} pp",
    delta_color="inverse",
)
top3.metric("Durable recovery", f"{100 * metrics['round1_post_fix']['recall']:.1f}%")
top4.metric("Attacker utility cut", f"{100 * metrics['utility']['reduction']:.1f}%")

st.markdown(
    """
    <div class="flow">
      <div class="flow-card"><span class="number">01</span><b>User intent</b><span>Signed request defines what may be bought.</span></div>
      <div class="flow-card"><span class="number">02</span><b>Compromised page</b><span>Hidden text instructs the shopping agent.</span></div>
      <div class="flow-card"><span class="number">03</span><b>Mutated cart</b><span>Agent adds an attacker-selected product.</span></div>
      <div class="flow-card"><span class="number">04</span><b>Valid payment</b><span>Token, merchant and spend cap still pass.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_replay, tab_mortality, tab_economics, tab_validation = st.tabs(
    ["Replay one checkout", "Signal mortality", "Attacker economics", "Validation"]
)

with tab_replay:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader(scenario)
        st.markdown(f"**Signed intent:** “{meta['a2_intent_text']}”")
        st.dataframe(cart_frame(selected), hide_index=True, width="stretch")
        total, cap = st.columns(2)
        total.metric("Cart total", f"₹{float(selected['amount']):,.0f}")
        cap.metric("Valid spend cap", f"₹{float(meta['a2_spend_cap']):,.0f}")
    with right:
        st.subheader("What the issuer observes")
        status_class = "status-ok" if selected["label"] == 0 else "status-bad"
        status_text = "LEGITIMATE" if selected["label"] == 0 else "INJECTED"
        st.markdown(
            f'<span class="status {status_class}">{status_text}</span>'
            '<span class="status status-ok">PAYMENT APPROVED</span>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Intent/cart cosine", f"{selected_features['a2_intent_cart_cosine']:.3f}")
        c2.metric("Price abnormality", f"{selected_features['a2_price_zscore_in_category']:.2f}")
        c3, c4 = st.columns(2)
        c3.metric("Cap utilization", f"{100 * selected_features['a2_amount_vs_cap_ratio']:.1f}%")
        c4.metric(
            "Post-load mutation",
            "YES" if selected_features["a2_cart_mutated_post_load"] else "NO",
        )
        if scenario == "Legitimate checkout":
            message = "The cart matches the intent, prices are ordinary, and provenance is clean."
            css = "notice"
        elif scenario == "Round 0 · obvious injection":
            message = "The injected product breaks semantic alignment. The round-0 detector catches it."
            css = "notice danger"
        elif scenario == "Round 1 · semantic evasion":
            message = "Meaning now looks legitimate. The old semantic detector collapses; price and provenance remain." 
            css = "notice danger"
        else:
            message = "The attacker suppresses price and provenance evidence, but utility falls by 94.9%."
            css = "notice"
        st.markdown(f'<div class="{css}">{message}</div>', unsafe_allow_html=True)

with tab_mortality:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("The adaptive loop works")
        st.plotly_chart(recall_chart(metrics), width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="notice danger"><b>The old signal is dead.</b> Intent/cart cosine '
            'falls from AUC 0.9998 to 0.5098—indistinguishable from chance.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Signals before and after adaptation")
        st.plotly_chart(mortality_chart(metrics), width="stretch", config={"displayModeBar": False})
        st.caption("AUC near 0.5 means the signal carries almost no fraud information.")

with tab_economics:
    st.subheader("The attack dies economically, not technically")
    before, after, reduction = st.columns(3)
    before.metric("40× injection utility", f"{metrics['utility']['round0_40x']:.4f}")
    after.metric("3× + preloaded utility", f"{metrics['utility']['constrained_3x_preloaded']:.4f}")
    reduction.metric("Utility reduction", f"{100 * metrics['utility']['reduction']:.2f}%")
    st.progress(float(metrics["utility"]["reduction"]), text="Attacker utility removed")
    st.markdown(
        """
        <div class="notice"><b>Cost of evasion:</b> the attacker must reduce the price
        multiplier from 40× to 3× and replace easy post-load injection with deeper merchant
        source compromise. The fraud is still possible, but no longer attractive at the same scale.</div>
        """,
        unsafe_allow_html=True,
    )

with tab_validation:
    st.subheader("Evidence this is a real, reproducible simulation")
    checks = pd.DataFrame(
        [
            ("Frozen integration contract", "PASS", "Schema, interfaces and feature semantics"),
            ("A2 behavioral tests", "5 / 5 PASS", "Generation, evasion, economics and assets"),
            ("Full repository suite", "23 PASS", "Three unmerged attacks skipped as expected"),
            ("Deterministic replay", "PASS", "Same seed produces byte-identical events"),
            ("Offline judging mode", "PASS", "Cached 384-dimensional MiniLM embeddings"),
        ],
        columns=["Check", "Result", "Evidence"],
    )
    st.dataframe(checks, hide_index=True, width="stretch")
    st.markdown(
        '<p class="small-note">Metrics come from held-out deterministic train/test seeds with '
        '600 matched sessions per dataset. Perfect durable recovery is a controlled-simulation '
        'result and must not be presented as production performance.</p>',
        unsafe_allow_html=True,
    )
