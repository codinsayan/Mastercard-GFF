"""Judge-facing, replay-only Streamlit experience for A1."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .demo_data import (
    ROLE_LABEL,
    ArtifactError,
    build_campaign_graph,
    build_replay_steps,
    build_round_comparison,
    build_signal_mortality,
    campaign_events,
    format_currency,
    format_duration,
    list_a1_campaigns,
    load_external_metrics,
    load_round_artifacts,
    render_graph_svg,
)


def _styles() -> None:
    st.html(
        """
        <style>
        :root {
          --ink:#0B1220; --muted:#5B6472; --paper:#F6F7F9; --card:#FFFFFF;
          --border:#E6E8EE; --brand:#EF4B39; --brand-soft:#FDECEA;
          --good:#0F9D58; --good-soft:#E8F7EE; --bad:#D93025; --bad-soft:#FDECEA;
        }
        header[data-testid="stHeader"] { background:var(--paper) !important; box-shadow:none !important; }
        div[data-testid="stDecoration"] { display:none !important; }
        .stApp { background:var(--paper); }
        .block-container { max-width:1180px; padding-top:1.2rem; padding-bottom:3.5rem; }
        h1,h2,h3,h4 { letter-spacing:-.02em; color:var(--ink) !important; }
        p, span, div, li, label { color:var(--ink); }
        .stCaption, [data-testid="stCaptionContainer"] { color:var(--muted) !important; }

        .hero { padding:.6rem 0 1.4rem; border-bottom:1px solid var(--border); margin-bottom:1.4rem; }
        .eyebrow { color:var(--brand); font-size:.74rem; font-weight:700; letter-spacing:.12em;
                   text-transform:uppercase; margin-bottom:.6rem; }
        .hero h1 { font-size:clamp(2rem,4.4vw,3.4rem); line-height:1.05; margin:.1rem 0 .7rem; color:var(--ink); }
        .hero p { max-width:680px; color:var(--muted); font-size:1.02rem; line-height:1.5; margin-bottom:1rem; }
        .badge-row { display:flex; flex-wrap:wrap; gap:.5rem; }
        .badge { display:inline-flex; align-items:center; gap:.4rem; border:1px solid var(--border);
                 background:var(--card); color:var(--ink); border-radius:999px; padding:.32rem .72rem;
                 font-size:.78rem; font-weight:600; }
        .badge .dot { width:.44rem; height:.44rem; border-radius:999px; background:var(--brand); }

        .card { background:var(--card); border:1px solid var(--border); border-radius:16px;
                padding:1.05rem 1.2rem; }
        .section-title { font-size:1.32rem; font-weight:700; color:var(--ink); margin:0 0 .2rem; }
        .section-sub { color:var(--muted); font-size:.9rem; margin-bottom:1rem; }

        .step-chip { display:inline-block; background:var(--brand-soft); color:var(--brand);
                     font-weight:700; font-size:.74rem; letter-spacing:.04em; text-transform:uppercase;
                     padding:.28rem .6rem; border-radius:8px; margin-bottom:.5rem; }
        .event-card { background:var(--ink); color:#fff; border-radius:16px; padding:1.15rem 1.3rem;
                      min-height:172px; display:flex; flex-direction:column; justify-content:center; }
        .event-card .kicker { color:#FDA396; font-size:.72rem; font-weight:700; letter-spacing:.08em;
                               text-transform:uppercase; }
        .event-card h3 { color:#fff !important; margin:.4rem 0; font-size:1.3rem; }
        .event-card .meta { color:#D0D5DD; font-size:.86rem; line-height:1.6; }
        .event-card .meta b { color:#fff; }

        .compare-card { background:var(--card); border:1px solid var(--border); border-radius:14px;
                        padding:1rem 1.15rem; border-left:4px solid var(--good); }
        .compare-card.network { border-left-color:#D97706; }
        .compare-card .head { font-weight:700; color:var(--ink); margin-bottom:.55rem; font-size:.98rem; }
        .compare-card ul { margin:0; padding-left:1.1rem; color:var(--ink); font-size:.9rem; line-height:1.85; }

        .reasoning-card { background:var(--card); border:1px solid var(--border); border-radius:14px;
                           padding:1rem 1.2rem; }
        .reasoning-card .label { color:var(--brand); font-size:.74rem; font-weight:700; letter-spacing:.1em;
                                  text-transform:uppercase; margin-bottom:.5rem; display:flex;
                                  align-items:center; gap:.5rem; }
        .reasoning-card .body { color:var(--ink); font-size:.98rem; line-height:1.55; }
        .reasoning-card .fallback-detail { color:var(--muted); font-size:.84rem; line-height:1.6;
                                            margin-top:.55rem; }
        .reasoning-card .fallback-detail code { background:#F2F4F7; color:var(--ink); border-radius:6px;
                                                 padding:.05rem .35rem; font-size:.8rem; }
        .pill { display:inline-block; border-radius:999px; padding:.18rem .58rem; font-size:.68rem;
                font-weight:700; letter-spacing:.02em; text-transform:none; }
        .pill-amber { background:#FEF6E7; color:#B54708; }
        .pill-green { background:var(--good-soft); color:var(--good); }

        div[data-testid="stMetric"] { background:var(--card); border:1px solid var(--border);
                                      padding:.85rem 1rem; border-radius:14px; }
        div[data-testid="stMetricLabel"] { color:var(--muted) !important; }
        div[data-testid="stMetricValue"] { color:var(--ink) !important; }

        button[kind="primary"] { border-radius:10px; }
        button[kind="secondary"] { border-radius:10px; }
        [data-testid="stTabs"] button { font-weight:650; }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:var(--brand); }
        .footer-note { color:var(--muted); font-size:.82rem; }
        @media (max-width:700px) { .hero h1{font-size:2.1rem} .block-container{padding:.7rem .9rem} }
        </style>
        """
    )


@st.cache_data(show_spinner=False)
def _cached_round(root: str, round_number: int, stamp: tuple[float, ...]):
    del stamp
    return load_round_artifacts(root, round_number)


def _load_cached(root: Path, round_number: int):
    round_dir = root / f"round_{round_number}"
    stamps = tuple(
        (round_dir / name).stat().st_mtime if (round_dir / name).exists() else -1.0
        for name in ("events.parquet", "features.parquet", "scores.parquet", "metrics.json")
    )
    return _cached_round(str(root), round_number, stamps)


def _hero(external: dict[str, Any] | None) -> None:
    reference = external.get("methodology", {}).get("reference", {}) if external else {}
    external_badge = (
        "External Sparkov calibration" if reference.get("kind") == "external_reference"
        else "Development fixture"
    )
    st.html(
        f"""
        <section class="hero">
          <div class="eyebrow">Adaptive fraud arena · Attack A1</div>
          <h1>The payment is genuine.<br>The intent is not.</h1>
          <p>A deepfake caller coerces a customer into authorizing a real UPI transfer.
          Watch the money move through a synthetic mule network — and watch the defense adapt.</p>
          <div class="badge-row">
            <span class="badge"><span class="dot"></span>Authorized synthetic simulation</span>
            <span class="badge"><span class="dot"></span>UPI P2P</span>
            <span class="badge"><span class="dot"></span>{html.escape(external_badge)}</span>
          </div>
        </section>
        """
    )


def _graph_document(graph, active_edges: int) -> str:
    """Wrap the SVG graph in a self-contained HTML document.

    ``st.html`` sanitizes its input with DOMPurify and strips SVG entirely, so
    the graph must go through ``st.iframe`` instead, which embeds raw HTML
    without sanitization. Colors are hard-coded (not CSS variables) because
    the iframe is a separate document with no access to the page stylesheet.
    """
    svg = render_graph_svg(graph, active_edges)
    return (
        "<!doctype html><html><head><meta charset='utf-8'/><style>"
        "html,body{margin:0;padding:0;background:transparent;}"
        ".card{background:#FFFFFF;border:1px solid #E6E8EE;border-radius:16px;"
        "padding:14px 16px;box-sizing:border-box;}"
        "</style></head><body><div class='card'>" + svg + "</div></body></html>"
    )


def _event_card(row: pd.Series, position: int, total: int) -> None:
    risk = "Unavailable" if pd.isna(row["risk_score"]) else f"{row['risk_score']:.1%} risk"
    st.html(
        f"""
        <div class="event-card">
          <div class="kicker">Step {position + 1} of {total} · {html.escape(str(row['stage']))}</div>
          <h3>{format_currency(row['amount'])} → {html.escape(ROLE_LABEL.get(row['destination_role'], row['destination_role']))}</h3>
          <div class="meta">
            +{format_duration(row['elapsed_seconds'])} elapsed · <b>{html.escape(str(row['event_id']))}</b><br>
            {risk} · {html.escape(str(row['decision']))} · Call {'<b>active</b>' if row['call_active'] else 'not active'}
          </div>
        </div>
        """
    )


def _render_replay(artifacts, campaign_id: str) -> None:
    selected = campaign_events(artifacts.events, campaign_id)
    steps = build_replay_steps(selected, artifacts.scores)
    graph = build_campaign_graph(selected)
    state_key = f"a1_cursor_{artifacts.round_number}_{campaign_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    st.session_state[state_key] = min(st.session_state[state_key], max(len(steps) - 1, 0))
    cursor = st.session_state[state_key]

    nav_prev, nav_next, nav_reset, nav_progress = st.columns([1, 1, 1, 4])
    if nav_prev.button("← Back", disabled=cursor == 0, width="stretch"):
        st.session_state[state_key] -= 1
        st.rerun()
    if nav_next.button("Next →", type="primary", disabled=cursor >= len(steps) - 1, width="stretch"):
        st.session_state[state_key] += 1
        st.rerun()
    if nav_reset.button("Reset", width="stretch"):
        st.session_state[state_key] = 0
        st.rerun()
    with nav_progress:
        st.progress((cursor + 1) / len(steps), text=f"Step {cursor + 1} of {len(steps)}")

    graph_column, event_column = st.columns([1.9, 1], gap="large")
    with graph_column:
        st.iframe(_graph_document(graph, cursor + 1), height="content")
        if graph.sampled:
            st.caption("Representative high-value paths shown; the cached campaign is larger.")
    with event_column:
        _event_card(steps.iloc[cursor], cursor, len(steps))
        st.caption("Replay uses cached synthetic events and cached detector scores only.")

    st.markdown("#### Why authentication alone misses it")
    payer, receiver = st.columns(2, gap="medium")
    with payer:
        st.html(
            """<div class="compare-card"><div class="head">Payer side · all valid</div>
            <ul><li>Registered customer</li><li>Known device</li>
            <li>PIN / biometric authorization</li><li>Normal network context</li></ul></div>"""
        )
    with receiver:
        st.html(
            """<div class="compare-card network"><div class="head">Receiver network · coordinated</div>
            <ul><li>Newly linked beneficiary</li><li>Multi-sender fan-in</li>
            <li>Layered, timed movement</li><li>Shared cash-out convergence</li></ul></div>"""
        )


def _render_adaptation(rounds: dict[int, Any]) -> None:
    comparison = build_round_comparison({key: value.metrics for key, value in rounds.items()})
    if len(comparison) < 2:
        st.info("Precompute Round 1 to unlock the adaptation comparison.")
        return
    st.markdown('<div class="section-title">The attacker changes shape</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Read-only parameters from frozen rounds. No live optimization runs in this page.</div>',
        unsafe_allow_html=True,
    )
    columns = ["round", "mule_age_days", "split_factor", "dwell_time_hours", "n_cashout_points", "n_mules_layer1", "utility"]
    labels = {
        "round": "Round", "mule_age_days": "Mule age (days)", "split_factor": "Split factor",
        "dwell_time_hours": "Dwell (hours)", "n_cashout_points": "Cash-out points",
        "n_mules_layer1": "Layer-1 mules", "utility": "Attacker utility",
    }
    st.dataframe(comparison[columns].rename(columns=labels), hide_index=True, width="stretch")

    adapted = rounds[max(rounds)]
    proposal = adapted.reasoning.get("proposal", {})
    reasoning = proposal.get("reasoning", "No cached reasoning was recorded for this round.")
    fallback_used = bool(adapted.reasoning.get("fallback_used", False))
    failures = adapted.reasoning.get("failures", [])
    if fallback_used:
        pill = '<span class="pill pill-amber">Offline safety fallback</span>'
        headline = (
            "The judging pipeline never calls a live external reasoner — by design, this "
            "round's parameter changes came from deterministic CEM search instead."
        )
        detail = html.escape(str(failures[0])) if failures else html.escape(str(reasoning))
        body = (
            f'<div class="body">{headline}</div>'
            f'<div class="fallback-detail">This proves the adaptive loop survives a missing or '
            f"disabled reasoner without breaking. Recorded reason: <code>{detail}</code></div>"
        )
    else:
        pill = '<span class="pill pill-green">Live Codex reasoning (cached)</span>'
        body = f'<div class="body">{html.escape(str(reasoning))}</div>'
    st.html(
        f'<div class="reasoning-card"><div class="label">Cached red-agent reasoning {pill}</div>{body}</div>'
    )
    hypotheses = proposal.get("hypotheses", [])
    if hypotheses:
        with st.expander("See the tested hypotheses"):
            for item in hypotheses:
                st.markdown(
                    f"**{item.get('signal_targeted', 'Signal')}** — {item.get('action', '')}  \n"
                    f"*Cost:* {item.get('cost_to_attacker', '')}"
                )

    st.markdown("#### Signals fade as the attacker adapts")
    mortality = build_signal_mortality({key: value.metrics for key, value in rounds.items()})
    pivot = mortality.pivot(index="signal", columns="round", values="auc")
    figure = go.Figure()
    for round_number in pivot.columns:
        figure.add_trace(
            go.Bar(
                name=f"Round {round_number}", y=[name.removeprefix("a1_") for name in pivot.index],
                x=pivot[round_number], orientation="h",
                marker_color="#344054" if round_number == 0 else "#EF4B39",
            )
        )
    figure.update_layout(
        barmode="group", height=440, margin={"l": 20, "r": 30, "t": 50, "b": 30},
        xaxis={"title": "Measured AUC", "range": [0.45, 1.02], "gridcolor": "#EEF0F3"},
        yaxis={"gridcolor": "#EEF0F3"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBFBFC",
        legend={"orientation": "h", "y": 1.1, "x": 0},
        font={"color": "#0B1220"},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_evidence(external: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-title">Evidence, not theatre</div>', unsafe_allow_html=True)
    if not external:
        st.warning("External metrics artifact is unavailable; no showcase metrics are fabricated.")
        return
    reference = external["methodology"]["reference"]
    if reference.get("kind") != "external_reference":
        st.warning("This artifact is not externally calibrated.")
        return
    st.success(
        f"Externally calibrated with {reference['format']} · "
        f"{reference['source_rows']:,} source rows · SHA-256 provenance recorded"
    )
    experiments = external["experiments"]
    baseline = experiments["round_0"]
    evaded = experiments["adapted_frozen_detector"]
    recovered = experiments["adapted_durable_features"]
    one, two, three = st.columns(3)
    one.metric("Round-0 recall", f"{baseline['recall']:.0%}", help="Held-out default campaigns")
    two.metric("After adaptation", f"{evaded['recall']:.0%}", delta="Frozen detector", delta_color="inverse")
    three.metric("After durable recovery", f"{recovered['recall']:.0%}", delta="Retrained defense")

    infrastructure = external["infrastructure"]
    rows = pd.DataFrame(
        [
            {"Scenario": "Round 0", "Mules": infrastructure["round_0"]["mules"],
             "Cash-out points": infrastructure["round_0"]["cashout_points"],
             "Cash-out value": format_currency(infrastructure["round_0"]["cashout_value"]),
             "Utility": infrastructure["round_0"]["utility"]},
            {"Scenario": "Adapted", "Mules": infrastructure["adapted"]["mules"],
             "Cash-out points": infrastructure["adapted"]["cashout_points"],
             "Cash-out value": format_currency(infrastructure["adapted"]["cashout_value"]),
             "Utility": infrastructure["adapted"]["utility"]},
        ]
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.info(
        "The attacker can reshape the money path, but doing so multiplies infrastructure "
        "and reduces utility. Sparkov is an external synthetic reference—not real customer data."
    )


def render_a1_demo(results_root: str | Path = "results") -> None:
    """Render the complete replay-only A1 experience."""
    _styles()
    root = Path(results_root)
    external_path = Path(__file__).with_name("results") / "iteration8_metrics_external.json"
    try:
        external = load_external_metrics(external_path) if external_path.is_file() else None
    except ArtifactError:
        external = None
    _hero(external)

    available: dict[int, Any] = {}
    errors = []
    for round_number in (0, 1, 2):
        try:
            available[round_number] = _load_cached(root, round_number)
        except ArtifactError as exc:
            errors.append(str(exc))
    if not available:
        st.error(errors[0] if errors else "No cached rounds were found.")
        st.code("python -m loop.run_rounds --rounds 0 1 2 --seed 42")
        return

    replay_tab, adaptation_tab, evidence_tab = st.tabs(
        ["01 · Campaign replay", "02 · Adaptation", "03 · Evidence"]
    )
    with replay_tab:
        control_one, control_two = st.columns([1, 2])
        with control_one:
            round_number = st.selectbox("Cached round", sorted(available), format_func=lambda value: f"Round {value}")
        artifacts = available[round_number]
        campaigns = list_a1_campaigns(artifacts.events)
        if not campaigns:
            st.warning("This round contains no cached A1 campaigns.")
        else:
            with control_two:
                campaign_id = st.selectbox(
                    "Synthetic campaign", campaigns,
                    format_func=lambda c: c.replace("a1_camp_", "Campaign "),
                )
            _render_replay(artifacts, campaign_id)
    with adaptation_tab:
        _render_adaptation(available)
    with evidence_tab:
        _render_evidence(external)

    st.divider()
    st.html(
        '<div class="footer-note">Authorized synthetic simulation · Replay-only · No payment endpoint · '
        "No live model training · No live Codex call</div>"
    )
