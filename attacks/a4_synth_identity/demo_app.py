"""Replay-only judge demo for A4.

Reads precomputed artifacts under ``results/a4/`` and renders them. It never trains a
model, never generates events, never calls Codex, and needs no network -- the team brief
requires the judging demo to replay and nothing else. ``manual_app.py`` is the interactive
developer lab and is not for judging.

    streamlit run attacks/a4_synth_identity/demo_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "a4"
FRAUD = "#d1495b"
LEGIT = "#3f8f9b"
MUTED = "#8c94a5"

SIGNAL_VERDICTS = {
    "a4_shared_address_token_count": "static - dead by round 1",
    "a4_shared_device_prefix_count": "static - dead by round 1",
    "a4_phone_block_density": "static - dead by round 1",
    "a4_account_age_at_first_large_txn": "behavioral",
    "a4_trajectory_cluster_density": "behavioral - replacement",
    "a4_trajectory_dist_to_centroid": "behavioral - replacement",
    "a4_trajectory_cluster_size": "behavioral - replacement",
    "a4_ramp_convexity": "behavioral - durable",
    "a4_payment_ratio_regularity": "behavioral - attacked in round 2",
    "a4_bustout_sync_score": "behavioral - durable, see caveats",
}


@st.cache_data(show_spinner=False)
def load() -> dict[str, Any]:
    summary_path = RESULTS / "a4_summary.json"
    assets_path = RESULTS / "demo_assets.json"
    if not summary_path.exists() or not assets_path.exists():
        return {}
    reasoning = {}
    for name in ("1", "2"):
        path = RESULTS / f"round_{name}" / "redteam_reasoning.json"
        if path.exists():
            reasoning[name] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "assets": json.loads(assets_path.read_text(encoding="utf-8")),
        "reasoning": reasoning,
    }


def presenter_note(text: str) -> None:
    st.caption(f":grey[Say this: {text}]")


def headline(text: str, sub: str = "") -> None:
    st.markdown(f"## {text}")
    if sub:
        st.markdown(f"<p style='color:{MUTED};font-size:1.05rem'>{sub}</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------- screen 1
def screen_hook(data: dict[str, Any]) -> None:
    hook = data["assets"]["hook"]
    headline(
        "This transaction is fraud.",
        "Nothing about it is anomalous. That is the entire problem.",
    )

    rows = pd.DataFrame(hook["rows"])
    build_rows = rows[rows["phase"].eq("BUILD")]
    first = build_rows.iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
<div style='border:2px solid {FRAUD};border-radius:10px;padding:1.1rem 1.3rem'>
  <div style='color:{MUTED};font-size:.8rem;letter-spacing:.09em'>DAY {int(first["day"])}
  OF THIS ACCOUNT'S LIFE</div>
  <div style='font-size:2.5rem;font-weight:700;line-height:1.2'>{first["amount"]:,.2f}</div>
  <div style='font-size:1rem'>{first["utilization"]:.0%} of the credit limit used &nbsp;·&nbsp;
  statement paid {first["payment_ratio"]:.0%} on time</div>
  <div style='margin-top:.7rem;color:{FRAUD};font-weight:700;font-size:1.1rem'>
  label = 1 &nbsp;(fraud)</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            "A perfectly ordinary purchase, paid off in full. A per-transaction model cannot "
            "flag this without flagging every honest customer alongside it. The fraud is not in "
            "the row -- it is in the **shape of the account over time**, and in the fact that "
            "hundreds of unrelated-looking customers are tracing that same shape together."
        )

    with right:
        arc = pd.DataFrame(hook["trajectory"])
        base = alt.Chart(arc)
        line = base.mark_line(color=FRAUD, strokeWidth=2.5).encode(
            x=alt.X("day:Q", title="Day since onboarding"),
            y=alt.Y("utilization:Q", title="Credit utilization", axis=alt.Axis(format="%")),
        )
        points = base.mark_circle(color=FRAUD, size=45).encode(
            x="day:Q",
            y="utilization:Q",
            tooltip=["day:Q", alt.Tooltip("utilization:Q", format=".1%"), "amount:Q", "phase:N"],
        )
        marks = alt.Chart(rows).mark_point(
            shape="triangle-down", size=140, color="#1f2937", filled=True
        ).encode(x="day:Q", y=alt.Y("utilization:Q"))
        st.altair_chart(
            (line + points + marks).properties(height=330, title="The same account, 120 days"),
            width="stretch",
        )

    st.markdown("##### Four rows from one synthetic identity")
    display = rows.assign(
        utilization=rows["utilization"].map("{:.0%}".format),
        payment_ratio=rows["payment_ratio"].map("{:.0%}".format),
        amount=rows["amount"].map("{:,.2f}".format),
    )[["event_id", "day", "phase", "utilization", "payment_ratio", "amount", "label"]]
    st.dataframe(display, hide_index=True, width="stretch")
    presenter_note(
        "Every one of these rows is labelled fraud. The first three are indistinguishable from "
        "a good customer. Only the last one looks wrong, and by then the money is gone."
    )


# ---------------------------------------------------------------- screen 2
def screen_cohort(data: dict[str, Any]) -> None:
    cohort = data["assets"]["cohort"]
    round0 = data["assets"]["rounds"]["0"]
    headline(
        "A thousand strangers, moving in step.",
        "Different names, different addresses, different devices - one operating playbook.",
    )

    def frame(entries: list[dict[str, Any]], kind: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"customer_id": e["customer_id"], "day": d, "utilization": u, "kind": kind}
                for e in entries
                for d, u in e["points"]
            ]
        )

    synthetic = frame(cohort["synthetic"], "Synthetic identities (one cluster)")
    organic = frame(cohort["organic"], "Organic customers")

    left, right = st.columns(2)
    for column, source, color in ((left, synthetic, FRAUD), (right, organic, LEGIT)):
        with column:
            chart = (
                alt.Chart(source)
                .mark_line(opacity=0.35, strokeWidth=1.4, color=color)
                .encode(
                    x=alt.X("day:Q", title="Day"),
                    y=alt.Y(
                        "utilization:Q",
                        title="Utilization",
                        axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    detail="customer_id:N",
                )
                .properties(height=290, title=source["kind"].iloc[0])
            )
            st.altair_chart(chart, width="stretch")

    st.markdown("##### When each account first crosses 85% utilization")
    spikes = pd.DataFrame(
        [{"day": d, "kind": "Synthetic"} for d in round0["spike_days"]["synthetic"]]
        + [{"day": d, "kind": "Organic"} for d in round0["spike_days"]["organic"]]
    )
    histogram = (
        alt.Chart(spikes)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("day:Q", bin=alt.Bin(maxbins=45), title="Day of first draw-down"),
            y=alt.Y("count()", title="Accounts", stack=None),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["Synthetic", "Organic"], range=[FRAUD, LEGIT]),
                legend=alt.Legend(title=None, orient="top-right"),
            ),
        )
        .properties(height=230)
    )
    st.altair_chart(histogram, width="stretch")
    presenter_note(
        "Organic customers max out whenever life happens to them - spread flat across the year. "
        "The synthetic cluster spikes together inside a few days, because one operator is "
        "running all of them."
    )


# ---------------------------------------------------------------- screen 3
def screen_mortality(data: dict[str, Any]) -> None:
    summary = data["summary"]
    headline(
        "Signal mortality.",
        "What the attacker killed, and what replaced it. 0.50 AUC is a coin flip.",
    )

    records = []
    for round_name in ("0", "1", "2"):
        for signal, auc in summary[round_name]["signal_auc"].items():
            records.append(
                {
                    "signal": signal,
                    "round": f"Round {round_name}",
                    "auc": float(auc),
                    "verdict": SIGNAL_VERDICTS.get(signal, ""),
                }
            )
    frame = pd.DataFrame(records)
    order = list(summary["0"]["signal_auc"])

    heat = (
        alt.Chart(frame)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("round:N", title=None, axis=alt.Axis(orient="top", labelFontSize=13)),
            y=alt.Y("signal:N", title=None, sort=order, axis=alt.Axis(labelFontSize=12)),
            color=alt.Color(
                "auc:Q",
                scale=alt.Scale(domain=[0.5, 0.75, 1.0], range=["#c9ccd4", "#7fb2bd", "#14616e"]),
                legend=alt.Legend(title="AUC", gradientLength=180),
            ),
            tooltip=["signal:N", "round:N", alt.Tooltip("auc:Q", format=".3f"), "verdict:N"],
        )
    )
    labels = heat.mark_text(fontSize=13, fontWeight="bold").encode(
        text=alt.Text("auc:Q", format=".3f"),
        color=alt.condition(alt.datum.auc > 0.8, alt.value("white"), alt.value("#1f2937")),
    )
    st.altair_chart((heat + labels).properties(height=380), width="stretch")

    st.markdown(
        f"""
<div style='display:flex;gap:1.2rem;margin-top:.4rem'>
  <div style='flex:1;border-left:4px solid {FRAUD};padding-left:.9rem'>
    <b>The three static signals died.</b><br>
    Shared addresses, device prefixes and phone blocks all fell from ~0.85 to ~0.50 the moment
    the attacker diversified identity attributes. That evasion cost it nothing.
  </div>
  <div style='flex:1;border-left:4px solid {LEGIT};padding-left:.9rem'>
    <b>The behavioral signals replaced them.</b><br>
    Trajectory clustering strengthens across rounds as the campaign is forced to spread out.
    Ramp convexity holds at ~0.99 throughout.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    presenter_note(
        "Read the top three rows left to right. That is our entire round-zero detector dying, "
        "and it died because diversifying identities is free for a generative model."
    )


# ---------------------------------------------------------------- screen 4
def screen_reasoning(data: dict[str, Any]) -> None:
    summary = data["summary"]
    headline(
        "The attacker, in its own words.",
        "Codex reads our detector report each round and writes its own evasion plan.",
    )

    round_name = st.radio(
        "Round", ["1", "2"], horizontal=True, format_func=lambda value: f"Round {value}"
    )
    entry = summary[round_name]
    proposal = entry.get("proposal", {})
    cached = data["reasoning"].get(round_name, {})
    runner = cached.get("runner", {})

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Its reasoning")
        st.info(proposal.get("reasoning", "unavailable"))
        for hypothesis in proposal.get("hypotheses", [])[:4]:
            st.markdown(
                f"- **{hypothesis.get('signal_targeted','')}** — {hypothesis.get('action','')}  \n"
                f"  <span style='color:{MUTED}'>cost to attacker: "
                f"{hypothesis.get('cost_to_attacker','')}</span>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("##### What it cost us")
        if "pre_fix" in entry:
            st.metric(
                "AUC-PR when the plan landed",
                f"{entry['pre_fix']['auc_pr']:.3f}",
                delta=f"{entry['pre_fix']['auc_pr'] - 1.0:+.3f} vs a perfect detector",
                delta_color="inverse",
            )
        st.metric("AUC-PR after we rebuilt", f"{entry['post_fix']['auc_pr']:.3f}")

        st.markdown("##### What it cost the attacker")
        previous = summary[str(int(round_name) - 1)]["utility"]
        current = entry["utility"]
        st.metric(
            "Attacker utility",
            f"{current:.3f}",
            delta=f"{current - previous:+.3f} vs round {int(round_name) - 1}",
            delta_color="inverse" if current < previous else "normal",
        )
        if current >= previous - 0.01:
            st.success("This evasion was **free**. That is why those signals were fragile.")
        else:
            st.warning(
                f"This evasion **cost {(previous - current) / previous:.0%} of the payoff**. "
                "Strategy is expensive to randomise."
            )

    changed = {
        name: value
        for name, value in entry["params"].items()
        if summary[str(int(round_name) - 1)]["params"].get(name) != value
    }
    if changed:
        st.markdown("##### Parameters it moved")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "parameter": name,
                        f"round {int(round_name) - 1}": summary[str(int(round_name) - 1)][
                            "params"
                        ].get(name),
                        f"round {round_name}": round(value, 4)
                        if isinstance(value, float)
                        else value,
                    }
                    for name, value in changed.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    if runner:
        st.caption(
            f":grey[Generated offline by {runner.get('runner','?')} "
            f"{runner.get('cli_version','?')}, model {runner.get('model','?')}, "
            f"fallback_used={cached.get('fallback_used')}. Cached to "
            f"results/a4/round_{round_name}/redteam_reasoning.json and replayed here.]"
        )
    presenter_note(
        "This is not us describing the attacker. This is the attacker describing itself, "
        "offline, before we measured the damage on the previous screen."
    )


# ---------------------------------------------------------------- screen 5
def screen_scoreboard(data: dict[str, Any]) -> None:
    summary = data["summary"]
    assets = data["assets"]
    headline("Scoreboard, and what we do not claim.", "")

    stages = [
        ("Round 0 · static detector", summary["0"]["post_fix"]),
        ("Round 1 · static, after evasion", summary["1"]["pre_fix"]),
        ("Round 1 · behavioral rebuild", summary["1"]["post_fix"]),
        ("Round 2 · behavioral, after evasion", summary["2"]["pre_fix"]),
        ("Round 2 · retrained", summary["2"]["post_fix"]),
    ]
    table = pd.DataFrame(
        [
            {
                "stage": label,
                "AUC-PR": round(metrics["auc_pr"], 3),
                "recall @ FPR 0.001": round(metrics["recall_at_fpr_0.001"], 3),
                "FPR @ 85% recall": round(metrics["fpr_at_recall_0.85"], 3),
                "recall @ 0.5": round(metrics["recall"], 3),
            }
            for label, metrics in stages
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    histograms = []
    for name in ("0", "1", "2"):
        entry = assets["rounds"][name]["score_histogram"]
        for left, fraud_count, legit_count in zip(
            entry["bin_left"], entry["fraud"], entry["legit"], strict=True
        ):
            histograms.append(
                {"round": f"Round {name}", "score": left, "n": fraud_count, "kind": "Synthetic"}
            )
            histograms.append(
                {"round": f"Round {name}", "score": left, "n": legit_count, "kind": "Organic"}
            )
    chart = (
        alt.Chart(pd.DataFrame(histograms))
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("score:Q", title="Detector score", bin="binned", scale=alt.Scale(domain=[0, 1])),
            x2="score2:Q",
            y=alt.Y("n:Q", title="Accounts", stack=None, scale=alt.Scale(type="symlog")),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["Synthetic", "Organic"], range=[FRAUD, LEGIT]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            column=alt.Column("round:N", title=None),
        )
        .transform_calculate(score2="datum.score + 0.05")
        .properties(height=210, width=210)
    )
    st.altair_chart(chart)

    st.markdown("##### What we do not claim")
    reference = assets.get("reference", {})
    st.warning(
        f"""
**A fixed 0.5 threshold flatters the wrong model.** At round 2 the pre-evasion detector scores
recall 1.000 at 0.5 but **0.000** at FPR 0.001 — it is uncalibrated and flags nearly everyone.
Read AUC-PR and FPR-at-recall instead.

**`a4_bustout_sync_score` is still suspiciously strong (0.978–1.000).** Organic accounts can now
genuinely cross the 85% line — 13.3% of them do — but synthetic clusters still spike far more
tightly than any organic cohort. We treat it as unproven, not as a result.

**Fraud prevalence here is far above reality.** Roughly a third of accounts in the evaluation set
are synthetic against ~0.6% in the real world, so precision is optimistic.

**Our reference data is a public benchmark, not real transactions.** Marginals are fitted to
Sparkov ({reference.get("rows", 0):,} rows, {reference.get("distinct_mcc", 0)} categories), which
is itself simulated. IEEE-CIS would be the stronger claim.
"""
    )
    presenter_note(
        "We would rather tell you these four things than have you find them. Every one is in "
        "the repository README as well."
    )


SCREENS = {
    "1 · The innocent transaction": screen_hook,
    "2 · Moving in step": screen_cohort,
    "3 · Signal mortality": screen_mortality,
    "4 · The attacker's words": screen_reasoning,
    "5 · Scoreboard & caveats": screen_scoreboard,
}


def main() -> None:
    st.set_page_config(page_title="A4 · Synthetic Identity Bust-Out", layout="wide")
    data = load()
    if not data:
        st.error(
            "Precomputed results are missing. Run:\n\n"
            "```\npython -m attacks.a4_synth_identity.experiment --codex --seed 42\n"
            "python -m attacks.a4_synth_identity.build_demo_assets\n```"
        )
        return

    st.sidebar.markdown("### A4 · Synthetic identity bust-out")
    st.sidebar.caption(
        "GenAI opens accounts for people who do not exist, behaves perfectly for months, "
        "then draws every limit down at once."
    )
    choice = st.sidebar.radio("Screen", list(SCREENS), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(
        ":grey[Replay only. No model is trained, no data generated and no network used while "
        "this is open — every number is read from results/a4/.]"
    )
    SCREENS[choice](data)


if __name__ == "__main__":
    main()
