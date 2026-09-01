"""Interactive A4 smoke test: generate, inspect trajectories, and score snapshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit executes this file as a script, so make the repository packages importable
# regardless of the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.a4_synth_identity.detector import A4Detector
from attacks.a4_synth_identity.experiment import make_round_data
from attacks.a4_synth_identity.features import FEATURE_NAMES
from attacks.a4_synth_identity.generator import A4Generator


@st.cache_data(show_spinner="Generating the 120-day campaign…")
def simulate(params_json: str, seed: int):
    params = json.loads(params_json)
    generator = A4Generator()
    data = make_round_data(params, seed, 0, trajectory_epochs=4)
    detector = A4Detector()
    detector.fit(data.accounts[list(FEATURE_NAMES)], data.accounts["label"].to_numpy())
    accounts = data.accounts.assign(a4_score=detector.score(data.accounts[list(FEATURE_NAMES)]))
    fraud = data.events[data.events["attack_id"].eq("A4")]
    return fraud, accounts, generator.utility(fraud)


def measured_results() -> dict | None:
    path = PROJECT_ROOT / "results" / "a4" / "a4_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    st.set_page_config(page_title="A4 Synthetic Identity Lab", layout="wide")
    st.error(
        "**Developer lab - not the judging demo.** This page generates data and fits a "
        "detector live, in-sample, on every interaction: its scores are for eyeballing "
        "generator changes and are not evaluation metrics. For judges run "
        "`streamlit run attacks/a4_synth_identity/demo_app.py`, which replays precomputed "
        "results and trains nothing.",
        icon=":material/science:",
    )
    st.title("A4 · Synthetic Identity Bust-Out Lab")
    st.caption(
        "Interactive smoke test using campaign-snapshot features. Scores are fitted in-sample "
        "for inspection and are not reported evaluation metrics."
    )

    defaults = A4Generator().default_params()
    with st.sidebar.form("parameters"):
        st.subheader("Campaign parameters")
        preset = st.radio("Preset", ["Round 0: shared attributes", "Round 1: diversified"])
        params = {
            "n_clusters": st.slider("Identity clusters", 1, 6, 3),
            "identities_per_cluster": st.slider("Identities per cluster", 10, 40, 15),
            "attribute_overlap_rate": st.slider(
                "Attribute overlap",
                0.0,
                0.9,
                0.7 if preset.startswith("Round 0") else 0.02,
                0.01,
                key=preset,
            ),
            "build_days": st.slider("Build days", 30, 220, defaults["build_days"]),
            "utilization_curve_exponent": st.slider(
                "Utilization exponent", 1.0, 3.5, defaults["utilization_curve_exponent"], 0.1
            ),
            "bustout_sync_window_days": st.slider(
                "Bust-out window (days)", 1, 45, defaults["bustout_sync_window_days"]
            ),
            "playbook_variance": st.slider(
                "Playbook variance", 0.0, 0.8, defaults["playbook_variance"], 0.01
            ),
        }
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        st.form_submit_button("Generate and score", type="primary")

    fraud, accounts, utility = simulate(json.dumps(params, sort_keys=True), int(seed))
    fraud_accounts = accounts[accounts["label"].eq(1)]
    legit_accounts = accounts[accounts["label"].eq(0)]
    columns = st.columns(5)
    columns[0].metric("Fraud events", f"{len(fraud):,}")
    columns[1].metric("Synthetic identities", f"{fraud['customer_id'].nunique():,}")
    columns[2].metric("Attacker utility", f"{utility:.2f}")
    columns[3].metric("Mean A4 score", f"{fraud_accounts['a4_score'].mean():.2f}")
    columns[4].metric("Mean legit score", f"{legit_accounts['a4_score'].mean():.2f}")

    trajectory_tab, signals_tab, scores_tab, rows_tab, measured_tab = st.tabs(
        [
            "Identity trajectory",
            "Signal comparison",
            "Detector scores",
            "Generated rows",
            "Measured rounds",
        ]
    )
    with trajectory_tab:
        identity = st.selectbox("Synthetic identity", sorted(fraud["customer_id"].unique()))
        selected = fraud[fraud["customer_id"].eq(identity)].copy()
        selected["day"] = selected["meta"].map(lambda value: value["a4_days_since_onboarding"])
        selected["utilization"] = selected["meta"].map(lambda value: value["a4_utilization"])
        selected["credit_limit"] = selected["meta"].map(lambda value: value["a4_credit_limit"])
        selected["payment_ratio"] = selected["meta"].map(lambda value: value["a4_payment_ratio"])
        selected["phase"] = selected["meta"].map(lambda value: value["a4_phase"])
        st.line_chart(selected.set_index("day")[["utilization", "payment_ratio"]])
        st.dataframe(
            selected[["timestamp", "day", "phase", "amount", "credit_limit", "utilization"]],
            width="stretch",
            hide_index=True,
        )
    with signals_tab:
        comparison = accounts.groupby("attack_id")[list(FEATURE_NAMES)].mean().T
        st.dataframe(comparison, width="stretch")
    with scores_tab:
        bins = pd.cut(accounts["a4_score"], np.linspace(0, 1, 11), include_lowest=True)
        histogram = (
            accounts.groupby([bins, "attack_id"], observed=True).size().unstack(fill_value=0)
        )
        histogram.index = histogram.index.astype(str)
        st.bar_chart(histogram)
        st.dataframe(
            accounts.sort_values("a4_score", ascending=False).head(100),
            width="stretch",
        )
    with rows_tab:
        display = fraud.copy()
        display["phase"] = display["meta"].map(lambda value: value["a4_phase"])
        display["day"] = display["meta"].map(lambda value: value["a4_days_since_onboarding"])
        display["utilization"] = display["meta"].map(lambda value: value["a4_utilization"])
        st.dataframe(
            display[
                ["event_id", "timestamp", "customer_id", "day", "phase", "utilization", "amount"]
            ],
            width="stretch",
            hide_index=True,
        )
    with measured_tab:
        results = measured_results()
        if results is None:
            st.info(
                "Run the full experiment to populate this tab: "
                "`.venv/bin/python -m attacks.a4_synth_identity.experiment --codex`."
            )
        else:
            stages = [
                ("Round 0 · static", results["0"]["post_fix"]),
                ("Round 1 · static pre-fix", results["1"]["pre_fix"]),
                ("Round 1 · behavioral post-fix", results["1"]["post_fix"]),
                ("Round 2 · behavioral pre-fix", results["2"]["pre_fix"]),
                ("Round 2 · full recovery", results["2"]["post_fix"]),
            ]
            st.dataframe(
                pd.DataFrame([{"stage": stage, **metrics} for stage, metrics in stages]).set_index(
                    "stage"
                ),
                width="stretch",
            )
            budget = results["2"]["budget"]
            st.success(
                f"Final realism/utility budget accepted: {budget['accepted']} · "
                f"utility {budget['utility']:.3f} · amount KS {budget['max_ks']:.3f}"
            )
            for round_number in ("1", "2"):
                proposal = results[round_number]["proposal"]
                with st.expander(f"Round {round_number} Codex reasoning"):
                    st.write(proposal["reasoning"])
                    st.json(proposal["proposed_params"])


if __name__ == "__main__":
    main()
