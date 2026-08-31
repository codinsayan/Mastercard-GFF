"""Replay-only Streamlit demo for A3 agentic card testing.

Renders the already-committed evaluation, presentation, and fidelity artifacts
under ``results/``. It does not regenerate campaigns, run the detector, or
invoke Codex/CEM -- every number and chart shown here was produced by
``evaluation.py``, ``presentation.py``, and ``fidelity.py`` ahead of time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

A3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = A3_DIR.parents[1]
RESULTS_DIR = A3_DIR / "results"
PRESENTATION_DIR = RESULTS_DIR / "presentation"

SUMMARY_PATH = RESULTS_DIR / "summary.json"
PRESENTATION_JSON_PATH = PRESENTATION_DIR / "presentation.json"
CARD_PATH = A3_DIR / "card.json"
FIDELITY_PATH = PROJECT_ROOT / "results" / "fidelity.json"

SVG_CHARTS = {
    "round_metrics.svg": "Detector performance across adaptive rounds",
    "throughput.svg": "Attacker throughput: measured cards per hour",
    "signal_mortality.svg": "Signal mortality (direction-normalized ROC AUC)",
    "bin_clustering.svg": "BIN clustering and detection-signal persistence",
}


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _missing(label: str, path: Path) -> None:
    st.warning(
        f"{label} not found at `{path.relative_to(PROJECT_ROOT)}`. "
        "Run the evaluation/presentation/fidelity commands documented in the A3 README first."
    )


def _round_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for round_result in summary["rounds"]:
        metrics = round_result["detector_metrics"]
        frozen = round_result.get("frozen_previous_detector_metrics")
        rows.append(
            {
                "Requested cards/hr": round_result["requested_cards_per_hour"],
                "Actual cards/hr": round_result["actual_cards_per_hour"],
                "Utility": round(round_result["utility"], 4),
                "Active ROC AUC": round(metrics["roc_auc"], 4),
                "Active recall": round(metrics["recall"], 4),
                "Active precision": round(metrics["precision"], 4),
                "Frozen prev-head recall": round(frozen["recall"], 4) if frozen else None,
            }
        )
    return pd.DataFrame(rows)


def _signal_mortality_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).rename(
        columns={
            "feature": "Signal",
            "round_0_auc": "Round 0 AUC",
            "round_1_auc": "Round 1 AUC",
            "round_2_auc": "Round 2 AUC",
            "round_0_to_1_delta": "Round 0 to 1 delta",
        }
    )
    return frame[["Signal", "Round 0 AUC", "Round 1 AUC", "Round 2 AUC", "Round 0 to 1 delta"]]


def _svg(name: str, caption: str) -> None:
    path = PRESENTATION_DIR / name
    if not path.exists():
        _missing(caption, path)
        return
    st.image(str(path), caption=caption, width="stretch")


def _demo_takeaway() -> None:
    st.markdown(
        """
<div style="
    padding: 18px 22px;
    border-radius: 10px;
    background: linear-gradient(90deg, #ff5f00 0%, #eb001b 100%);
    color: #ffffff;
    margin-bottom: 8px;
">
  <div style="font-size: 13px; letter-spacing: 1px; text-transform: uppercase; opacity: 0.85;">
    Demo takeaway
  </div>
  <div style="font-size: 22px; font-weight: 700; margin-top: 4px;">
    The attacker adapts after the defender learns its first-generation behavior.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _attack_tab(card: dict[str, Any] | None) -> None:
    st.subheader("What is agentic card testing?")
    st.markdown(
        """
An attacker sits on a batch of **stolen card numbers** ("card dumps") and needs to find out,
cheaply and quickly, which ones still work before using them for real fraud. Instead of a human
doing this by hand, an **agent** automates the process: it picks a merchant, fires a small
**card-not-present (CNP)** transaction (no physical card or chip involved -- just card number,
expiry, and amount, like an online checkout), watches whether it is approved or declined, and
moves on to the next card -- adjusting its own behavior as it goes to stay under the radar.
"""
    )
    glossary = {
        "CNP transaction (card-not-present)": (
            "A payment where the card itself is never physically presented -- typical of online "
            "or phone purchases. It's the easiest channel for an attacker to script."
        ),
        "BIN dump": (
            "A batch of stolen card numbers that share the same 6-digit Bank Identification "
            "Number (BIN) prefix because they came from the same breach source. Cards from one "
            "dump cluster together, which is exactly the pattern a defender can look for."
        ),
        "Device rotation": (
            "Switching the device/browser fingerprint used to submit transactions after a few "
            "cards, so no single device appears to be testing an unusually large number of cards."
        ),
        "Merchant friction": (
            "How much verification a merchant demands (e.g. strong authentication). HIGH-friction "
            "merchants reliably decline every attempt regardless of whether the card is live, so "
            "they waste attacker attempts without leaking information -- but attackers can still "
            "route around them by favoring LOW-friction merchants."
        ),
        "Decline / retry behavior": (
            "If a card is declined at a high-friction merchant, the agent may retry the same card "
            "moments later at a different, lower-friction merchant with a slightly adjusted amount "
            "-- a distinctive pattern in itself."
        ),
    }
    for term, explanation in glossary.items():
        with st.expander(term):
            st.write(explanation)

    if card:
        with st.expander("How the defender sees it -- the 9 signals fed to the detector"):
            for name, description in card.get("defender_signals", {}).items():
                st.markdown(f"**`{name}`** -- {description}")
    else:
        _missing("card.json", CARD_PATH)


def _rounds_tab(summary: dict[str, Any] | None) -> None:
    if not summary:
        _missing("summary.json", SUMMARY_PATH)
        return

    st.subheader("Three measured adaptive rounds")
    st.markdown(
        "Round 0 &rarr; **Detect** &rarr; **Adapt** &rarr; Round 1 &rarr; **Retrain** &rarr; "
        "**Adapt** &rarr; Round 2",
        unsafe_allow_html=True,
    )
    flow = st.columns(7)
    steps = [
        ("Round 0", "40 cards/hr baseline campaign"),
        ("Detect", "Fit detector on round-0 features"),
        ("Adapt", "Codex + CEM propose an evasion"),
        ("Round 1", "12 cards/hr adapted campaign"),
        ("Retrain", "Fit detector on all 9 signals"),
        ("Adapt", "Codex + CEM propose again"),
        ("Round 2", "4 cards/hr adapted campaign"),
    ]
    for column, (title, caption) in zip(flow, steps, strict=True):
        with column:
            st.markdown(f"**{title}**")
            st.caption(caption)

    st.divider()
    throughput = summary["throughput"]
    cols = st.columns(3)
    for column, entry in zip(cols, throughput, strict=True):
        with column:
            st.metric(
                f"Round {entry['round']} throughput",
                f"{entry['actual']:.0f} cards/hr",
                help=f"Target {entry['target']} cards/hr, utility {entry['utility']:.4f}",
            )

    st.subheader("Detector metrics per round")
    table = _round_table(summary)
    table["Round"] = [entry["round"] for entry in throughput]
    table = table[
        [
            "Round",
            "Requested cards/hr",
            "Actual cards/hr",
            "Utility",
            "Active ROC AUC",
            "Active recall",
            "Active precision",
            "Frozen prev-head recall",
        ]
    ]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Round 1's frozen prev-head recall (0.2479) shows the round-0 detector, still using only "
        "its original 4 signals, collapsing against round-1 traffic -- the mortality event. "
        "Retraining on all 9 signals restores recall to 1.0."
    )


def _signal_tab(presentation: dict[str, Any] | None) -> None:
    st.subheader("Signal mortality")
    _svg("signal_mortality.svg", SVG_CHARTS["signal_mortality.svg"])
    if not presentation:
        _missing("presentation.json", PRESENTATION_JSON_PATH)
        return
    st.dataframe(
        _signal_mortality_table(presentation["signal_mortality"]),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Direction-normalized ROC AUC = max(AUC, 1 - AUC), measured on held-out campaigns per round."
    )


def _bin_tab(presentation: dict[str, Any] | None) -> None:
    st.subheader("BIN clustering and campaign throughput")
    col1, col2 = st.columns(2)
    with col1:
        _svg("bin_clustering.svg", SVG_CHARTS["bin_clustering.svg"])
    with col2:
        _svg("throughput.svg", SVG_CHARTS["throughput.svg"])
    if presentation:
        st.dataframe(
            pd.DataFrame(presentation["bin_clustering"]).rename(
                columns={
                    "round": "Round",
                    "cards": "Cards",
                    "distinct_bins": "Distinct BINs",
                    "dominant_bin_share": "Dominant BIN share",
                    "normalized_bin_entropy": "Normalized BIN entropy",
                    "bin_cooccurrence_auc": "BIN co-occurrence AUC",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        _missing("presentation.json", PRESENTATION_JSON_PATH)


def _fidelity_tab(fidelity: dict[str, Any] | None) -> None:
    st.subheader("Synthetic-vs-real fidelity")
    if not fidelity:
        _missing("results/fidelity.json", FIDELITY_PATH)
        return
    if fidelity.get("status") != "complete":
        st.warning(f"Fidelity status: `{fidelity.get('status')}` -- no dataset was measured.")
        return

    ks_summary = fidelity["ks_summary"]
    correlation = fidelity["correlation_matrix_distance"]
    discriminator = fidelity["discriminator"]
    dataset = fidelity["dataset"]
    synthetic = fidelity["synthetic"]

    row1 = st.columns(4)
    row1[0].metric("KS mean", f"{ks_summary['mean']:.4f}")
    row1[1].metric("KS maximum", f"{ks_summary['maximum']:.4f}")
    row1[2].metric("Correlation matrix distance", f"{correlation['value']:.4f}")
    row1[3].metric("Discriminator ROC AUC", f"{discriminator['roc_auc']:.4f}")

    row2 = st.columns(4)
    row2[0].metric("Real rows read", f"{dataset['rows_read']:,}")
    row2[1].metric("Real rows sampled", f"{dataset['rows_sampled']:,}")
    row2[2].metric("Synthetic rows compared", f"{synthetic['card_rows_compared']:,}")
    row2[3].metric("Dataset format", dataset["format"])

    with st.expander("Per-feature KS statistics"):
        ks_table = pd.DataFrame(
            sorted(fidelity["ks_statistics"].items()), columns=["Feature", "KS statistic"]
        )
        st.dataframe(ks_table, hide_index=True, width="stretch")

    with st.expander("Discriminator feature importance (gain)"):
        importance = pd.DataFrame(
            sorted(
                discriminator["feature_importance_gain"].items(), key=lambda item: -item[1]
            ),
            columns=["Feature", "Gain"],
        )
        st.dataframe(importance, hide_index=True, width="stretch")

    st.caption(
        f"Source: `{dataset['path']}` (SHA-256 `{dataset['sha256'][:16]}...`), "
        f"{dataset['valid_legitimate_rows']:,} valid legitimate rows, seed {synthetic['seed']}."
    )


def _evidence_tab() -> None:
    st.subheader("All measured charts")
    columns = st.columns(2)
    for index, (name, caption) in enumerate(SVG_CHARTS.items()):
        with columns[index % 2]:
            _svg(name, caption)


def main() -> None:
    st.set_page_config(page_title="A3 Agentic Card Testing", layout="wide", page_icon="\U0001f575️")
    st.title("A3 · Agentic Card Testing")
    st.caption(
        "Replay-only demo. Every number and chart below was produced ahead of time by "
        "`evaluation.py`, `presentation.py`, and `fidelity.py` -- nothing is recomputed here."
    )
    _demo_takeaway()

    summary = _load_json(str(SUMMARY_PATH))
    presentation = _load_json(str(PRESENTATION_JSON_PATH))
    fidelity = _load_json(str(FIDELITY_PATH))
    card = _load_json(str(CARD_PATH))

    tabs = st.tabs(
        [
            "A3 Attack",
            "3-Round Attack",
            "Signal Mortality",
            "BIN / Campaign Behavior",
            "Fidelity",
            "Evidence / Charts",
        ]
    )
    with tabs[0]:
        _attack_tab(card)
    with tabs[1]:
        _rounds_tab(summary)
    with tabs[2]:
        _signal_tab(presentation)
    with tabs[3]:
        _bin_tab(presentation)
    with tabs[4]:
        _fidelity_tab(fidelity)
    with tabs[5]:
        _evidence_tab()


if __name__ == "__main__":
    main()
