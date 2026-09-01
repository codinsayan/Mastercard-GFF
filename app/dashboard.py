"""Replay-only Streamlit dashboard entry point."""

import streamlit as st

from attacks.a1_upi_mule.demo import render_a1_demo

st.set_page_config(
    page_title="A1 · Adaptive Fraud Arena",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
render_a1_demo(results_root="results")
