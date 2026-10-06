"""
Powershoring Interactive Tool
==============================
Morocco industry targeting for powershoring: filter, score, and prioritize
energy-intensive trade-exposed products.

This file is the entry point. It sets up the theme and the top navigation;
each page's content lives in pages/.

Run: streamlit run How_To.py   (from 02_Code/app/ or the deployment repo root)
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui  # noqa: E402

st.set_page_config(
    page_title="Morocco Powershoring",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.apply_theme()

# Session state shared across pages
st.session_state.setdefault("saved_scenarios", {})
st.session_state.setdefault("_store", {})

# Page order: the answer first, then the three analysis steps, then tools.
pages = [
    st.Page("pages/0_Overview.py", title="Overview", url_path="overview", default=True),
    st.Page("pages/5_Summary.py", title="Shortlist", url_path="shortlist"),
    st.Page("pages/1_Filtering.py", title="1 · Candidate pool", url_path="filter"),
    st.Page("pages/2_Likelihood_Prioritization.py", title="2 · Score and rank", url_path="score"),
    st.Page("pages/3_Scenarios.py", title="3 · Scenarios", url_path="scenarios"),
    st.Page("pages/4_Expert_Comparison.py", title="Compare", url_path="compare"),
]
st.navigation(pages, position="top").run()
