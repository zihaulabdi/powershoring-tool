"""
Powershoring Interactive Tool
==============================
Morocco industry targeting for powershoring — filter, score, and prioritize
energy-intensive trade-exposed products.

Run: streamlit run How_To.py
"""

import streamlit as st

st.set_page_config(
    page_title="Powershoring Tool — Morocco",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for cross-page data passing
if "filtered_products" not in st.session_state:
    st.session_state.filtered_products = None
if "likelihood_products" not in st.session_state:
    st.session_state.likelihood_products = None
if "prioritized_products" not in st.session_state:
    st.session_state.prioritized_products = None
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = {}

st.title("Powershoring Industry Targeting Tool")
st.markdown("**Morocco** — Identifying energy-intensive industries for powershoring")

st.markdown("""
### How to use this tool

Navigate through the three stages using the sidebar:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
**1. Filtering**

Set energy intensity and trade volume thresholds to define the candidate product universe (~500 products from ~5,200).
""")

with col2:
    st.markdown("""
**2. Likelihood and Prioritization**

Score products by likelihood of powershoring (energy intensity, CBAM, incumbent vulnerability), then rank by feasibility (Morocco capabilities) and attractiveness (market opportunity).
""")

with col3:
    st.markdown("""
**3. Comparison**

Compare different shortlists and saved scenarios side by side — treemaps, bar charts, overlap table.
""")

st.info("Start with **Filtering** in the sidebar.")
