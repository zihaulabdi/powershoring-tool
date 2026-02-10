"""
Powershoring Interactive Tool
==============================
Morocco industry targeting for powershoring — filter, score, and prioritize
energy-intensive trade-exposed products.

Run: streamlit run 02_Code/app/How_To.py
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
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = {}

st.title("Powershoring Industry Targeting Tool")
st.markdown("**Morocco** — Identifying energy-intensive industries for powershoring")

st.markdown("""
### How to use this tool

Navigate through the stages using the sidebar:

1. **Stage 1: Filtering** — Set energy intensity and trade volume thresholds to define the candidate universe
2. **Stage 2a: Likelihood** — Score products by how likely they are to powershore (energy intensity, CBAM exposure, incumbent vulnerability)
3. **Stage 2b: Prioritization** — Rank products by feasibility (Morocco capabilities) and attractiveness (market size, complexity, growth)
4. **Comparison** — Compare different scenarios and shortlists side by side

Each stage feeds into the next. Start with **Stage 1: Filtering**.
""")

st.info("Select a page from the sidebar to begin.")
