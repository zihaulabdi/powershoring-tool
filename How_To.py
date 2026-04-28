"""
Powershoring Interactive Tool
==============================
Morocco industry targeting for powershoring: filter, score, and prioritize
energy-intensive trade-exposed products.

Run: streamlit run 02_Code/app/How_To.py
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import inject_custom_css

st.set_page_config(
    page_title="Introduction",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Initialize session state for cross-page data passing
for _key, _default in [
    ("filtered_products", None),
    ("likelihood_products", None),
    ("prioritized_products", None),
    ("saved_scenarios", {}),
    ("stage_1_complete", False),
    ("stage_2_complete", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

st.title("Introduction")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### What this tool does")
    st.markdown(
        "Systematically screens ~5,000 globally traded products to identify which "
        "energy-intensive industries Morocco should target for powershoring: attracting "
        "firms that benefit from cheap, clean electricity."
    )

with col2:
    st.markdown("### Who it's for")
    st.markdown(
        "**Policy researchers and analysts** conducting systematic industry targeting.\n\n"
        "**Senior stakeholders** can start with the **Summary** page for an immediate answer."
    )

with col3:
    st.markdown("### How to start")
    st.markdown(
        "**Quickest path:** Go to **Summary** for instant top candidates.\n\n"
        "**Full analysis:** Run Filtering, then Likelihood and Prioritization, then Scenarios.\n\n"
        "All controls are always visible — adjust weights and thresholds as needed."
    )

st.divider()

st.markdown("#### Analysis pipeline")
p1, arr1, p2, arr2, p3, arr3, p4 = st.columns([2, 0.3, 2, 0.3, 2, 0.3, 2])
with p1:
    st.info("**1. Filtering**\nDefine the candidate universe by energy and trade thresholds.")
with arr1:
    st.markdown("<div style='text-align:center; font-size:28px; padding-top:30px'>→</div>", unsafe_allow_html=True)
with p2:
    st.info("**2. Likelihood and Prioritization**\nScore by relocation likelihood, then rank by feasibility and attractiveness.")
with arr2:
    st.markdown("<div style='text-align:center; font-size:28px; padding-top:30px'>→</div>", unsafe_allow_html=True)
with p3:
    st.info("**3. Scenarios**\nCross-reference 4 scenarios to find robust candidates.")
with arr3:
    st.markdown("<div style='text-align:center; font-size:28px; padding-top:30px'>→</div>", unsafe_allow_html=True)
with p4:
    st.info("**4. Summary**\nTop candidates at a glance using default assumptions.")

st.divider()

with st.expander("Detailed methodology and variable definitions"):
    st.markdown("""
**Stage 1: Filtering**
Reduce ~5,000 HS6 products to ~500 candidates using energy intensity thresholds (fuel + electricity per $ of output) and minimum trade volume. Excludes raw materials and extractive products.

**Stage 2a: Likelihood scoring**
Four scenarios, each representing a different theory of why industries relocate:
- *No Prior*: Equal weight on electricity intensity, incumbent vulnerability, and CBAM exposure
- *Electricity Cost*: Pure electricity intensity, industries where electricity dominates production costs
- *Carbon Regulation*: CBAM-covered products facing EU carbon border pressure
- *Disruption Opportunity*: Where current top exporters are most energy-deficit and vulnerable

**Stage 2b: Prioritization**
Composite score = Feasibility x weight + Attractiveness x weight
- *Feasibility*: How ready is Morocco? (capability proximity, existing RCA, market openness, trade distance)
- *Attractiveness*: How valuable is the market? (size, growth, diversification gain, complexity, spillovers)

**Scenario Analysis**
Run all 4 scenarios simultaneously and identify products appearing in 2 or more scenario top lists. These are robust candidates regardless of which relocation driver you believe in.

**Key variables**
- *Capability proximity (Density)*: How close a product is to Morocco's existing export basket in the product space
- *Diversification value (COG)*: How much entering this product would improve Morocco's future diversification options
- *Market openness (1/HHI)*: How fragmented the global market is. More fragmented = easier entry
- *Incumbent vulnerability*: How energy-deficit the current top exporters are. Higher = more disruptable
    """)
