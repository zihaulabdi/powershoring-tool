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
st.caption("Harvard Growth Lab · OCP Morocco Green Growth Study")

st.divider()

st.markdown("## What this tool does")
st.markdown(
    "Powershoring refers to the relocation of energy-intensive, trade-exposed manufacturing to countries "
    "with cheap, clean electricity. As renewable costs fall and the EU Carbon Border Adjustment Mechanism "
    "(CBAM) raises the cost of carbon-intensive production, firms in sectors like aluminium, steel, chemicals, "
    "and industrial gases have growing incentives to move production. Morocco, with abundant solar and wind, "
    "proximity to European markets, and an industrial base in transition, is a plausible destination."
)
st.markdown(
    "The question this tool addresses is: *which industries, specifically?* "
    "The answer depends on what is assumed to drive relocation. An industry facing high CBAM exposure "
    "looks different from one where the top exporters are in energy-deficit countries. "
    "This tool makes those assumptions explicit, lets you adjust them, and shows how the candidate "
    "list changes as a result. The starting universe is roughly 5,200 traded products at the HS6 level. "
    "A typical analysis narrows that to 20 to 30 priority industries."
)

st.divider()

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        "**1. Filtering**\n\n"
        "Define the eligible universe by setting energy intensity and trade volume thresholds. "
        "Roughly 400 to 600 products survive."
    )
with col_b:
    st.markdown(
        "**2. Likelihood and Prioritization**\n\n"
        "Score the filtered pool on likelihood of relocation, then rank by Morocco's readiness "
        "and market opportunity."
    )
with col_c:
    st.markdown(
        "**3. Comparison**\n\n"
        "Save two scenarios with different assumptions and compare which products appear in both. "
        "Those are the robust candidates."
    )

st.divider()

# ============================================================
# STAGE 1
# ============================================================
with st.expander("Stage 1: Filtering", expanded=True):
    st.markdown(
        "The universe starts at roughly 5,200 HS6 products. HS6 is the most granular level of the "
        "Harmonized System, the international product classification used in customs data. "
        "Filtering removes products that are not plausible powershoring candidates before any scoring."
    )

    st.markdown("**What gets excluded by default**")
    st.markdown(
        "HS chapters 01 to 24 (agriculture and food) are excluded because powershoring is a manufacturing "
        "phenomenon driven by production energy costs, not agricultural input costs. "
        "HS chapters 25 to 27 (mining, stone, mineral fuels) are excluded because energy is a revenue "
        "source in extractive industries, not a cost input. Both exclusions can be turned off."
    )

    st.markdown("**Energy intensity**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "*Energy Intensity (Carriers), MJ per dollar of output.* "
            "Total energy use per dollar of production, combining fuel and electricity. "
            "Drawn from the US EPA USEEIO input-output model, mapped to HS products via a "
            "NAICS to HS concordance. Default threshold: 75th percentile of the full universe."
        )
    with col2:
        st.markdown(
            "*Electricity Intensity, MJ per dollar of output.* "
            "The electricity-specific share of total energy use. Some industries are fuel-heavy "
            "(cement, glass); others are nearly purely electric (aluminium smelting, chlorine production). "
            "Electricity intensity isolates the products most sensitive to electricity price differences. "
            "Default threshold: 50th percentile."
        )
    st.markdown(
        "The filter logic (OR / AND) determines whether a product needs to clear one threshold or both. "
        "OR is the default and produces a broader pool covering both fuel-intensive and "
        "electricity-intensive products. AND produces a narrower, electricity-focused list."
    )

    st.markdown("**Trade volume**")
    st.markdown(
        "Global export value sets a minimum market size. The default (15th percentile, roughly $50M) "
        "removes products that are barely traded internationally. "
        "The KPI row at the top of the Filtering page updates in real time. "
        "A pool of 400 to 600 products is typical."
    )

st.divider()

# ============================================================
# STAGE 2
# ============================================================
with st.expander("Stage 2: Likelihood and Prioritization", expanded=True):

    st.markdown("### Likelihood scoring")
    st.markdown(
        "Each product in the filtered pool receives a likelihood score from 0 to 100. "
        "The score is a weighted average of five components, each expressed as a percentile rank "
        "within the pool. Setting a weight to zero removes that component. "
        "The weights reflect your assumption about what drives industry relocation."
    )

    st.markdown("**Fuel Energy Intensity** (MJ of fuel energy per dollar of output)")
    st.markdown(
        "Industries with high fuel use in production stand to gain the most from switching to locations "
        "where clean electricity can substitute. Source: USEEIO via NAICS to HS concordance."
    )

    st.markdown("**Electricity Intensity** (MJ of electricity per dollar of output)")
    st.markdown(
        "The most direct signal of electricity cost sensitivity. When electricity is a large share "
        "of production costs, a meaningful price differential is sufficient to justify relocation. "
        "Source: USEEIO."
    )

    st.markdown("**Incumbent Vulnerability**")
    st.markdown(
        "Measures how energy-deficit the current top exporters of each product are. "
        "A country is energy-deficit if it is a net energy importer. If incumbents in a product "
        "are heavily energy-deficit, their cost position is structurally exposed to "
        "competition from low-electricity-cost producers. "
        "Computed from IEA energy balance data and UN Comtrade exporter shares."
    )

    st.markdown("**CBAM Exposure**")
    st.markdown(
        "The EU Carbon Border Adjustment Mechanism imposes a carbon cost on imports of "
        "iron and steel, aluminium, cement, fertilisers, electricity, and hydrogen into the EU, "
        "phased in from 2026. Products covered by CBAM with high EU market exposure face direct "
        "regulatory pressure to shift production to low-carbon sites. "
        "Source: EU Regulation 2023/956 and UN Comtrade EU import shares."
    )

    st.markdown("**Market Growth** (export CAGR 2012 to 2024)")
    st.markdown(
        "Growing markets are more receptive to new entrants. "
        "Useful for tilting the analysis toward dynamic sectors rather than mature commodities."
    )

    st.markdown("**Selection cutoff**")
    st.markdown(
        "After scoring, a cutoff determines which products pass the likelihood screen. "
        "Top 50% keeps the upper half of the distribution; Top N keeps exactly N products. "
        "Only products above the cutoff appear in the scatter and output list. "
        "The scatter title shows the exact count passing the cutoff and updates as you move the sliders."
    )

    st.divider()

    st.markdown("### Feasibility and Attractiveness")
    st.markdown(
        "Products passing the likelihood cutoff are scored on two dimensions. "
        "The composite score is a weighted average of both, with the balance set by the "
        "Feasibility/Attractiveness slider (default: 60/40). "
        "All scores are 0 to 100 percentile ranks within the selected pool."
    )

    col_f, col_a = st.columns(2)
    with col_f:
        st.markdown("**Feasibility: Morocco's readiness**")
        st.markdown(
            "*Morocco RCA.* Revealed Comparative Advantage measures whether Morocco already has a competitive "
            "export position in this product (RCA above 1 signals existing firms, skills, and supply chains).\n\n"
            "*Morocco Density.* How close is this product to Morocco's existing export basket in the "
            "product space? High density means the required capabilities are largely already present. "
            "Low density means Morocco would need to build from scratch. "
            "This is typically the most informative feasibility variable.\n\n"
            "*Market Fragmentation (1/HHI).* The Herfindahl-Hirschman Index measures how concentrated "
            "a market is among existing exporters. A fragmented market is one where no "
            "single country dominates, which is easier for a new entrant to enter.\n\n"
            "*Avg. Trade Distance.* Average km between producers and buyers, weighted by "
            "trade flows. Products traded over shorter distances are more transport-cost-sensitive "
            "and benefit more from Morocco's proximity to European markets."
        )

    with col_a:
        st.markdown("**Attractiveness: Market opportunity**")
        st.markdown(
            "*Market Size.* Total global export value. Larger markets mean more revenue "
            "potential at any given market share.\n\n"
            "*Market Growth.* Export CAGR 2012 to 2024. Fast-growing markets are more "
            "forgiving of an imperfect entry, since share can be gained without displacing incumbents.\n\n"
            "*COG (Complexity Outlook Gain).* How much does entering this product expand Morocco's "
            "future diversification options? COG measures the new, higher-complexity products that become "
            "reachable via the product space once this product is mastered. "
            "High COG products are strategic stepping stones.\n\n"
            "*Product Complexity (PCI).* How knowledge-intensive is this product? "
            "PCI is derived from the structure of the global export network. Products that only a "
            "few, highly diversified countries produce have high complexity and generally "
            "carry higher value added.\n\n"
            "*Spillover Potential.* Network centrality of the NAICS industry. "
            "Central industries generate knowledge and input linkages that benefit adjacent sectors."
        )

    st.divider()

    st.markdown("### The Feasibility vs. Attractiveness scatter")
    st.markdown(
        "The scatter is the primary output of this stage. Each bubble is one product that passed "
        "the likelihood cutoff. The x-axis is feasibility (Morocco's readiness, 0 to 100), "
        "the y-axis is attractiveness (market opportunity, 0 to 100), and bubble size reflects "
        "global trade volume. Red bubbles are the top N by composite score. "
        "Grey bubbles passed the likelihood cutoff but fall outside the top N."
    )
    st.markdown(
        "The dashed lines mark the medians of the selected pool. "
        "The upper-right quadrant is where the strongest candidates sit, scoring above median "
        "on both dimensions. Upper-left (high attractiveness, lower feasibility) are longer-horizon "
        "targets that require capability investment first. "
        "Lower-right (high feasibility, lower attractiveness) are easier entry points "
        "with more limited strategic payoff."
    )
    st.markdown(
        "The scatter updates dynamically. Changing the likelihood weights shifts which products "
        "pass the cutoff. Tightening the cutoff reduces the number of bubbles shown."
    )

st.divider()

# ============================================================
# STAGE 3
# ============================================================
with st.expander("Stage 3: Comparison", expanded=False):
    st.markdown(
        "The Comparison page is for testing robustness. Save two scenarios on the "
        "Likelihood and Prioritization page, for example one weighted toward CBAM exposure "
        "and one weighted toward electricity intensity, then compare them side by side. "
        "Products that appear in both shortlists are robust to the choice of assumption. "
        "Products that appear in only one are theory-dependent candidates worth closer examination."
    )
    st.markdown(
        "To save a scenario: set your weights and cutoff, enter a name in the Save Scenario box "
        "at the bottom of the sidebar, and click Save. Repeat with different settings. "
        "Both saved scenarios will be available in the Comparison page selector."
    )

st.divider()
st.markdown("Start with **Filtering** in the sidebar.")
