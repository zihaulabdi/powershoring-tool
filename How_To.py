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
    "with cheap, clean electricity. As renewable costs fall and the EU's Carbon Border Adjustment Mechanism "
    "(CBAM) raises the cost of carbon-intensive production, firms in sectors like aluminium, steel, chemicals, "
    "and industrial gases face growing pressure to move. Morocco — with abundant solar and wind, proximity "
    "to European markets, and a stated ambition to become a green industrial hub — is a plausible destination."
)
st.markdown(
    "The question this tool addresses is: *which industries, specifically?* "
    "The answer depends on what you believe drives relocation. An industry facing high CBAM exposure "
    "looks different from one where the top exporters happen to be in energy-deficit countries. "
    "This tool makes those assumptions explicit, lets you adjust them, and shows how the candidate "
    "list changes as a result. The starting universe is ~5,200 traded products (HS6 level). "
    "A typical analysis narrows that to 20–30 priority industries."
)

st.divider()

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.info("**1. Filtering**\nDefine the eligible universe by setting energy intensity and trade volume thresholds. Roughly 400–600 products survive.")
with col_b:
    st.info("**2. Likelihood & Prioritization**\nScore the filtered pool on likelihood of relocation, then rank by Morocco's readiness and market opportunity.")
with col_c:
    st.info("**3. Comparison**\nSave two scenarios with different assumptions and compare which products appear in both — those are the robust candidates.")

st.divider()

# ============================================================
# STAGE 1
# ============================================================
with st.expander("Stage 1 — Filtering", expanded=True):
    st.markdown(
        "The universe starts at ~5,200 HS6 products. HS6 is the most granular level of the Harmonized "
        "System — the international product classification used in customs data. Filtering removes "
        "products that are not plausible powershoring candidates before any scoring takes place."
    )

    st.markdown("**What gets excluded by default**")
    st.markdown(
        "HS chapters 01–24 (agriculture and food) are excluded because powershoring is driven by "
        "manufacturing energy costs, not agricultural input costs. HS chapters 25–27 (mining, stone, "
        "mineral fuels) are excluded because energy is a revenue source for extractive industries, "
        "not a cost input. Both exclusions can be turned off."
    )

    st.markdown("**Energy intensity**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "*Energy Intensity (Carriers) — MJ per $ of output.* "
            "Total energy use per dollar of production, combining fuel and electricity. "
            "Drawn from the US EPA's USEEIO input-output model, mapped to HS products via a "
            "NAICS–HS concordance. Default threshold: 75th percentile of the full universe."
        )
    with col2:
        st.markdown(
            "*Electricity Intensity — MJ per $ of output.* "
            "The electricity-specific share of total energy use. Some industries are fuel-heavy "
            "(cement, glass); others are almost purely electric (aluminium smelting, chlorine production). "
            "Electricity intensity isolates the products most sensitive to electricity price differences. "
            "Default threshold: 50th percentile."
        )
    st.markdown(
        "The filter logic (OR / AND) determines whether a product needs to clear one threshold or both. "
        "OR is the default — it produces a broader pool that includes both fuel-intensive and "
        "electricity-intensive products. AND produces a narrower, electricity-focused list."
    )

    st.markdown("**Trade volume**")
    st.markdown(
        "Global export value sets a minimum market size. The default (15th percentile, roughly $50M) "
        "removes products that are barely traded internationally — where no meaningful entry opportunity "
        "exists. The KPI row at the top of the Filtering page updates in real time as you adjust "
        "thresholds. A pool of 400–600 products is typical."
    )

st.divider()

# ============================================================
# STAGE 2
# ============================================================
with st.expander("Stage 2 — Likelihood & Prioritization", expanded=True):

    st.markdown("### Likelihood scoring")
    st.markdown(
        "Each product in the filtered pool receives a likelihood score (0–100). "
        "The score is a weighted average of five components, each expressed as a percentile rank "
        "within the pool. Setting a weight to zero removes that component entirely. "
        "The weights reflect your assumption about why industries relocate."
    )

    st.markdown("**Fuel Energy Intensity** — MJ of fuel energy per $ of output.")
    st.markdown(
        "Industries that burn a lot of fuel in production gain the most from switching to locations "
        "where clean electricity can substitute. Source: USEEIO via NAICS–HS concordance."
    )

    st.markdown("**Electricity Intensity** — MJ of electricity per $ of output.")
    st.markdown(
        "The most direct signal of electricity cost sensitivity. When electricity is a large share "
        "of production costs, a meaningful price differential is enough to justify relocation. "
        "Source: USEEIO."
    )

    st.markdown("**Incumbent Vulnerability**")
    st.markdown(
        "Measures how energy-deficit the current top exporters of each product are. "
        "A country is energy-deficit if it is a net energy importer. If the incumbents in a product "
        "are heavily energy-deficit, their cost position is structurally weak — they are exposed to "
        "competition from low-electricity-cost producers. Higher score = more disruptable market. "
        "Computed from IEA energy balance data and UN Comtrade exporter shares."
    )

    st.markdown("**CBAM Exposure**")
    st.markdown(
        "The EU's Carbon Border Adjustment Mechanism (CBAM) imposes a carbon cost on imports of "
        "iron and steel, aluminium, cement, fertilisers, electricity, and hydrogen into the EU, "
        "phased in from 2026. Products covered by CBAM with high EU market exposure face direct "
        "regulatory pressure to shift production to low-carbon sites. "
        "Source: EU Regulation 2023/956 + UN Comtrade EU import shares."
    )

    st.markdown("**Market Growth** — Export CAGR 2012–2024.")
    st.markdown(
        "Growing markets are more receptive to new entrants and offer stronger long-run returns. "
        "Useful for tilting the analysis toward dynamic sectors rather than mature commodities."
    )

    st.markdown("**Selection cutoff**")
    st.markdown(
        "After scoring, a cutoff determines which products 'pass' the likelihood screen. "
        "Top 50% keeps the upper half of the distribution; Top N keeps exactly N products. "
        "Only products above the cutoff appear in the scatter and output list. "
        "The scatter title shows the exact count passing the cutoff — it updates as you move the sliders."
    )

    st.divider()

    st.markdown("### Feasibility and Attractiveness")
    st.markdown(
        "Products passing the likelihood cutoff are scored on two dimensions. "
        "The composite score is a weighted average of both, with the balance set by the "
        "Feasibility/Attractiveness slider (default: 60/40). "
        "All scores are 0–100 percentile ranks within the selected pool."
    )

    col_f, col_a = st.columns(2)
    with col_f:
        st.markdown("**Feasibility — Morocco's readiness**")
        st.markdown(
            "*Morocco RCA.* Revealed Comparative Advantage — whether Morocco already has a competitive "
            "export position in this product (RCA > 1). A higher RCA means existing firms, skills, "
            "and supply chains are already present.\n\n"
            "*Morocco Density.* How close is this product to Morocco's existing export basket in the "
            "product space? High density means the capabilities needed are already largely present. "
            "Low density means Morocco would need to build from scratch. "
            "This is typically the most informative feasibility variable.\n\n"
            "*Market Fragmentation (1/HHI).* The Herfindahl-Hirschman Index measures how concentrated "
            "a market is among existing exporters. A fragmented market (high 1/HHI) is one where no "
            "single country dominates — easier for a new entrant to gain share.\n\n"
            "*Avg. Trade Distance.* Average distance in km between producers and buyers, weighted by "
            "trade flows. Products traded over shorter distances are more transport-cost-sensitive "
            "and benefit more from Morocco's proximity to European markets."
        )

    with col_a:
        st.markdown("**Attractiveness — Market opportunity**")
        st.markdown(
            "*Market Size.* Total global export value. Larger markets mean more absolute revenue "
            "at any given market share.\n\n"
            "*Market Growth.* Export CAGR 2012–2024. A fast-growing market is more forgiving of "
            "an imperfect entry — share can be gained without displacing incumbents.\n\n"
            "*COG — Complexity Outlook Gain.* How much does entering this product expand Morocco's "
            "future diversification options? COG measures the new, high-complexity products that become "
            "accessible via the product space once this product is mastered. High COG products are "
            "strategic entry points — not just valuable in themselves, but as stepping stones.\n\n"
            "*Product Complexity (PCI).* How knowledge-intensive is this product? "
            "PCI is derived from the structure of the global export network — products that only a "
            "few, highly diversified countries can make have high complexity. "
            "Higher PCI generally means higher value added and more durable competitive advantages.\n\n"
            "*Spillover Potential.* Network centrality of the NAICS industry in the US input-output "
            "network. Central industries generate knowledge and input linkages that benefit a wider "
            "range of adjacent sectors."
        )

    st.divider()

    st.markdown("### The Feasibility vs. Attractiveness scatter")
    st.markdown(
        "The scatter is the primary output of this stage. Each bubble is one product that passed "
        "the likelihood cutoff. The x-axis is feasibility (Morocco's readiness, 0–100), "
        "the y-axis is attractiveness (market opportunity, 0–100), and bubble size reflects "
        "global trade volume. Red bubbles are the top N by composite score. "
        "Grey bubbles passed the likelihood cutoff but fall outside the top N."
    )
    st.markdown(
        "The dashed lines mark the medians of the selected pool. "
        "The upper-right quadrant — above median on both dimensions — is where the strongest "
        "candidates sit. Upper-left (high attractiveness, low feasibility) are longer-horizon "
        "targets that require capability investment first. "
        "Lower-right (high feasibility, low attractiveness) are easier entry points but "
        "with limited strategic payoff."
    )
    st.markdown(
        "The scatter updates dynamically: changing the likelihood weights shifts which products "
        "pass the cutoff, and tightening the cutoff reduces the number of bubbles shown."
    )

st.divider()

# ============================================================
# STAGE 3
# ============================================================
with st.expander("Stage 3 — Comparison", expanded=False):
    st.markdown(
        "The Comparison page is for testing robustness. Save two scenarios on the "
        "Likelihood & Prioritization page — for example, one weighted toward CBAM exposure "
        "and one weighted toward electricity intensity — then compare them side by side. "
        "Products that appear in both shortlists are robust to the choice of assumption. "
        "Products that appear in only one are theory-dependent candidates worth scrutinising more carefully."
    )
    st.markdown(
        "To save a scenario: set your weights and cutoff, enter a name in the Save Scenario box "
        "at the bottom of the sidebar, and click Save. Repeat with different settings. "
        "Both saved scenarios will be available in the Comparison page selector."
    )

st.divider()

# ============================================================
# DATA SOURCES
# ============================================================
with st.expander("Data sources and coverage", expanded=False):
    st.markdown(
        "| Variable group | Source |\n"
        "|---|---|\n"
        "| Trade values, RCA, HHI, CAGR, exporter shares | UN Comtrade via Growth Lab Atlas (2021; panel 2012–2024) |\n"
        "| Energy intensities (fuel, electricity, carriers) | US EPA USEEIO model, mapped to HS6 via NAICS–HS concordance |\n"
        "| Product Complexity (PCI), Density, COG | Harvard Growth Lab Atlas of Economic Complexity |\n"
        "| Spillover / network centrality | NAICS industry network, Growth Lab |\n"
        "| CBAM coverage and scores | EU Regulation 2023/956; EU import share from UN Comtrade |\n"
        "| Incumbent vulnerability | IEA energy balance data + UN Comtrade exporter shares |\n"
        "| Trade distance | CEPII GeoDist + bilateral trade flows |\n\n"
        "The master dataset covers 5,204 HS6 products across 97 HS2 chapters, "
        "representing approximately $20 trillion in global trade. "
        "Energy intensity data is available for 84% of products (4,393 of 5,204); "
        "the remainder is assigned the pool median."
    )

st.divider()
st.markdown("Start with **Filtering** in the sidebar.")
