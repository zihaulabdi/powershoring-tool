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

# ============================================================
# WHAT IS POWERSHORING
# ============================================================
st.markdown("## What is powershoring?")
st.markdown(
    "**Powershoring** is the relocation or expansion of energy-intensive, trade-exposed industries "
    "to countries with competitive, low-carbon electricity. As renewable energy costs fall and carbon "
    "regulations tighten (notably the EU's Carbon Border Adjustment Mechanism, CBAM), firms in "
    "energy-intensive sectors have a growing incentive to locate production where electricity is "
    "cheap, clean, and abundant. Morocco — with large-scale solar and wind resources, proximity to "
    "European markets, and an industrial base in transition — is a strong candidate host."
)
st.markdown(
    "This tool systematically screens ~5,200 globally traded products across three stages to identify "
    "which industries Morocco should target, under what assumptions, and with what confidence."
)

st.divider()

# ============================================================
# THE THREE-STAGE PIPELINE
# ============================================================
st.markdown("## How the tool works")

st.markdown(
    "The analysis follows a **funnel**: start with all traded products, eliminate ineligible ones, "
    "score the rest on likelihood of relocation, then rank by Morocco's readiness and market opportunity."
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.info(
        "**Stage 1 — Filtering**\n\n"
        "Reduce ~5,200 products to ~400–600 candidates by setting energy intensity and trade volume thresholds. "
        "Removes products that are too low-energy or too thinly traded to be relevant."
    )
with col_b:
    st.info(
        "**Stage 2 — Likelihood & Prioritization**\n\n"
        "Score surviving products by how likely they are to relocate to Morocco, then rank by feasibility "
        "(Morocco's readiness) and attractiveness (market opportunity)."
    )
with col_c:
    st.info(
        "**Stage 3 — Comparison**\n\n"
        "Compare different saved shortlists side by side — useful for testing which products are robust "
        "across different weighting assumptions."
    )

st.divider()

# ============================================================
# STAGE 1: FILTERING
# ============================================================
with st.expander("📋  Stage 1 — Filtering: variables and controls", expanded=True):
    st.markdown("### What filtering does")
    st.markdown(
        "The universe starts at ~5,200 HS6 products (6-digit Harmonized System codes — the most granular "
        "level of internationally traded goods). Filtering eliminates products that are not plausible "
        "powershoring candidates before any scoring occurs."
    )

    st.markdown("### Exclusions")
    st.markdown(
        "- **Non-manufacturing (HS 01–24):** Agricultural and food products. These are excluded by default "
        "because powershoring is a manufacturing phenomenon driven by production energy costs.\n"
        "- **Extractive (HS 25–27):** Mining, mineral fuels, and quarrying products. Energy cost is a "
        "revenue driver here, not a cost input — irrelevant for powershoring targeting."
    )

    st.markdown("### Energy intensity thresholds")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Energy Intensity (Carriers) — MJ per $ of output**\n\n"
            "Total energy input per dollar of production output, covering all fuel and electricity "
            "combined. Sourced from the US EPA's USEEIO model, mapped from NAICS industries to "
            "HS products via a concordance. Higher = more energy-dependent production.\n\n"
            "*Default threshold: 75th percentile of the full universe.*"
        )
    with col2:
        st.markdown(
            "**Electricity Intensity — MJ per $ of output**\n\n"
            "The electricity-specific component of total energy use. Some industries use a lot of total "
            "energy but mostly as fuel (e.g. cement); others are almost purely electric (e.g. aluminium "
            "smelting). Electricity intensity captures the subset most directly affected by electricity "
            "price differences.\n\n"
            "*Default threshold: 50th percentile.*"
        )

    st.markdown(
        "**Filter logic (OR / AND):** With OR (default), a product qualifies if it exceeds *either* "
        "the energy or electricity threshold. AND requires both — producing a more conservative, "
        "electricity-focused shortlist."
    )

    st.markdown("### Trade volume threshold")
    st.markdown(
        "**Global Export Value ($):** Total value of global exports for that product in 2021. "
        "The threshold (default: 15th percentile) removes micro-traded products where market "
        "entry would be difficult to assess. Products below the threshold represent niche or "
        "regionally traded goods unlikely to generate significant industrial investment."
    )

    st.markdown("### How to set thresholds")
    st.markdown(
        "The **KPI row** at the top of the Filtering page shows how many products survive each "
        "threshold combination in real time. A typical working pool is 400–600 products. "
        "Too few (< 200) risks missing important candidates; too many (> 800) makes scoring less "
        "discriminating. The **threshold distribution charts** (bottom of the page) show where your "
        "cutoff sits relative to the full universe."
    )

st.divider()

# ============================================================
# STAGE 2: LIKELIHOOD
# ============================================================
with st.expander("⚡  Stage 2 — Likelihood & Prioritization: variables and controls", expanded=True):

    st.markdown("### Part A — Likelihood scoring")
    st.markdown(
        "Each product in the filtered pool receives a **likelihood score** (0–100) reflecting how "
        "likely its industry is to relocate production to a renewable-energy-rich location. "
        "The score is a weighted average of five component percentile ranks. "
        "You control the weights — set a weight to 0 to exclude that component entirely."
    )

    st.markdown("#### Likelihood components")

    rows = [
        ("Fuel Energy Intensity", "MJ of fuel energy per $ of output. Industries with high fuel intensity have the most to gain by switching to locations with cheap clean energy. *Source: USEEIO via NAICS–HS concordance.*"),
        ("Electricity Intensity", "MJ of electricity per $ of output. The most direct signal of electricity cost sensitivity — industries where electricity is a significant share of production costs. *Source: USEEIO.*"),
        ("Incumbent Vulnerability", "How energy-deficit are the current top exporters of this product? A country is 'energy-deficit' if it imports most of its energy. If the top exporters of a product are highly energy-deficit, they are vulnerable to being undercut by low-cost-electricity producers. *Higher score = more disruptable incumbents.* Source: computed from IEA energy balance data."),
        ("CBAM Exposure", "Coverage under the EU Carbon Border Adjustment Mechanism (CBAM). CBAM imposes a carbon cost on imports of iron/steel, aluminium, cement, fertilisers, electricity, and hydrogen into the EU. Products covered by CBAM and with high EU market exposure face the strongest regulatory push to relocate to low-carbon production sites. *Source: EU CBAM regulation + UN Comtrade EU import data.*"),
        ("Market Growth", "Export CAGR 2012–2024. Growing markets offer better entry opportunities and signal future relevance. Can be used to tilt the analysis toward dynamic sectors rather than established commodities."),
    ]

    for name, desc in rows:
        st.markdown(f"**{name}:** {desc}")

    st.markdown("#### Selection cutoff")
    st.markdown(
        "After scoring, you apply a cutoff to select the products that 'pass' the likelihood screen:\n\n"
        "- **Top %:** Keep the top X% by likelihood score (e.g. Top 50% keeps the 50% of products with the highest likelihood scores).\n"
        "- **Top N:** Keep exactly the N highest-scoring products.\n\n"
        "Products below the cutoff are excluded from the Feasibility vs. Attractiveness scatter and the output list. "
        "The **scatter title shows the exact count** of products that passed the cutoff."
    )

    st.divider()

    st.markdown("### Part B — Feasibility & Attractiveness scoring")
    st.markdown(
        "Products passing the likelihood cutoff are scored on two dimensions that together form the "
        "**composite score**. You control the balance between them (default: 60% feasibility, 40% attractiveness) "
        "and the weights within each dimension."
    )

    col_f, col_a = st.columns(2)

    with col_f:
        st.markdown("#### Feasibility — Morocco's readiness")
        st.markdown("*How prepared is Morocco to produce and export this product?*")
        st.markdown(
            "**Morocco RCA (Revealed Comparative Advantage):** RCA > 1 means Morocco already has a "
            "competitive export position in this product. Higher RCA = stronger existing base to build on. "
            "Set this weight high if you believe existing capabilities are essential.\n\n"
            "**Morocco Density:** How proximate is this product to Morocco's current export basket in the "
            "product space? High density = many of the capabilities needed for this product already exist in "
            "Morocco's economy. Low density = would require building many new capabilities from scratch. "
            "*This is typically the most important feasibility signal.*\n\n"
            "**Market Fragmentation (1/HHI):** HHI is the Herfindahl-Hirschman Index measuring global "
            "market concentration (0 = perfectly fragmented, 1 = monopoly). 1/HHI = fragmentation. "
            "Higher fragmentation = market is not dominated by a few players = easier for a new entrant "
            "like Morocco to gain share.\n\n"
            "**Avg. Trade Distance:** Average distance in km between producers and consumers for this "
            "product, weighted by trade flows. Products traded over shorter distances are more "
            "transport-cost-sensitive. Morocco's advantage is proximity to Europe — shorter-distance "
            "products benefit more from Morocco's geographic position."
        )

    with col_a:
        st.markdown("#### Attractiveness — Market opportunity")
        st.markdown("*How valuable is the prize if Morocco successfully enters this market?*")
        st.markdown(
            "**Market Size:** Total global export value ($). Larger markets offer greater absolute revenue "
            "potential and more resilience to market share shifts.\n\n"
            "**Market Growth:** Export CAGR 2012–2024. Fast-growing markets offer better entry conditions "
            "and compound returns to early movers.\n\n"
            "**COG — Complexity Outlook Gain:** How much does entering this product improve Morocco's "
            "future diversification options? COG measures how many new, high-complexity products would become "
            "accessible via the product space once Morocco produces this one. High COG = strategic 'gateway' "
            "product that unlocks further industrial development.\n\n"
            "**Product Complexity (PCI):** The Product Complexity Index measures the knowledge intensity "
            "of a product based on how many countries can make it and the diversity of those countries' "
            "export baskets. Higher PCI = more sophisticated manufacturing = higher value added.\n\n"
            "**Spillover Potential:** Network centrality of the industry's NAICS sector. Central industries "
            "generate more knowledge spillovers to related industries — investing in them benefits a broader "
            "range of adjacent sectors."
        )

    st.divider()

    st.markdown("### Reading the Feasibility vs. Attractiveness scatter")
    st.markdown(
        "The scatter plot is the main output of Stage 2. Each bubble is one product that passed "
        "the likelihood cutoff.\n\n"
        "- **X-axis:** Feasibility score (0–100). Further right = Morocco is more ready.\n"
        "- **Y-axis:** Attractiveness score (0–100). Higher = more valuable market opportunity.\n"
        "- **Bubble size:** Global trade volume. Larger bubble = more globally traded product.\n"
        "- **Red bubbles:** Top N products by composite score — the recommended shortlist.\n"
        "- **Grey bubbles:** Passed the likelihood cutoff but not in the top N.\n"
        "- **Dashed lines:** Medians of the selected pool. Upper-right quadrant = above median on both dimensions.\n\n"
        "**Key interpretation:** The ideal candidates sit in the **upper-right** — high feasibility *and* "
        "high attractiveness. Products in the upper-left (high attractiveness, low feasibility) may be "
        "longer-term targets requiring capability building. Products in the lower-right (high feasibility, "
        "low attractiveness) may be easier entry points but with limited strategic value."
    )

    st.markdown("### Composite score")
    st.markdown(
        "The composite score combines feasibility and attractiveness according to the F/A balance slider:\n\n"
        "> **Composite = (Feasibility weight × Feasibility score) + (Attractiveness weight × Attractiveness score)**\n\n"
        "All scores are 0–100 percentile ranks within the selected pool. The composite score determines "
        "the ranking in the output table and which products are highlighted red in the scatter. "
        "The **Top N to Prioritize** control sets how many products are highlighted."
    )

st.divider()

# ============================================================
# STAGE 3: COMPARISON
# ============================================================
with st.expander("🔀  Stage 3 — Comparison: how to use it", expanded=False):
    st.markdown(
        "The Comparison page lets you compare two saved shortlists side by side. To use it:\n\n"
        "1. On the **Likelihood & Prioritization** page, adjust weights and cutoff to define a scenario, "
        "give it a name, and click **Save current scenario**.\n"
        "2. Adjust the weights again to create a different scenario and save it with a different name.\n"
        "3. Navigate to **Comparison** and select the two scenarios to compare.\n\n"
        "The page shows:\n"
        "- **Treemaps:** Industry composition of each shortlist by HS2 chapter.\n"
        "- **Bar charts:** Top industries by any variable you choose.\n"
        "- **Overlap table:** Which products appear in both shortlists vs. only one — "
        "products in both are robust to the weighting assumption.\n\n"
        "**Tip:** Save one scenario with high CBAM/vulnerability weights (regulatory-driven) and "
        "one with high electricity intensity weights (cost-driven). The overlap between them — "
        "products that appear under both theories — are the strongest candidates."
    )

st.divider()

# ============================================================
# DATA SOURCES
# ============================================================
with st.expander("📦  Data sources", expanded=False):
    st.markdown(
        "| Variable group | Source |\n"
        "|---|---|\n"
        "| Trade values, export shares, RCA, HHI, CAGR | UN Comtrade via Growth Lab Atlas of Economic Complexity (2021, 2012–2024) |\n"
        "| Energy intensities (fuel, electricity, carriers) | US EPA USEEIO model, mapped to HS6 via NAICS–HS concordance |\n"
        "| Product Complexity (PCI), Density, COG | Harvard Growth Lab Atlas of Economic Complexity |\n"
        "| Spillover / centrality | NAICS industry network from Growth Lab |\n"
        "| CBAM coverage | EU Regulation 2023/956; product mapping by Growth Lab |\n"
        "| Incumbent vulnerability | Computed from IEA energy balance + UN Comtrade exporter shares |\n"
        "| Trade distance | CEPII GeoDist + UN Comtrade bilateral flows |\n\n"
        "The master dataset covers **5,204 HS6 products** across **97 HS2 chapters**, representing "
        "**~$20 trillion** in global trade. Energy intensity data covers 84% of products "
        "(4,393/5,204); the remainder receives the pool median."
    )

st.divider()
st.info("👈  Start with **Filtering** in the sidebar to define your candidate universe.")
