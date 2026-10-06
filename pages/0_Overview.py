"""Overview: what the tool does, how the funnel works, and where to start."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui  # noqa: E402
import utils as U  # noqa: E402

# ------------------------------------------------------------
# Live numbers for the default run (cached)
# ------------------------------------------------------------
with st.spinner("Loading data..."):
    df_all = U.load_data()
    default_s1 = dict(ui.STAGE1_DEFAULTS)
    pool, _ = ui.candidate_pool(default_s1)
    results = ui.scenario_results(stage1=default_s1)
    robust = U.robustness_table(results, top_n=30, level="HS4")
n_universe = len(df_all)
n_pool = len(pool)
n_robust = int((robust["n_scenarios"] >= U.ROBUST_MIN_SCENARIOS).sum()) if len(robust) else 0
n_shortlisted = len(robust)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
ui.page_header(
    "Morocco powershoring · Industry targeting tool",
    "Which energy-intensive industries could Morocco attract?",
    lede=(
        "Cheap renewable electricity is hard to transport, so it gives the places that have it "
        "an advantage in energy-intensive production. This tool screens every globally traded "
        "product to find the industries most likely to relocate toward clean power, and that "
        "Morocco is best placed to win."
    ),
    byline="Harvard Growth Lab · <span>UM6P Economic Complexity Unit</span>",
)

c1, c2, _ = st.columns([1, 1.4, 2])
with c1:
    st.page_link("pages/5_Summary.py", label="See the shortlist  →")
with c2:
    st.page_link("pages/1_Filtering.py", label="Run the analysis step by step  →")

# ------------------------------------------------------------
# The funnel
# ------------------------------------------------------------
st.markdown("## How the tool narrows the field")
st.markdown(
    "The analysis is a funnel. Each step removes products for a stated reason, and every "
    "threshold and weight can be changed in the sidebar of the relevant page."
)

steps = [
    ("01", "Candidate pool",
     "Keep manufactured products that use a lot of energy or electricity per dollar of output "
     "and are traded in meaningful volumes.",
     f"{n_universe:,} → {n_pool:,} products"),
    ("02", "Score and rank",
     "Score how strongly each product is pushed to relocate, keep the strongest, then rank them on "
     "Morocco's readiness (feasibility) and the value of the market (attractiveness).",
     "3 scores per product"),
    ("03", "Scenarios",
     "Repeat step 2 under four theories of why industries relocate, and keep the industries that "
     "make the shortlist under more than one theory.",
     f"{n_shortlisted} industries shortlisted"),
    ("04", "Shortlist",
     f"Industries in the Top 30 of at least {U.ROBUST_MIN_SCENARIOS} scenarios. "
     "These do not depend on a single assumption about what drives relocation.",
     f"{n_robust} robust industries"),
]
cols = st.columns(4)
for col, (n, h, b, k) in zip(cols, steps):
    with col:
        st.markdown(
            f"<div class='ecu-step'><div class='n'>{n}</div><div class='h'>{h}</div>"
            f"<div class='b'>{b}</div><div class='k'>{k}</div></div>",
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------
st.markdown("## Four theories of relocation")
st.markdown(
    "No single theory of relocation is clearly right, so the tool runs four and compares them. "
    "An industry that ranks highly under several theories is a safer bet than one that depends on "
    "a single assumption."
)
scen_cols = st.columns(4)
for col, (name, sdef) in zip(scen_cols, U.SCENARIO_DEFS.items()):
    with col:
        st.markdown(f"**{name}**")
        ui.small(sdef["desc"])

# ------------------------------------------------------------
# Where to start
# ------------------------------------------------------------
st.markdown("## Where to start")
a, b = st.columns(2)
with a:
    st.markdown("**If you want the answer.** Open the **Shortlist**. It uses the default assumptions "
                "and lists the robust industries, with the scenarios each one appears in.")
with b:
    st.markdown("**If you want to test the assumptions.** Work through steps 1 to 3. Your settings "
                "carry over from page to page. Save any shortlist and compare it with another on the "
                "**Compare** page.")

# ------------------------------------------------------------
# Glossary and data notes
# ------------------------------------------------------------
st.markdown("## Key terms")
terms = [
    ("Energy intensity", "Fuel plus electricity used per dollar of output (MJ/$), from the US EPA's USEEIO model."),
    ("Incumbent vulnerability", "How energy-poor today's leading exporters are. Energy-poor incumbents are easier to displace."),
    ("CBAM exposure", "Whether the EU carbon border tax covers the product, scaled by energy use and the EU's share of world imports."),
    ("Capability proximity (density)", "How close a product is to Morocco's current exports in the product space."),
    ("Diversification value (COG)", "How much entering the product would open up new related products for Morocco."),
    ("Product complexity (PCI)", "How much productive know-how the product requires."),
    ("Market openness (1/HHI)", "How fragmented world supply is. Fragmented markets are easier to enter."),
    ("Robust", f"In the Top 30 of at least {U.ROBUST_MIN_SCENARIOS} of the four scenarios."),
]
t1, t2 = st.columns(2)
for i, (term, desc) in enumerate(terms):
    with (t1 if i % 2 == 0 else t2):
        st.markdown(f"**{term}.** {desc}")

with st.expander("Data sources and limitations"):
    st.markdown(
        "- **Trade:** Growth Lab country-product exports, HS 2012, 6-digit (2021 base year; growth 2012 to 2024).\n"
        "- **Energy intensities:** USEEIO (US economy, 2017), mapped from NAICS industries to HS products. "
        "US technology is applied to every country, and biomass inflates intensities for paper and wood. "
        f"{int(df_all['amount_carriers'].isna().sum()):,} of {n_universe:,} products have no energy mapping "
        "and drop out once any energy threshold is set.\n"
        "- **Morocco capabilities:** RCA, density and COG from the Atlas of Economic Complexity.\n"
        "- **CBAM coverage:** EU CBAM annex, matched at HS6.\n"
        "- **Scores** are percentile ranks within the candidate pool (0 to 100). They rank products "
        "against each other; they are not absolute measures."
    )
ui.small(f"Methodology version {U.METHODOLOGY_VERSION}.")
