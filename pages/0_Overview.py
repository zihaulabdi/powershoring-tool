"""Introduction: what the tool is, what powershoring is, and how identification works."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui  # noqa: E402
import utils as U  # noqa: E402

df_all = U.load_data()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
ui.page_header(
    "Morocco powershoring",
    "Industry identification tool",
    lede=(
        "This tool ranks energy-intensive products that Morocco could attract by offering cheap "
        "renewable electricity. You set the assumptions, such as energy thresholds and scoring "
        "weights, and the tool shows the resulting ranking. It is useful for checking how the list "
        "of candidate industries changes when the assumptions change."
    ),
    byline="Harvard Growth Lab · <span>UM6P Economic Complexity Unit</span>",
)
st.page_link("pages/1_Filtering.py", label="Start  →")

# ------------------------------------------------------------
# What is powershoring
# ------------------------------------------------------------
st.markdown("## What is powershoring?")
st.markdown(
    "Powershoring is the relocation of energy-intensive industries to countries with cheap, "
    "low-carbon electricity (Ahuja and Hausmann 2025). Fossil fuels are cheap to ship, so countries "
    "without their own fuel could import it. Electricity is expensive to transmit over long distances, "
    "so cheap renewable power mainly benefits producers located near it. Industries where energy is a "
    "large share of costs, such as metals, chemicals, glass and paper, therefore have a reason to move "
    "to countries like Morocco that have large solar and wind resources."
)

# ------------------------------------------------------------
# Identification process
# ------------------------------------------------------------
st.markdown("## Identification process")
st.markdown("The tool works in four steps. Each step has its own page and its own settings.")
steps = [
    ("01", "Candidate pool",
     "Keep manufactured products that use a lot of energy or electricity per dollar of output "
     "and are traded in large enough volumes.",
     "1 · Candidate pool"),
    ("02", "Likelihood of relocation",
     "Score each product on how strongly it is pushed to relocate, using one of four relocation "
     "theories or your own weights.",
     "2 · Score and rank"),
    ("03", "Feasibility and attractiveness",
     "Rank the most likely products on Morocco's readiness to produce them (feasibility) and on "
     "the value of the market (attractiveness).",
     "2 · Score and rank"),
    ("04", "Scenarios",
     "Run all four relocation theories and see which industries rank highly under more than one.",
     "3 · Scenarios"),
]
cols = st.columns(4)
for col, (n, h, b, where) in zip(cols, steps):
    with col:
        st.markdown(
            f"<div class='ecu-step'><div class='n'>{n}</div><div class='h'>{h}</div>"
            f"<div class='b'>{b}</div><div class='k'>Page: {where}</div></div>",
            unsafe_allow_html=True,
        )
st.markdown("")
st.markdown("Any result can be saved and compared with another on the **Compare** page. "
            "Settings carry over from page to page.")

# ------------------------------------------------------------
# Relocation theories
# ------------------------------------------------------------
st.markdown("## The four relocation theories")
st.markdown("Each theory uses a different reason for relocation to score likelihood in step 2.")
scen_cols = st.columns(4)
for col, (name, sdef) in zip(scen_cols, U.SCENARIO_DEFS.items()):
    with col:
        st.markdown(f"**{name}**")
        ui.small(sdef["desc"])

# ------------------------------------------------------------
# Glossary and data notes
# ------------------------------------------------------------
st.markdown("## Key terms")
terms = [
    ("Energy intensity", "Fuel and electricity used per dollar of output (MJ/$)."),
    ("Incumbent vulnerability", "How energy-poor the current leading exporters are."),
    ("CBAM exposure", "Whether the EU carbon border tax covers the product, scaled by energy use and EU import share."),
    ("Capability proximity (density)", "How close a product is to what Morocco already exports."),
    ("Diversification value (COG)", "How many new related products entering this one would open up for Morocco."),
    ("Product complexity (PCI)", "How much know-how the product requires."),
    ("Market openness (1/HHI)", "How spread out world supply is across exporters."),
    ("Score", "A percentile rank within the candidate pool, from 0 to 100."),
]
t1, t2 = st.columns(2)
for i, (term, desc) in enumerate(terms):
    with (t1 if i % 2 == 0 else t2):
        st.markdown(f"**{term}.** {desc}")

with st.expander("Data sources and limitations"):
    st.markdown(
        f"- **Trade:** Growth Lab exports data for {len(df_all):,} HS 2012 six-digit products "
        "(2021 base year; growth 2012 to 2024).\n"
        "- **Energy intensity:** USEEIO (US economy, 2017), mapped from industries to products. "
        "US technology is assumed for all countries, and biomass raises the figures for paper and wood. "
        f"{int(df_all['amount_carriers'].isna().sum()):,} products have no energy data.\n"
        "- **Morocco capabilities:** RCA, density and COG from the Atlas of Economic Complexity.\n"
        "- **CBAM coverage:** EU CBAM annex, matched at HS6.\n"
        "- **Scores** are ranks within the pool you define, so they change when the pool changes."
    )
ui.small(f"Methodology version {U.METHODOLOGY_VERSION}.")
