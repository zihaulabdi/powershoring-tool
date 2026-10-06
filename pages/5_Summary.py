"""Shortlist: robust candidate industries under the default assumptions."""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui  # noqa: E402
import utils as U  # noqa: E402

TOP_N = 30
SCEN = list(U.SCENARIO_DEFS.keys())

# ------------------------------------------------------------
# Compute: four scenarios on the default pool and default weights.
# The shortlist deliberately ignores the user's own settings so that
# everyone who opens this page sees the same answer.
# ------------------------------------------------------------
with st.spinner("Running the four scenarios..."):
    default_s1 = dict(ui.STAGE1_DEFAULTS)
    pool, _ = ui.candidate_pool(default_s1)
    results = ui.scenario_results(stage1=default_s1)
    table = U.robustness_table(results, top_n=TOP_N, level="HS4")

robust = table[table["n_scenarios"] >= U.ROBUST_MIN_SCENARIOS].copy().reset_index(drop=True)
single = table[table["n_scenarios"] == 1].copy()
robust["label"] = robust["description"].map(ui.short_label)
n_all = int((robust["n_scenarios"] == len(SCEN)).sum())
n_three = int((robust["n_scenarios"] == 3).sum())

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
ui.page_header(
    "Result · default assumptions",
    "Robust candidate industries",
    lede=(
        f"<b>{len(robust)} industries</b> make the Top {TOP_N} under at least "
        f"{U.ROBUST_MIN_SCENARIOS} of the four relocation scenarios. "
        f"{n_all} appears under all four and {n_three} under three. "
        "Because they do not depend on a single theory of why industries relocate, "
        "these are the strongest starting points for deeper sector work."
    ),
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Robust industries", f"{len(robust)}")
k2.metric("In 3 or 4 scenarios", f"{n_all + n_three}")
k3.metric("Sectors (HS2)", f"{robust['hs2_name'].nunique()}")
k4.metric("World exports", ui.fmt_usd(robust["global_export_value"].sum()))

# ------------------------------------------------------------
# Chart: which scenarios each industry appears in
# ------------------------------------------------------------
ui.chart_header(
    "Where each robust industry makes the shortlist",
    f"Each row is an HS4 industry. A filled dot means the industry is in that scenario's Top {TOP_N}. "
    "Rows are ordered by number of scenarios, then by average composite score (shown on the right).",
)

ui.plot(ui.scenario_dots(robust, SCEN))

# ------------------------------------------------------------
# Sector view
# ------------------------------------------------------------
left, right = st.columns([1, 1])
with left:
    ui.chart_header("Robust industries by sector", "Number of robust HS4 industries in each HS2 chapter.")
    by_sector = robust.groupby("hs2_name").size().sort_values(ascending=False)
    ui.plot(ui.hbar(by_sector.index, by_sector.values, value_fmt=",.0f"))
with right:
    ui.chart_header("How to read the scores", None)
    st.markdown(
        "- **Composite** = 60% feasibility + 40% attractiveness, averaged over the scenarios "
        "where the industry appears.\n"
        "- **Feasibility** measures Morocco's readiness: capability proximity, existing export "
        "strength, market openness and trade distance.\n"
        "- **Attractiveness** measures the prize: product complexity, diversification value, "
        "market size, growth and spillovers.\n"
        "- All scores are **percentile ranks** within the candidate pool (0 to 100), so 80 means "
        "better than 80% of candidates."
    )

# ------------------------------------------------------------
# Table
# ------------------------------------------------------------
ui.chart_header("Robust industries: full table", "Sort any column by clicking its header.")
tbl = robust.rename(columns={"code": "HS4", "label": "Industry", "hs2_name": "Sector",
                             "n_scenarios": "Scenarios"})
tbl.index = tbl.index + 1
tbl.index.name = "Rank"
show = ["HS4", "Industry", "Sector", "Scenarios"] + SCEN + [
    "composite_score", "feasibility_score", "attractiveness_score", "global_export_value"]
colcfg = {
    "HS4": st.column_config.TextColumn(width="small"),
    "Industry": st.column_config.TextColumn(width="medium"),
    "Sector": st.column_config.TextColumn(width="small"),
    "Scenarios": st.column_config.NumberColumn(format="%d of 4", width="small"),
    "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.0f"),
    "feasibility_score": st.column_config.NumberColumn("Feasibility", format="%.0f", width="small"),
    "attractiveness_score": st.column_config.NumberColumn("Attractiveness", format="%.0f", width="small"),
    "global_export_value": st.column_config.NumberColumn("World exports ($)", format="compact"),
}
for s in SCEN:
    colcfg[s] = st.column_config.CheckboxColumn(ui.SCENARIO_SHORT[s], width="small")
st.dataframe(tbl[show], column_config=colcfg, width="stretch", height=min(38 * len(tbl) + 40, 720))

export = robust.drop(columns=["label"]).rename(columns={"code": "hs4_code"})
U.download_csv(export, "powershoring_robust_shortlist.csv",
               f"Robust = Top {TOP_N} in {U.ROBUST_MIN_SCENARIOS}+ scenarios | default assumptions | "
               f"{U.METHODOLOGY_VERSION}")

# ------------------------------------------------------------
# Scenario-specific candidates
# ------------------------------------------------------------
with st.expander(f"Industries that make only one scenario's shortlist ({len(single)})"):
    st.markdown("These depend on one theory of relocation. They are worth a look if you believe that theory "
                "is the right one, but they are less safe bets.")
    for s in SCEN:
        only = single[single[s]]
        if len(only):
            st.markdown(f"**{s}** ({len(only)}): " + "; ".join(
                f"{ui.short_label(d, 45)} ({c})" for c, d in zip(only["code"], only["description"])))

ui.note(
    f"<b>Assumptions.</b> Candidate pool: {len(pool):,} manufactured products with energy intensity at or above "
    "the 75th percentile or electricity intensity at or above the 50th, and world trade at or above the 15th "
    "percentile. Ranking: default feasibility and attractiveness weights, 60/40. To test other assumptions, "
    "work through steps 1 to 3; this page does not change when you do.",
)
