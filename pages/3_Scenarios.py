"""Step 3: run all four relocation theories and find the robust candidates."""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui  # noqa: E402
import utils as U  # noqa: E402

ALL = list(U.SCENARIO_DEFS.keys())

# ============================================================
# SIDEBAR
# ============================================================
sb = st.sidebar
sb.header("Theories to include")
active = [s for s in ALL
          if ui.persist(sb.checkbox, s, f"sc_on_{s}", True, help=U.SCENARIO_DEFS[s]["desc"])]
level_label = ui.persist(sb.radio, "Compare at the level of", "sc_level", "Industries (HS4)",
                         options=["Industries (HS4)", "Products (HS6)"])
level = "HS4" if "HS4" in level_label else "HS6"
ranking = ui.ranking_controls()
top_n = ranking["top_n"]

if not active:
    st.warning("Tick at least one theory in the sidebar.")
    st.stop()

# ============================================================
# COMPUTE
# ============================================================
pool, meta = ui.candidate_pool()
with st.spinner("Running the scenarios..."):
    results = ui.scenario_results(ranking=ranking, names=active)
table = U.robustness_table(results, top_n=top_n, level=level)

robust_min = min(U.ROBUST_MIN_SCENARIOS, len(active))
robust = table[table["n_scenarios"] >= robust_min].reset_index(drop=True)
code_word = "industries" if level == "HS4" else "products"

# Pairwise overlap (share of the two shortlists that is shared)
src = "hs4" if level == "HS4" else "selected"
code_col = "hs4_code" if level == "HS4" else "hs_product_code"
tops = {s: set(results[s][src].nlargest(top_n, "composite_score")[code_col].astype(str)) for s in active}
overlap = pd.DataFrame(index=active, columns=active, dtype=float)
for a in active:
    for b in active:
        overlap.loc[a, b] = len(tops[a] & tops[b]) / max(len(tops[a] | tops[b]), 1) * 100
pairs = [overlap.loc[a, b] for i, a in enumerate(active) for b in active[i + 1:]]

# Save for comparison
sb.header("Save for comparison")
to_save = sb.selectbox("Theory", active, key="sc_save_which")
if sb.button("Save its shortlist", type="primary"):
    top_rows = results[to_save]["selected"].nlargest(top_n, "composite_score")
    ui.save_shortlist(f"{to_save} · top {top_n}", top_rows,
                      f"{to_save} theory · {ranking['feas_pct']}F/{100 - ranking['feas_pct']}A · top {top_n} products")
    sb.success("Saved. Open Compare to see it.")

# ============================================================
# HEADER
# ============================================================
ui.page_header(
    "Step 3 of 3 · Scenarios",
    "Which candidates hold up under every theory?",
    lede=(
        f"Each theory produces its own Top {top_n}. An entry that appears in several of them does not "
        "depend on one contested assumption about why industries relocate. "
        f"Here, <b>{len(robust)} {code_word}</b> appear in at least {robust_min} of the "
        f"{len(active)} shortlists."
    ),
)
if not (ui.stage1_is_default() and ui.ranking_is_default(ranking)):
    ui.note("These results use your own settings from step 1 or the sidebar, so they may differ from the "
            "Shortlist page, which always uses the defaults.", accent=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{code_word.capitalize()} shortlisted", f"{len(table)}")
k2.metric(f"Robust ({robust_min}+ theories)", f"{len(robust)}")
k3.metric(f"In all {len(active)}", f"{int((table['n_scenarios'] == len(active)).sum())}")
k4.metric("Average overlap", f"{np.mean(pairs):.0f}%" if pairs else "n/a")

# ============================================================
# TABS
# ============================================================
tab_robust, tab_sector, tab_overlap, tab_table = st.tabs(
    ["Robust candidates", "By sector", "Overlap between theories", "Full table"])

with tab_robust:
    ui.chart_header(
        f"Robust {code_word}",
        f"A filled dot means the entry is in that theory's Top {top_n}. Rows are ordered by number of "
        "theories, then by average composite score (right).",
    )
    if len(robust):
        ui.plot(ui.scenario_dots(robust, active))
    else:
        st.info(f"Nothing appears in {robust_min}+ shortlists. Try a larger Top N.")

with tab_sector:
    ui.chart_header(
        "Shortlists by sector",
        f"Number of {code_word} from each HS2 chapter in each theory's Top {top_n}. "
        "Chapters filled across the whole row are robust at the sector level.",
    )
    rows = []
    for s in active:
        t = results[s][src].nlargest(top_n, "composite_score")
        rows.append(t.groupby("hs2_name").size().rename(s))
    sector = pd.concat(rows, axis=1).fillna(0).astype(int)
    # Order: chapters present in more theories first, then by total count
    order_key = sector.astype(bool).sum(axis=1) * 1000 + sector.sum(axis=1)
    sector = sector.loc[order_key.sort_values(ascending=False).index]
    ui.plot(ui.count_heatmap(sector))

with tab_overlap:
    ui.chart_header(
        "How similar are the shortlists?",
        "Share of entries two shortlists have in common (intersection over union). "
        "Low overlap means the choice of theory matters a lot.",
    )
    ov = overlap.round(0)
    fig = ui.count_heatmap(ov.rename(index=ui.SCENARIO_SHORT))
    fig.update_traces(text=[[f"{v:.0f}%" for v in row] for row in ov.values], zmax=100,
                      hovertemplate="<b>%{y}</b> vs %{x}: %{z:.0f}%<extra></extra>")
    ui.plot(fig)

with tab_table:
    ui.chart_header(f"Every {'industry' if level == 'HS4' else 'product'} in any shortlist",
                    "Scores are averages over the theories where the entry appears.")
    t = table.copy()
    t["description"] = t["description"].map(lambda d: ui.short_label(d, 70))
    t.index = t.index + 1
    t.index.name = "Rank"
    colcfg = {
        "code": st.column_config.TextColumn(level, width="small"),
        "description": st.column_config.TextColumn("Description", width="large"),
        "hs2_name": st.column_config.TextColumn("Sector", width="medium"),
        "n_scenarios": st.column_config.NumberColumn("Theories", format=f"%d of {len(active)}"),
        "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.0f"),
        "feasibility_score": st.column_config.NumberColumn("Feasibility", format="%.0f", width="small"),
        "attractiveness_score": st.column_config.NumberColumn("Attractiveness", format="%.0f", width="small"),
        "global_export_value": st.column_config.NumberColumn("World exports ($)", format="compact"),
    }
    for s in active:
        colcfg[s] = st.column_config.CheckboxColumn(ui.SCENARIO_SHORT[s], width="small")
    st.dataframe(t[["code", "description", "hs2_name", "n_scenarios"] + active +
                   ["composite_score", "feasibility_score", "attractiveness_score", "global_export_value"]],
                 column_config=colcfg, width="stretch", height=560)
    U.download_csv(table, f"powershoring_scenarios_{level.lower()}.csv",
                   f"Theories: {', '.join(active)} | {ranking['feas_pct']}F/{100 - ranking['feas_pct']}A | Top {top_n}")

st.divider()
st.page_link("pages/4_Expert_Comparison.py", label="Compare saved shortlists side by side  →")
