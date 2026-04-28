"""Stage 2: Likelihood Scenario and Prioritization -- Score and rank products."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_treemap, download_csv, format_dollars,
    percentile_rank, weighted_score,
    SCENARIO_DEFS, DEFAULT_FEAS_WEIGHTS, DEFAULT_ATTR_WEIGHTS,
    inject_custom_css, _HS4_DESC_LOOKUP,
    add_feasibility_attractiveness_scores,
)

st.set_page_config(page_title="2. Likelihood and Prioritization", layout="wide")
inject_custom_css()
st.title("Likelihood Scenario and Prioritization")

# ============================================================
# STAGE GATE
# ============================================================
if not st.session_state.get("stage_1_complete"):
    st.error("Complete **Filtering** first to define the candidate universe.")
    if st.button("Go to Filtering"):
        st.switch_page("pages/1_Filtering.py")
    st.stop()

df_all = load_data()
pool = st.session_state.filtered_products.copy()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.divider()

    # --- Likelihood Scenario ---
    st.header("Likelihood Scenario")
    scenario_options = list(SCENARIO_DEFS.keys())
    chosen_scenario = st.radio("Scenario", scenario_options, index=0,
                               help="Each scenario represents a different reason why industries relocate.")
    sdef = SCENARIO_DEFS[chosen_scenario]
    st.caption(sdef["desc"])

    pre_filter = sdef.get("pre_filter")
    if pre_filter == "cbam":
        n_cbam = (pool["cbam_flag"] == 1).sum()
        st.caption(f"*Pre-filtered to CBAM-covered products ({n_cbam} products).*")

    w_fuel   = sdef["weights"].get("fuel", 0)
    w_elec   = sdef["weights"].get("elec", 0)
    w_vuln   = sdef["weights"].get("vuln", 0)
    w_cbam   = sdef["weights"].get("cbam", 0)
    w_growth = sdef["weights"].get("growth", 0)

    st.divider()

    # --- Selection Cutoff ---
    st.header("Selection Cutoff")
    selection_method = st.radio("Method", ["Top %", "Top N", "Score threshold"])
    if selection_method == "Top %":
        top_pct = st.slider("Top %", 10, 100, 50, 5)
    elif selection_method == "Top N":
        top_n_val = st.number_input("Top N", 10, len(pool), min(200, len(pool)), 10)
    else:
        score_thresh = st.slider("Min score", 0.0, 100.0, 50.0, 1.0)

    st.divider()

    # --- Ranking ---
    st.header("Ranking")
    feas_pct = st.slider("Feasibility weight (%)", 0, 100, 60, 5, key="fa_balance")
    attr_pct = 100 - feas_pct
    st.caption(f"**{feas_pct}% Feasibility / {attr_pct}% Attractiveness**")
    top_n_count = st.slider("Top N to highlight", 10, 50, 30, 5, key="topn")

    st.divider()

    with st.expander("Feasibility weights"):
        f_rca     = st.slider("Morocco RCA", 0, 100, DEFAULT_FEAS_WEIGHTS["rca"], key="f_rca")
        f_density = st.slider("Capability proximity (Density)", 0, 100, DEFAULT_FEAS_WEIGHTS["density"], key="f_density")
        f_hhi     = st.slider("Market openness (1/HHI)", 0, 100, DEFAULT_FEAS_WEIGHTS["hhi"], key="f_hhi")
        f_dist    = st.slider("Trade distance", 0, 100, DEFAULT_FEAS_WEIGHTS["distance"], key="f_dist")

    with st.expander("Attractiveness weights"):
        a_market    = st.slider("Market size", 0, 100, DEFAULT_ATTR_WEIGHTS["market_size"], key="a_market")
        a_growth    = st.slider("Market growth", 0, 100, DEFAULT_ATTR_WEIGHTS["growth"], key="a_growth")
        a_cog       = st.slider("Diversification value (COG)", 0, 100, DEFAULT_ATTR_WEIGHTS["cog"], key="a_cog")
        a_pci       = st.slider("Product complexity (PCI)", 0, 100, DEFAULT_ATTR_WEIGHTS["pci"], key="a_pci")
        a_spillover = st.slider("Spillover potential", 0, 100, DEFAULT_ATTR_WEIGHTS["spillover"], key="a_spillover")

    st.divider()
    st.header("Save Scenario")
    scenario_name = st.text_input("Scenario name", "", key="scenario_name_input")
    if st.button("Save current scenario") and scenario_name:
        st.session_state["_pending_scenario_save"] = scenario_name

# ============================================================
# COMPUTE LIKELIHOOD SCORES
# ============================================================
score_df = pool.copy()
if pre_filter == "cbam":
    score_df = score_df[score_df["cbam_flag"] == 1].copy()

total_w = w_fuel + w_elec + w_vuln + w_cbam + w_growth or 1

likelihood_components = {
    "fuel_pctile": percentile_rank(score_df["amount_fuel_energy"].fillna(0)),
    "elec_pctile": percentile_rank(score_df["amount_electric_energy"].fillna(0)),
    "vulnerability_pctile": (
        percentile_rank(-score_df["vulnerability_score"].fillna(0))
        if "vulnerability_score" in score_df.columns
        else pd.Series(50, index=score_df.index)
    ),
    "cbam_pctile": (
        percentile_rank(score_df["cbam_score"].fillna(0))
        if "cbam_score" in score_df.columns
        else pd.Series(50, index=score_df.index)
    ),
    "growth_pctile": (
        percentile_rank(score_df["export_cagr_2012_2023"].fillna(0))
        if "export_cagr_2012_2023" in score_df.columns
        else pd.Series(50, index=score_df.index)
    ),
}

weights_like = {
    "fuel_pctile": w_fuel, "elec_pctile": w_elec,
    "vulnerability_pctile": w_vuln, "cbam_pctile": w_cbam, "growth_pctile": w_growth,
}
score_df["likelihood_score"] = weighted_score(score_df, likelihood_components, weights_like)
for k, v in likelihood_components.items():
    score_df[k] = v

# ============================================================
# COMPUTE FEASIBILITY + ATTRACTIVENESS
# ============================================================
# F/A percentiles are ranked against the full Stage 1 candidate pool, not the
# scenario-specific post-likelihood subset. This keeps scores comparable across
# scenarios: likelihood changes the gate, while F/A measures Morocco readiness
# and market attractiveness against a fixed eligible universe.
feas_weights = {"rca": f_rca, "density": f_density, "hhi": f_hhi, "distance": f_dist}
attr_weights = {
    "market_size": a_market, "growth": a_growth,
    "cog": a_cog, "pci": a_pci, "spillover": a_spillover,
}
score_df = add_feasibility_attractiveness_scores(
    score_df,
    feas_weights,
    attr_weights,
    reference_df=pool,
)

# Selection cutoff (F/A already assigned, so selected inherits stable scores)
if selection_method == "Top %":
    cutoff = score_df["likelihood_score"].quantile(1 - top_pct / 100)
    selected = score_df[score_df["likelihood_score"] >= cutoff].copy()
elif selection_method == "Top N":
    selected = score_df.nlargest(top_n_val, "likelihood_score").copy()
    cutoff = selected["likelihood_score"].min() if len(selected) > 0 else 0.0
else:
    cutoff = score_thresh
    selected = score_df[score_df["likelihood_score"] >= cutoff].copy()

selected["composite_score"] = (
    (feas_pct / 100) * selected["feasibility_score"] +
    (attr_pct / 100) * selected["attractiveness_score"]
)

# Assign quadrants using fixed threshold of 50 on both axes
# (fixed thresholds ensure stable quadrant definitions across saved scenarios)
selected["quadrant"] = selected.apply(
    lambda r: (
        "Top Priorities"   if r["feasibility_score"] >= 50 and r["attractiveness_score"] >= 50
        else "Strategic Bets"    if r["attractiveness_score"] >= 50
        else "Low-Hanging Fruit" if r["feasibility_score"] >= 50
        else "Deprioritize"
    ), axis=1
)

# Add HS4 label for display
selected["hs4_code_str"] = selected["hs_product_code"].astype(str).str.zfill(6).str[:4]
selected["hs4_short"] = selected["hs4_code_str"].map(
    lambda c: (_HS4_DESC_LOOKUP.get(c, str(c))[:40])
)

# Save to session state
st.session_state.likelihood_products  = selected.copy()
st.session_state.prioritized_products = selected.copy()
st.session_state.stage_2_complete     = True

# Handle pending scenario save
if st.session_state.get("_pending_scenario_save"):
    _sname = st.session_state.pop("_pending_scenario_save")
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = {}
    _weight_desc = (
        f"Fuel {w_fuel/total_w*100:.0f}%, Elec {w_elec/total_w*100:.0f}%, "
        f"Vuln {w_vuln/total_w*100:.0f}%, CBAM {w_cbam/total_w*100:.0f}%, "
        f"Growth {w_growth/total_w*100:.0f}%"
    )
    st.session_state.saved_scenarios[_sname] = {
        "products": selected.copy(),
        "stage": "likelihood",
        "desc": f"{chosen_scenario} | Weights: {_weight_desc} | Cutoff: {cutoff:.1f} | {len(selected)} products",
    }
    st.success(f"Saved scenario: **{_sname}** ({len(selected)} products)")

# ============================================================
# KPI ROW
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected Products", f"{len(selected):,}")
c2.metric("Avg Feasibility", f"{selected['feasibility_score'].mean():.1f}")
c3.metric("Avg Attractiveness", f"{selected['attractiveness_score'].mean():.1f}")
c4.metric("Combined Trade", format_dollars(selected["global_export_value"].sum()))

# ============================================================
# TREEMAP
# ============================================================
treemap_choice = st.radio(
    "Size by:", ["Number of products", "Global trade volume"],
    horizontal=True, key="lp_treemap_metric",
)
if treemap_choice == "Number of products":
    agg_tm = selected.groupby(["hs2_code", "hs2_name"]).agg(
        value=("hs_product_code", "count"),
    ).reset_index()
    tm_title = "HS2 Chapter Composition (by number of products)"
else:
    agg_tm = selected.groupby(["hs2_code", "hs2_name"]).agg(
        value=("global_export_value", "sum"),
        n_products=("hs_product_code", "count"),
    ).reset_index()
    tm_title = "HS2 Chapter Composition (by trade volume)"

fig_tm = px.treemap(
    agg_tm, path=["hs2_name"], values="value",
    color="hs2_name",
    color_discrete_sequence=GL_PALETTE_EXT,
    title=tm_title,
    custom_data=["hs2_code"],
)
fig_tm.update_traces(
    hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<extra></extra>",
    textinfo="label+percent root",
)
fig_tm.update_layout(template=GL_TEMPLATE, height=420, margin=dict(t=50, l=10, r=10, b=10))
st.plotly_chart(fig_tm, use_container_width=True)

# ============================================================
# RANKED TABLE (Top N)
# ============================================================
st.markdown(f"### Top {top_n_count} Candidates")
st.caption("Scores are 0-100 percentile ranks. Higher = stronger candidate.")

top_df = selected.nlargest(top_n_count, "composite_score").reset_index(drop=True)
top_df.index = top_df.index + 1
top_df.index.name = "Rank"
top_df["trade_fmt"] = top_df["global_export_value"].apply(format_dollars)

col_config_top = {
    "hs4_code_str": st.column_config.TextColumn("HS4", width="small"),
    "hs4_short":    st.column_config.TextColumn("Industry", width="large"),
    "hs2_name":     st.column_config.TextColumn("Sector", width="medium"),
    "composite_score": st.column_config.ProgressColumn(
        f"Composite ({feas_pct}F/{attr_pct}A)", min_value=0, max_value=100, format="%.0f"
    ),
    "feasibility_score": st.column_config.ProgressColumn(
        "Feasibility", min_value=0, max_value=100, format="%.0f"
    ),
    "attractiveness_score": st.column_config.ProgressColumn(
        "Attractiveness", min_value=0, max_value=100, format="%.0f"
    ),
    "quadrant": st.column_config.TextColumn("Quadrant"),
    "trade_fmt": st.column_config.TextColumn("Global Trade"),
}
show_top_cols = [c for c in [
    "hs4_code_str", "hs4_short", "hs2_name",
    "composite_score", "feasibility_score", "attractiveness_score",
    "quadrant", "trade_fmt",
] if c in top_df.columns]

st.dataframe(top_df[show_top_cols], column_config=col_config_top,
             use_container_width=True, height=500)

# ============================================================
# FEASIBILITY VS ATTRACTIVENESS SCATTER
# ============================================================
st.divider()
st.markdown(f"### Feasibility vs. Attractiveness — {len(selected):,} products ({chosen_scenario} scenario, cutoff={cutoff:.1f})")

selected_idx = set(selected.index)
top_idx      = set(selected.nlargest(top_n_count, "composite_score").index)

# Show ALL products in the scored pool so the scatter visually responds to the
# selection cutoff and scenario — selected products are highlighted in red,
# unselected (below cutoff) in grey.
plot_df = score_df.copy()
plot_df["_color"] = plot_df.index.map(
    lambda i: f"Top {top_n_count}" if i in top_idx
              else "Selected" if i in selected_idx
              else "Below cutoff"
)
plot_df["_size"]      = plot_df["global_export_value"].clip(lower=1).apply(np.log10)
plot_df["_cbam_str"]  = plot_df["cbam_flag"].map({1: "Yes", 0: "No"}).fillna("No")
plot_df["_trade_str"] = plot_df["global_export_value"].apply(format_dollars)
plot_df["_hs6_str"]   = plot_df["hs_product_code"].astype(str).str.zfill(6)
plot_df["_hs4_short"] = plot_df["hs_product_code"].astype(str).str.zfill(6).str[:4].map(
    lambda c: (_HS4_DESC_LOOKUP.get(c, str(c))[:40])
)

fig_sc = px.scatter(
    plot_df,
    x="feasibility_score",
    y="attractiveness_score",
    size="_size",
    size_max=20,
    color="_color",
    color_discrete_map={
        f"Top {top_n_count}": MOROCCO_RED,
        "Selected": "#204B82",
        "Below cutoff": GREY,
    },
    category_orders={"_color": ["Below cutoff", "Selected", f"Top {top_n_count}"]},
    custom_data=["_hs4_short", "_hs6_str", "likelihood_score", "_trade_str", "_cbam_str"],
)
fig_sc.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "HS6: %{customdata[1]}<br>"
        "Likelihood: %{customdata[2]:.0f}<br>"
        "Trade: %{customdata[3]}<br>"
        "CBAM: %{customdata[4]}<extra></extra>"
    )
)
# Fixed quadrant lines at 50
fig_sc.add_vline(x=50, line_dash="dash", line_color="#204B82", opacity=0.4)
fig_sc.add_hline(y=50, line_dash="dash", line_color="#204B82", opacity=0.4)
fig_sc.update_layout(
    xaxis_title="Feasibility Score (Morocco readiness)",
    yaxis_title="Attractiveness Score (market opportunity)",
    template=GL_TEMPLATE,
    height=550,
    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    margin=dict(b=100),
)
st.plotly_chart(fig_sc, use_container_width=True)

# Tabs below scatter
tab_detail, tab_tm2 = st.tabs(["Detail Table", "Top N Treemap"])

with tab_detail:
    detail_cols = [
        "hs_product_code", "description", "hs4_code_str", "hs4_short", "hs2_name",
        "composite_score", "feasibility_score", "attractiveness_score", "quadrant",
        "rca_pctile", "density_pctile", "hhi_pctile", "distance_pctile",
        "market_size_pctile", "attr_growth_pctile", "cog_pctile", "pci_pctile", "spillover_pctile",
    ]
    detail_cols = [c for c in detail_cols if c in selected.columns]
    ranked = selected[detail_cols].sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"
    st.dataframe(ranked, use_container_width=True, height=450)
    download_csv(
        selected, "powershoring_scored_products.csv",
        f"{chosen_scenario} | Cutoff={cutoff:.1f} | {feas_pct}F/{attr_pct}A | {len(selected)} products",
    )

with tab_tm2:
    top_n_prods = selected.nlargest(top_n_count, "composite_score")
    agg_top = top_n_prods.groupby(["hs2_code", "hs2_name"]).agg(
        value=("global_export_value", "sum"),
        n_products=("hs_product_code", "count"),
    ).reset_index()
    fig_tm2 = px.treemap(
        agg_top, path=["hs2_name"], values="value",
        color="hs2_name",
        color_discrete_sequence=GL_PALETTE_EXT,
        title=f"Top {top_n_count} Products by HS2 Chapter",
    )
    fig_tm2.update_traces(textinfo="label+percent root")
    fig_tm2.update_layout(template=GL_TEMPLATE, height=450, margin=dict(t=50, l=10, r=10, b=10))
    st.plotly_chart(fig_tm2, use_container_width=True)

# Stage gate
st.divider()
st.success(f"{len(selected):,} products scored and ranked. View the full cross-scenario analysis in Scenarios.")
if st.button("View Scenario Analysis", type="primary"):
    st.switch_page("pages/3_Scenarios.py")
