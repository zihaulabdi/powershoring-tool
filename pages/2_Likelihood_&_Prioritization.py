"""Stage 2: Likelihood & Prioritization — Score and rank products."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_treemap, get_hs2_color_map,
    download_csv, format_dollars, percentile_rank, weighted_score,
    aggregate_to_hs4,
)

st.set_page_config(page_title="Likelihood & Prioritization", layout="wide")
st.title("Likelihood & Prioritization")
st.markdown(
    "Adjust the weights to define which factors most likely drive powershoring, "
    "then rank surviving products by Morocco's readiness (feasibility) and "
    "market opportunity (attractiveness)."
)

# ============================================================
# LOAD DATA
# ============================================================
df_all = load_data()
hs2_color_map = get_hs2_color_map()

if st.session_state.get("filtered_products") is not None:
    df = st.session_state.filtered_products.copy()
    st.success(f"Using **{len(df):,}** filtered products from Filtering.")
else:
    df = df_all.copy()
    st.warning("No filtering applied — using full product universe. Run **Filtering** first.")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Likelihood Weights")
    st.markdown("Weights are normalized to 100%.")

    w_fuel = st.slider("Fuel Energy Intensity", 0, 100, 25, key="w_fuel",
                       help="Products with high fuel energy intensity score higher")
    w_elec = st.slider("Electricity Intensity", 0, 100, 25, key="w_elec",
                       help="Products with high electricity usage per $ score higher")
    w_vuln = st.slider("Incumbent Vulnerability", 0, 100, 25, key="w_vuln",
                       help="Products where current exporters are in energy-deficit countries score higher")
    w_cbam = st.slider("CBAM Exposure", 0, 100, 25, key="w_cbam",
                       help="CBAM-covered products with high EU trade exposure score higher")
    w_growth = st.slider("Market Growth", 0, 100, 0, key="w_growth",
                         help="Products with fast-growing global export markets score higher")

    total_w = w_fuel + w_elec + w_vuln + w_cbam + w_growth
    if total_w == 0:
        st.error("At least one weight must be > 0")
        st.stop()

    st.caption(
        f"**Effective:** Fuel {w_fuel/total_w*100:.0f}% | "
        f"Elec {w_elec/total_w*100:.0f}% | "
        f"Vuln {w_vuln/total_w*100:.0f}% | "
        f"CBAM {w_cbam/total_w*100:.0f}% | "
        f"Growth {w_growth/total_w*100:.0f}%"
    )

    st.divider()
    st.header("Selection Cutoff")
    selection_method = st.radio("Method", ["Top %", "Top N"])
    if selection_method == "Top %":
        top_pct = st.slider("Top % of products", 10, 100, 50, 5)
    else:
        top_n_cut = st.number_input("Top N products", 10, len(df), min(200, len(df)), 10)

    st.divider()
    st.header("Feasibility vs Attractiveness")
    feas_pct = st.slider("Feasibility weight (%)", 0, 100, 60, 5, key="fa_balance",
                         help="Remaining weight goes to attractiveness")
    attr_pct = 100 - feas_pct
    st.caption(f"**{feas_pct}% Feasibility / {attr_pct}% Attractiveness**")

    with st.expander("Feasibility weights"):
        f_rca = st.slider("Morocco RCA", 0, 100, 20, key="f_rca",
                          help="Revealed comparative advantage — existing export strength")
        f_density = st.slider("Morocco Density", 0, 100, 50, key="f_density",
                              help="Proximity to existing capabilities in product space")
        f_hhi = st.slider("Market fragmentation (1/HHI)", 0, 100, 15, key="f_hhi",
                          help="Lower market concentration = easier to enter")
        f_dist = st.slider("Avg. Trade Distance", 0, 100, 15, key="f_dist",
                           help="Short-distance products are more transport-sensitive — better powershoring candidates")

    with st.expander("Attractiveness weights"):
        a_market = st.slider("Market size", 0, 100, 15, key="a_market",
                             help="Global export value")
        a_growth_attr = st.slider("Market growth", 0, 100, 15, key="a_growth_attr",
                                  help="Export CAGR 2012-2024")
        a_cog = st.slider("COG (opportunity gain)", 0, 100, 30, key="a_cog",
                          help="Contribution to future diversification")
        a_pci = st.slider("Product complexity", 0, 100, 30, key="a_pci",
                          help="Higher complexity = more value added")
        a_spillover = st.slider("Spillover potential", 0, 100, 10, key="a_spillover",
                                help="Network centrality — benefits to related industries")

    st.divider()
    st.header("Top N to Prioritize")
    topn_method = st.radio("Select by", ["Top %", "Top N"], key="topn_method",
                           help="Defines which products are highlighted in the scatter and shown in the output list")
    if topn_method == "Top %":
        top_n_pct = st.slider("Top % by composite score", 5, 50, 20, 5, key="topn_pct")
    else:
        top_n_highlight = st.slider("Top N by composite score", 10, 100, 30, 5, key="topn_highlight")

    st.divider()
    st.header("Save Scenario")
    scenario_name = st.text_input("Scenario name", "", key="lp_scenario_name")
    if st.button("Save current scenario") and scenario_name:
        st.session_state["_pending_scenario_save"] = scenario_name


# ============================================================
# COMPUTE LIKELIHOOD SCORES
# ============================================================
likelihood_weights = {
    "fuel_pctile": w_fuel,
    "elec_pctile": w_elec,
    "vulnerability_pctile": w_vuln,
    "cbam_pctile": w_cbam,
    "growth_pctile": w_growth,
}

lhood_comps = {}
lhood_comps["fuel_pctile"] = percentile_rank(df["amount_fuel_energy"].fillna(0))
lhood_comps["elec_pctile"] = percentile_rank(df["amount_electric_energy"].fillna(0))
lhood_comps["vulnerability_pctile"] = (
    percentile_rank(-df["vulnerability_score"].fillna(0))
    if "vulnerability_score" in df.columns
    else pd.Series(50, index=df.index)
)
lhood_comps["cbam_pctile"] = (
    percentile_rank(df["cbam_score"].fillna(0))
    if "cbam_score" in df.columns
    else pd.Series(50, index=df.index)
)
lhood_comps["growth_pctile"] = (
    percentile_rank(df["export_cagr_2012_2023"].fillna(0))
    if "export_cagr_2012_2023" in df.columns
    else pd.Series(50, index=df.index)
)

df["likelihood_score"] = weighted_score(df, lhood_comps, likelihood_weights)
for k, v in lhood_comps.items():
    df[k] = v

# Apply selection cutoff
if selection_method == "Top %":
    cutoff = df["likelihood_score"].quantile(1 - top_pct / 100)
    selected = df[df["likelihood_score"] >= cutoff]
else:
    selected = df.nlargest(top_n_cut, "likelihood_score")
    cutoff = selected["likelihood_score"].min()

st.session_state.likelihood_products = selected


# ============================================================
# COMPUTE FEASIBILITY / ATTRACTIVENESS / COMPOSITE (all products)
# ============================================================
feas_weights = {"rca_pctile": f_rca, "density_pctile": f_density,
                "hhi_pctile": f_hhi, "distance_pctile": f_dist}
feas_comps = {}
feas_comps["rca_pctile"] = (
    percentile_rank(df["rca_mar"].fillna(0)) if "rca_mar" in df.columns
    else pd.Series(50, index=df.index)
)
feas_comps["density_pctile"] = (
    percentile_rank(df["density_mar"].fillna(0)) if "density_mar" in df.columns
    else pd.Series(50, index=df.index)
)
feas_comps["hhi_pctile"] = (
    percentile_rank(df["inv_hhi"].fillna(0)) if "inv_hhi" in df.columns
    else pd.Series(50, index=df.index)
)
feas_comps["distance_pctile"] = (
    percentile_rank(-df["product_distance"].fillna(df["product_distance"].median()))
    if "product_distance" in df.columns
    else pd.Series(50, index=df.index)
)

df["feasibility_score"] = weighted_score(df, feas_comps, feas_weights)

attr_weights = {"market_size_pctile": a_market, "growth_pctile_attr": a_growth_attr,
                "cog_pctile": a_cog, "pci_pctile": a_pci, "spillover_pctile": a_spillover}
attr_comps = {}
attr_comps["market_size_pctile"] = (
    percentile_rank(df["global_export_value"].fillna(0)) if "global_export_value" in df.columns
    else pd.Series(50, index=df.index)
)
attr_comps["growth_pctile_attr"] = (
    percentile_rank(df["export_cagr_2012_2023"].fillna(0)) if "export_cagr_2012_2023" in df.columns
    else pd.Series(50, index=df.index)
)
attr_comps["cog_pctile"] = (
    percentile_rank(df["cog_mar"].fillna(0)) if "cog_mar" in df.columns
    else pd.Series(50, index=df.index)
)
attr_comps["pci_pctile"] = (
    percentile_rank(df["pci"].fillna(0)) if "pci" in df.columns
    else pd.Series(50, index=df.index)
)
attr_comps["spillover_pctile"] = (
    percentile_rank(df["weighted_degree"].fillna(df["weighted_degree"].median()))
    if "weighted_degree" in df.columns
    else pd.Series(50, index=df.index)
)

df["attractiveness_score"] = weighted_score(df, attr_comps, attr_weights)
df["composite_score"] = (
    (feas_pct / 100) * df["feasibility_score"] +
    (attr_pct / 100) * df["attractiveness_score"]
)

for k, v in feas_comps.items():
    df[k] = v
for k, v in attr_comps.items():
    df[k] = v

st.session_state.prioritized_products = df

# Resolve top_n_highlight from whichever method was selected
if topn_method == "Top %":
    top_n_highlight = max(1, int(len(df) * top_n_pct / 100))

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
        "desc": f"Weights: {_weight_desc} | Cutoff: {cutoff:.1f} | {len(selected)} products",
    }
    st.success(f"Saved scenario: **{_sname}** ({len(selected)} products)")


# ============================================================
# SECTION 1 — LIKELIHOOD TREEMAP
# ============================================================
cutoff_label = (
    f"Top {top_pct}% by Likelihood" if selection_method == "Top %"
    else f"Top {top_n_cut} by Likelihood"
)
st.markdown(f"### Industry Composition — {cutoff_label}")
st.markdown(
    f"**{len(selected):,} products** pass the likelihood cutoff "
    f"(score ≥ {cutoff:.1f}). "
    "Adjust likelihood weights to see how industry composition changes."
)

lhood_size_by = st.radio(
    "Size tiles by:", ["Number of products", "Global export value"],
    horizontal=True, key="lhood_size_radio",
)
fig_lhood_tm = make_treemap(
    selected,
    value_col="global_export_value",
    title="",
    color_map=hs2_color_map,
    size_by="count" if lhood_size_by == "Number of products" else "value",
)
st.plotly_chart(fig_lhood_tm, use_container_width=True)


# ============================================================
# SECTION 2 — SCATTER: FEASIBILITY VS ATTRACTIVENESS
# ============================================================
st.markdown("### Feasibility vs. Attractiveness")

# Color: top N red, rest grey
top_codes = set(df.nlargest(top_n_highlight, "composite_score").index)
df["_color"] = df.index.map(lambda i: f"Top {top_n_highlight}" if i in top_codes else "Other")

# Truncated description for hover
df["_desc_short"] = df["description"].fillna("").str[:60]

fig_sc = px.scatter(
    df,
    x="feasibility_score",
    y="attractiveness_score",
    size=df["global_export_value"].clip(lower=0).fillna(0) if "global_export_value" in df.columns else None,
    size_max=28,
    color="_color",
    color_discrete_map={f"Top {top_n_highlight}": MOROCCO_RED, "Other": GREY},
    category_orders={"_color": ["Other", f"Top {top_n_highlight}"]},
    custom_data=["hs_product_code", "_desc_short", "hs2_name", "composite_score", "global_export_value"],
)

# Median reference lines
feas_med = df["feasibility_score"].median()
attr_med = df["attractiveness_score"].median()
fig_sc.add_vline(x=feas_med, line_dash="dash", line_color="#bbbbbb", line_width=1)
fig_sc.add_hline(y=attr_med, line_dash="dash", line_color="#bbbbbb", line_width=1)

fig_sc.update_traces(
    hovertemplate=(
        "<b>HS %{customdata[0]} — %{customdata[1]}</b><br>"
        "Chapter: %{customdata[2]}<br>"
        "Feasibility: %{x:.1f}  |  Attractiveness: %{y:.1f}<br>"
        "Composite: %{customdata[3]:.1f}<br>"
        "Trade: $%{customdata[4]:,.0f}"
        "<extra></extra>"
    )
)

fig_sc.update_layout(
    xaxis=dict(title="Feasibility Score", showgrid=False, showline=False, zeroline=False),
    yaxis=dict(title="Attractiveness Score", showgrid=False, showline=False, zeroline=False),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=520,
    legend=dict(
        orientation="h", yanchor="top", y=-0.12,
        xanchor="center", x=0.5,
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        title_text="",
    ),
    margin=dict(t=10, b=100, l=10, r=10),
    font=dict(family="Source Sans Pro, sans-serif", size=13),
)
st.plotly_chart(fig_sc, use_container_width=True)


# ============================================================
# SECTION 3 — IDENTIFIED PRODUCTS / INDUSTRIES
# ============================================================
st.markdown("---")
st.markdown("### Identified Products/Industries")

top_n_df = df.nlargest(top_n_highlight, "composite_score").copy()

agg_level = st.radio(
    "Aggregate by:",
    ["HS2 (chapter)", "HS4 (subheading)", "HS6 (product)"],
    horizontal=True,
    key="identified_agg",
)

id_size_by = st.radio(
    "Treemap size by:",
    ["Number of products", "Global export value"],
    horizontal=True,
    key="id_size_radio",
)

tab_treemap, tab_table = st.tabs(["Treemap", "Table"])

with tab_treemap:
    if agg_level == "HS2 (chapter)":
        fig_id_tm = make_treemap(
            top_n_df,
            value_col="global_export_value",
            title=f"Top {top_n_highlight} Products — HS2 Industry Composition",
            color_map=hs2_color_map,
            size_by="count" if id_size_by == "Number of products" else "value",
        )
        st.plotly_chart(fig_id_tm, use_container_width=True)

    elif agg_level == "HS4 (subheading)":
        # Aggregate to HS4 using trade-weighted mean scores
        hs4_df = aggregate_to_hs4(top_n_df)
        # Build HS4 treemap grouped by HS2
        size_col = "n_products" if id_size_by == "Number of products" else "global_export_value"
        fig_hs4 = px.treemap(
            hs4_df,
            path=["hs2_name", "description"],
            values=size_col,
            color="hs2_name",
            color_discrete_map=hs2_color_map,
            custom_data=["hs4_code", "n_products"],
        )
        fig_hs4.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "HS4: %{customdata[0]}<br>"
                "Products: %{customdata[1]}"
                "<extra></extra>"
            ),
            textinfo="label+percent root",
        )
        fig_hs4.update_layout(
            template=GL_TEMPLATE, height=520,
            margin=dict(t=30, l=10, r=10, b=10),
        )
        st.plotly_chart(fig_hs4, use_container_width=True)

    else:  # HS6
        size_col = "n_products_dummy"
        top_n_df["n_products_dummy"] = 1  # equal-area tiles at HS6 level
        plot_col = "n_products_dummy" if id_size_by == "Number of products" else "global_export_value"
        top_n_df["_desc_label"] = (
            top_n_df["hs_product_code"].astype(str) + " — " +
            top_n_df["description"].fillna("").str[:50]
        )
        fig_hs6 = px.treemap(
            top_n_df,
            path=["hs2_name", "_desc_label"],
            values=plot_col,
            color="hs2_name",
            color_discrete_map=hs2_color_map,
            custom_data=["hs_product_code", "composite_score", "feasibility_score", "attractiveness_score"],
        )
        fig_hs6.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "HS6: %{customdata[0]}<br>"
                "Composite: %{customdata[1]:.1f}<br>"
                "Feasibility: %{customdata[2]:.1f}  |  Attractiveness: %{customdata[3]:.1f}"
                "<extra></extra>"
            ),
            textinfo="label",
        )
        fig_hs6.update_layout(
            template=GL_TEMPLATE, height=550,
            margin=dict(t=30, l=10, r=10, b=10),
        )
        st.plotly_chart(fig_hs6, use_container_width=True)

with tab_table:
    if agg_level == "HS6 (product)":
        display_cols = [
            "hs_product_code", "description", "hs2_name",
            "composite_score", "feasibility_score", "attractiveness_score",
            "likelihood_score", "global_export_value", "rca_mar", "pci",
            "cbam_flag", "cbam_category",
        ]
        display_cols = [c for c in display_cols if c in top_n_df.columns]
        ranked = (
            top_n_df[display_cols]
            .sort_values("composite_score", ascending=False)
            .reset_index(drop=True)
        )
        ranked.index = ranked.index + 1
        ranked.index.name = "Rank"
        st.dataframe(ranked, use_container_width=True, height=500)
        download_csv(top_n_df, "powershoring_top_products.csv",
                     f"Top {top_n_highlight} by composite ({feas_pct}F/{attr_pct}A)")

    elif agg_level == "HS4 (subheading)":
        hs4_table = aggregate_to_hs4(top_n_df)
        show_cols = [c for c in [
            "hs4_code", "description", "hs2_name", "n_products",
            "composite_score", "feasibility_score", "attractiveness_score",
            "global_export_value", "rca_mar",
        ] if c in hs4_table.columns]
        st.dataframe(
            hs4_table[show_cols].sort_values("composite_score", ascending=False).reset_index(drop=True),
            use_container_width=True, height=500,
        )
        download_csv(hs4_table, "powershoring_top_hs4.csv",
                     f"Top {top_n_highlight} products aggregated to HS4")

    else:  # HS2
        hs2_table = (
            top_n_df.groupby(["hs2_code", "hs2_name"])
            .agg(
                n_products=("hs_product_code", "count"),
                global_export_value=("global_export_value", "sum"),
                composite_score=("composite_score", "mean"),
                feasibility_score=("feasibility_score", "mean"),
                attractiveness_score=("attractiveness_score", "mean"),
            )
            .reset_index()
            .sort_values("composite_score", ascending=False)
        )
        st.dataframe(hs2_table, use_container_width=True, height=400)
        download_csv(hs2_table, "powershoring_top_hs2.csv",
                     f"Top {top_n_highlight} products aggregated to HS2")
