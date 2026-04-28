"""Stage 1: Filtering -- Define the candidate product universe."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_treemap, make_bar_chart,
    download_csv, format_dollars,
    inject_custom_css, _HS4_DESC_LOOKUP,
    apply_stage1_filter,
)

st.set_page_config(page_title="1. Filtering", layout="wide")
inject_custom_css()
st.title("Filtering")
st.markdown("Define the candidate product universe by setting energy intensity and trade volume thresholds.")

df = load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.divider()
    st.header("Filter Controls")

    st.subheader("Exclusions")
    use_legacy_exclusion = st.checkbox(
        "Use legacy exclusion (HS 01-27 only)",
        value=False,
        help=(
            "Default (off): excludes HS 01-27 plus HS 68 (stone articles), "
            "HS 71 (precious metals), and raw agricultural fibers (HS4 headings "
            "5001 silk, 5101 wool, 5201 cotton, 5301 flax). "
            "Legacy (on): only excludes HS 01-27 -- reproduces pre-2026-04 results."
        ),
    )

    st.subheader("Energy Thresholds")
    energy_pct   = st.slider("Energy intensity (carriers) percentile", 0, 100, 75,
                              help="amount_carriers >= this percentile")
    elec_pct     = st.slider("Electricity intensity percentile", 0, 100, 50,
                              help="amount_electric_energy >= this percentile")
    filter_logic = st.radio("Energy filter logic", ["OR", "AND"], index=0,
                            help="OR: either criterion suffices. AND: both required.")

    st.subheader("Trade Volume")
    trade_pct = st.slider("Trade volume percentile", 0, 100, 15,
                           help="global_export_value >= this percentile")

    st.subheader("Additional Filters")
    cbam_only  = st.checkbox("CBAM-covered products only", value=False)
    green_only = st.checkbox("Green supply chain only", value=False)
    green_topics = []
    if green_only:
        all_topics = sorted([t for t in df["green_topic"].unique() if t])
        green_topics = st.multiselect("Select supply chains", all_topics, default=all_topics)

    rca_threshold = st.slider("Morocco RCA minimum", 0.0, 5.0, 0.0, 0.1,
                               help="Set > 0 to require Morocco has existing exports")

# ============================================================
# APPLY FILTERS
# ============================================================
filtered, stage1_meta = apply_stage1_filter(
    df,
    energy_percentile=energy_pct,
    elec_percentile=elec_pct,
    trade_percentile=trade_pct,
    filter_logic=filter_logic,
    use_legacy=use_legacy_exclusion,
    cbam_only=cbam_only,
    green_only=green_only,
    green_topics=green_topics,
    rca_threshold=rca_threshold,
    return_metadata=True,
)
filter_description = stage1_meta["description"]
thresholds = stage1_meta["thresholds"]
energy_thresh = thresholds["energy_threshold"]
elec_thresh = thresholds["elec_threshold"]
trade_thresh = thresholds["trade_threshold"]

# Save to session state
st.session_state.filtered_products = filtered
st.session_state.stage_1_complete  = True

# ============================================================
# KPI ROW
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Products", f"{len(filtered):,}")
col2.metric("% of Universe", f"{100 * len(filtered) / len(df):.1f}%")
col3.metric("Global Trade", format_dollars(filtered["global_export_value"].sum()))
col4.metric("HS2 Chapters", filtered["hs2_code"].nunique())
col5.metric("Avg Elec. Intensity", f"{filtered['amount_electric_energy'].mean():.2f} MJ/$")

st.caption(f"**Active filters:** {filter_description}")

# ============================================================
# MAIN AREA
# ============================================================
tab_treemap, tab_bar, tab_table = st.tabs(["Treemap", "Industry List", "Data Table"])

with tab_treemap:
    treemap_metric = st.radio(
        "Size by:", ["global_export_value", "amount_carriers", "amount_electric_energy"],
        format_func=lambda x: VARIABLE_LABELS.get(x, x),
        horizontal=True,
    )
    fig = make_treemap(filtered, treemap_metric,
                       title=f"Filtered Products by HS2 -- {VARIABLE_LABELS.get(treemap_metric, treemap_metric)}")
    st.plotly_chart(fig, use_container_width=True)

with tab_bar:
    bar_metric = st.radio(
        "Sort by:", ["global_export_value", "amount_carriers", "amount_electric_energy", "pci"],
        format_func=lambda x: VARIABLE_LABELS.get(x, x),
        horizontal=True,
        key="bar_sort",
    )
    agg_func = "sum" if bar_metric in ("global_export_value",) else "mean"
    hs2_bar = filtered.groupby(["hs2_code", "hs2_name"]).agg(
        value=(bar_metric, agg_func),
        n_products=("hs_product_code", "count"),
    ).reset_index().sort_values("value", ascending=True)

    bar_n = st.slider("Top N industries", 5, min(40, len(hs2_bar)), min(20, len(hs2_bar)), key="bar_n")
    top_hs2 = hs2_bar.nlargest(bar_n, "value").sort_values("value", ascending=True)

    fig = px.bar(
        top_hs2, x="value", y="hs2_name", orientation="h",
        color_discrete_sequence=[MOROCCO_RED],
        title=f"Top {bar_n} HS2 Industries by {VARIABLE_LABELS.get(bar_metric, bar_metric)}",
        custom_data=["hs2_code", "n_products"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Value: %{x:,.2f}<br>HS2: %{customdata[0]}<br>Products: %{customdata[1]}<extra></extra>"
    )
    agg_label = "Total" if agg_func == "sum" else "Avg"
    fig.update_layout(
        template=GL_TEMPLATE,
        height=max(400, bar_n * 25),
        yaxis_title="",
        xaxis_title=f"{agg_label} {VARIABLE_LABELS.get(bar_metric, bar_metric)}",
        margin=dict(l=300),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_table:
    tbl = filtered.copy()
    tbl["hs4_code"] = tbl["hs_product_code"].astype(str).str.zfill(6).str[:4]
    tbl["hs4_description"] = tbl["hs4_code"].map(_HS4_DESC_LOOKUP).fillna("")
    if "description" in tbl.columns:
        tbl["description"] = tbl["description"].where(
            tbl["description"].notna() & (tbl["description"] != ""),
            tbl["hs4_description"],
        )

    display_cols = [
        "hs_product_code", "hs4_code", "hs4_description", "hs2_name",
        "description", "amount_carriers", "amount_electric_energy",
        "global_export_value", "cbam_flag",
    ]
    display_cols = [c for c in display_cols if c in tbl.columns]

    col_config_tbl = {
        "hs_product_code": st.column_config.TextColumn("HS6 Code", width="small"),
        "hs4_code": st.column_config.TextColumn("HS4 Parent", width="small"),
        "hs4_description": st.column_config.TextColumn("HS4 Description", width="large"),
        "hs2_name": st.column_config.TextColumn("HS2 Chapter", width="medium"),
        "description": st.column_config.TextColumn("Product Description", width="large"),
        "amount_carriers": st.column_config.NumberColumn("Energy Intensity (MJ/$)", format="%.2f"),
        "amount_electric_energy": st.column_config.NumberColumn("Elec. Intensity (MJ/$)", format="%.2f"),
        "global_export_value": st.column_config.NumberColumn("Global Trade ($)", format="$%.0f"),
        "cbam_flag": st.column_config.CheckboxColumn("CBAM"),
    }

    st.dataframe(
        tbl[display_cols].sort_values("global_export_value", ascending=False),
        column_config=col_config_tbl,
        use_container_width=True,
        height=500,
    )
    st.markdown(f"**{len(filtered)} products** passing filters")
    download_csv(filtered, "powershoring_filtered_products.csv", filter_description)

# Threshold distributions in expander
with st.expander("View threshold distributions", expanded=False):
    dist_c1, dist_c2, dist_c3 = st.columns(3)
    with dist_c1:
        st.markdown("**Energy Intensity (Carriers)**")
        fig_e = go.Figure()
        fig_e.add_trace(go.Histogram(x=df["amount_carriers"].dropna(), name="All products",
                                     marker_color=GREY, opacity=0.6, nbinsx=50))
        fig_e.add_trace(go.Histogram(x=filtered["amount_carriers"].dropna(), name="Filtered",
                                     marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50))
        fig_e.add_vline(x=energy_thresh, line_dash="dash", line_color="black",
                        annotation_text=f"{energy_pct}th pct: {energy_thresh:.1f}")
        fig_e.update_layout(template=GL_TEMPLATE, barmode="overlay", height=300,
                            xaxis_title="Energy Intensity (MJ/$)", yaxis_title="Count")
        st.plotly_chart(fig_e, use_container_width=True)

    with dist_c2:
        st.markdown("**Electricity Intensity**")
        fig_elec = go.Figure()
        fig_elec.add_trace(go.Histogram(x=df["amount_electric_energy"].dropna(), name="All products",
                                        marker_color=GREY, opacity=0.6, nbinsx=50))
        fig_elec.add_trace(go.Histogram(x=filtered["amount_electric_energy"].dropna(), name="Filtered",
                                        marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50))
        fig_elec.add_vline(x=elec_thresh, line_dash="dash", line_color="black",
                           annotation_text=f"{elec_pct}th pct: {elec_thresh:.1f}")
        fig_elec.update_layout(template=GL_TEMPLATE, barmode="overlay", height=300,
                               xaxis_title="Electricity Intensity (MJ/$)", yaxis_title="Count")
        st.plotly_chart(fig_elec, use_container_width=True)

    with dist_c3:
        st.markdown("**Trade Volume**")
        fig_t = go.Figure()
        fig_t.add_trace(go.Histogram(
            x=df["global_export_value"].dropna().apply(lambda x: max(x, 1)).apply(math.log10),
            name="All products", marker_color=GREY, opacity=0.6, nbinsx=50))
        fig_t.add_trace(go.Histogram(
            x=filtered["global_export_value"].dropna().apply(lambda x: max(x, 1)).apply(math.log10),
            name="Filtered", marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50))
        fig_t.update_layout(template=GL_TEMPLATE, barmode="overlay", height=300,
                            xaxis_title="Log10(Global Export Value $)", yaxis_title="Count")
        st.plotly_chart(fig_t, use_container_width=True)

# Stage gate
st.divider()
st.success(f"{len(filtered):,} products selected. Proceed to Likelihood and Prioritization to score and rank.")
if st.button("Proceed to Likelihood and Prioritization", type="primary"):
    st.switch_page("pages/2_Likelihood_Prioritization.py")
