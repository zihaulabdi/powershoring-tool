"""Stage 1: Filtering — Define the candidate universe."""

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
    download_csv, format_dollars, percentile_rank,
)

st.set_page_config(page_title="Stage 1: Filtering", layout="wide")
st.title("Stage 1: Filtering")
st.markdown("Define the candidate product universe by setting energy intensity and trade volume thresholds.")

# Load data
df = load_data()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("Filter Controls")

    st.subheader("Exclusions")
    exclude_non_mfg = st.checkbox("Exclude non-manufacturing (HS 01-24)", value=False)
    exclude_extractive = st.checkbox("Exclude extractive (HS 25-27)", value=True)

    st.subheader("Energy Thresholds")
    energy_pct = st.slider(
        "Energy intensity (carriers) percentile",
        0, 100, 75,
        help="amount_carriers >= this percentile"
    )
    elec_pct = st.slider(
        "Electricity intensity percentile",
        0, 100, 50,
        help="amount_electric_energy >= this percentile"
    )
    filter_logic = st.radio("Energy filter logic", ["OR", "AND"], index=0,
                            help="OR: either criterion suffices. AND: both required.")

    st.subheader("Trade Volume")
    trade_pct = st.slider("Trade volume percentile", 0, 100, 15,
                          help="global_export_value >= this percentile")

    st.subheader("Additional Filters")
    cbam_only = st.checkbox("CBAM-covered products only", value=False)
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
filtered = df.copy()
filter_parts = []

# Exclusions
if exclude_non_mfg:
    filtered = filtered[filtered["hs2_code"].astype(int) >= 25]
    filter_parts.append("Manufacturing only (HS >= 25)")
if exclude_extractive:
    filtered = filtered[~filtered["hs2_code"].astype(int).isin([25, 26, 27])]
    filter_parts.append("Excl. extractive (HS 25-27)")

# Energy thresholds
energy_thresh = df["amount_carriers"].quantile(energy_pct / 100)
elec_thresh = df["amount_electric_energy"].quantile(elec_pct / 100)
trade_thresh = df["global_export_value"].quantile(trade_pct / 100)

if filter_logic == "OR":
    energy_mask = (
        (filtered["amount_carriers"] >= energy_thresh) |
        (filtered["amount_electric_energy"] >= elec_thresh)
    )
    filter_parts.append(f"Carriers >= {energy_thresh:.2f} MJ/$ ({energy_pct}th pct) OR Electricity >= {elec_thresh:.2f} MJ/$ ({elec_pct}th pct)")
else:
    energy_mask = (
        (filtered["amount_carriers"] >= energy_thresh) &
        (filtered["amount_electric_energy"] >= elec_thresh)
    )
    filter_parts.append(f"Carriers >= {energy_thresh:.2f} AND Electricity >= {elec_thresh:.2f}")

trade_mask = filtered["global_export_value"] >= trade_thresh
filter_parts.append(f"Trade >= {format_dollars(trade_thresh)} ({trade_pct}th pct)")

filtered = filtered[energy_mask & trade_mask]

if cbam_only:
    filtered = filtered[filtered["cbam_flag"] == 1]
    filter_parts.append("CBAM only")
if green_only and green_topics:
    filtered = filtered[filtered["green_supply_chain_flag"] == 1]
    filtered = filtered[filtered["green_topic"].apply(
        lambda x: any(t in str(x) for t in green_topics)
    )]
    filter_parts.append(f"Green SC: {', '.join(green_topics)}")
if rca_threshold > 0:
    filtered = filtered[filtered["rca_mar"] >= rca_threshold]
    filter_parts.append(f"Morocco RCA >= {rca_threshold}")

filter_description = " | ".join(filter_parts)

# Save to session state
st.session_state.filtered_products = filtered

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
# TABS  (Scatter removed per user request)
# ============================================================
tab_treemap, tab_bar, tab_dist, tab_table = st.tabs(
    ["Treemap", "Bar Chart", "Distributions", "Data Table"]
)

# --- TREEMAP ---
with tab_treemap:
    treemap_metric = st.radio(
        "Size by:", ["global_export_value", "amount_carriers", "amount_electric_energy"],
        format_func=lambda x: VARIABLE_LABELS.get(x, x),
        horizontal=True,
    )
    fig = make_treemap(filtered, treemap_metric,
                       title=f"Filtered Products by HS2 — {VARIABLE_LABELS.get(treemap_metric, treemap_metric)}")
    st.plotly_chart(fig, use_container_width=True)

# --- BAR CHART (HS2-level, like the treemap) ---
with tab_bar:
    bar_metric = st.radio(
        "Sort by:", ["global_export_value", "amount_carriers", "amount_electric_energy", "pci"],
        format_func=lambda x: VARIABLE_LABELS.get(x, x),
        horizontal=True,
        key="bar_sort",
    )
    # Aggregate to HS2 level
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

# --- DISTRIBUTIONS ---
with tab_dist:
    # Row 1: Energy Intensity + Electricity Intensity
    dist_r1c1, dist_r1c2 = st.columns(2)
    with dist_r1c1:
        st.markdown("**Energy Intensity (Carriers) Distribution**")
        fig_e = go.Figure()
        fig_e.add_trace(go.Histogram(
            x=df["amount_carriers"].dropna(), name="All products",
            marker_color=GREY, opacity=0.6, nbinsx=50,
        ))
        fig_e.add_trace(go.Histogram(
            x=filtered["amount_carriers"].dropna(), name="Filtered",
            marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50,
        ))
        fig_e.add_vline(x=energy_thresh, line_dash="dash", line_color="black",
                        annotation_text=f"{energy_pct}th pct: {energy_thresh:.1f}")
        fig_e.update_layout(template=GL_TEMPLATE, barmode="overlay", height=350,
                            xaxis_title="Amount Carriers (MJ/$)", yaxis_title="Count")
        st.plotly_chart(fig_e, use_container_width=True)

    with dist_r1c2:
        st.markdown("**Electricity Intensity Distribution**")
        fig_elec = go.Figure()
        fig_elec.add_trace(go.Histogram(
            x=df["amount_electric_energy"].dropna(), name="All products",
            marker_color=GREY, opacity=0.6, nbinsx=50,
        ))
        fig_elec.add_trace(go.Histogram(
            x=filtered["amount_electric_energy"].dropna(), name="Filtered",
            marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50,
        ))
        fig_elec.add_vline(x=elec_thresh, line_dash="dash", line_color="black",
                           annotation_text=f"{elec_pct}th pct: {elec_thresh:.1f}")
        fig_elec.update_layout(template=GL_TEMPLATE, barmode="overlay", height=350,
                               xaxis_title="Electricity Intensity (MJ/$)", yaxis_title="Count")
        st.plotly_chart(fig_elec, use_container_width=True)

    # Row 2: Electricity Share + Trade Volume
    dist_r2c1, dist_r2c2 = st.columns(2)
    with dist_r2c1:
        st.markdown("**Electricity Share Distribution**")
        fig_es = go.Figure()
        fig_es.add_trace(go.Histogram(
            x=df["electricity_share"].dropna(), name="All products",
            marker_color=GREY, opacity=0.6, nbinsx=50,
        ))
        fig_es.add_trace(go.Histogram(
            x=filtered["electricity_share"].dropna(), name="Filtered",
            marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50,
        ))
        fig_es.update_layout(template=GL_TEMPLATE, barmode="overlay", height=350,
                             xaxis_title="Electricity Share (electricity / carriers)", yaxis_title="Count")
        st.plotly_chart(fig_es, use_container_width=True)

    with dist_r2c2:
        st.markdown("**Trade Volume Distribution**")
        fig_t = go.Figure()
        fig_t.add_trace(go.Histogram(
            x=df["global_export_value"].dropna().apply(lambda x: max(x, 1)).apply(
                lambda x: math.log10(x)
            ), name="All products", marker_color=GREY, opacity=0.6, nbinsx=50,
        ))
        fig_t.add_trace(go.Histogram(
            x=filtered["global_export_value"].dropna().apply(lambda x: max(x, 1)).apply(
                lambda x: math.log10(x)
            ), name="Filtered", marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50,
        ))
        fig_t.update_layout(template=GL_TEMPLATE, barmode="overlay", height=350,
                            xaxis_title="Log10(Global Export Value $)", yaxis_title="Count")
        st.plotly_chart(fig_t, use_container_width=True)

# --- DATA TABLE ---
with tab_table:
    display_cols = [
        "hs_product_code", "description", "hs2_name",
        "global_export_value", "amount_carriers", "amount_electric_energy",
        "electricity_share", "cbam_flag", "cbam_category", "green_topic",
        "rca_mar", "pci",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values("global_export_value", ascending=False),
        use_container_width=True,
        height=500,
    )

    st.markdown(f"**{len(filtered)} products** passing filters")
    download_csv(filtered, "powershoring_filtered_products.csv", filter_description)
