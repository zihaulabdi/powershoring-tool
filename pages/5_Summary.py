"""Summary -- Top candidate industries at a glance."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    format_dollars, download_csv,
    SCENARIO_DEFS, run_scenario_scoring, DEFAULT_FEAS_WEIGHTS, DEFAULT_ATTR_WEIGHTS,
    aggregate_to_hs4, _default_stage1_filter,
    inject_custom_css,
)

st.set_page_config(page_title="Summary", layout="wide")
inject_custom_css()
st.title("Summary")
st.caption(
    "Top candidate industries based on the No Prior scenario. "
    "Standard thresholds: energy >= 75th pct OR electricity >= 50th pct, trade >= 15th pct."
)

df_all = load_data()

# ============================================================
# RUN NO PRIOR SCENARIO WITH DEFAULT THRESHOLDS
# ============================================================
with st.spinner("Calculating..."):
    filtered = _default_stage1_filter(df_all)

    no_prior_weights = SCENARIO_DEFS["No Prior"]["weights"]
    scored = run_scenario_scoring(
        filtered,
        no_prior_weights,
        DEFAULT_FEAS_WEIGHTS,
        DEFAULT_ATTR_WEIGHTS,
        fa_reference_df=filtered,
    )
    scored["composite_score"] = 0.60 * scored["feasibility_score"] + 0.40 * scored["attractiveness_score"]
    hs4 = aggregate_to_hs4(scored)
    top30_hs4 = hs4.nlargest(30, "composite_score").reset_index(drop=True)

# ============================================================
# TREEMAP
# ============================================================
treemap_choice = st.radio(
    "Size by:", ["By number of products", "By trade volume"],
    horizontal=True,
)

agg_tm = top30_hs4.groupby("hs2_name").agg(
    n_products=("n_products", "sum"),
    trade=("global_export_value", "sum"),
).reset_index()

if treemap_choice == "By number of products":
    agg_tm["value"] = agg_tm["n_products"]
else:
    agg_tm["value"] = agg_tm["trade"]

fig_tm = px.treemap(
    agg_tm, path=["hs2_name"], values="value",
    color="hs2_name",
    color_discrete_sequence=GL_PALETTE_EXT,
    title="HS2 Chapter Composition - Top 30 Products (No Prior Scenario)",
)
fig_tm.update_traces(textinfo="label+percent root")
fig_tm.update_layout(template=GL_TEMPLATE, height=450, margin=dict(t=50, l=10, r=10, b=10))
st.plotly_chart(fig_tm, use_container_width=True)

# ============================================================
# TABLE: TOP 5 HS2 CHAPTERS AND THEIR HS4 PRODUCTS
# ============================================================
st.markdown("### Top 5 HS2 Chapters and Their Candidate Industries")

# Rank HS2 chapters by avg composite of their HS4s in the top 30
hs2_avg = (
    top30_hs4.groupby("hs2_name")["composite_score"]
    .mean()
    .sort_values(ascending=False)
)
top5_hs2 = list(hs2_avg.index[:5])

# Filter top30 to those chapters
table_df = top30_hs4[top30_hs4["hs2_name"].isin(top5_hs2)].copy()
table_df["_hs2_rank"] = table_df["hs2_name"].map({h: i for i, h in enumerate(top5_hs2)})
table_df = (
    table_df
    .sort_values(["_hs2_rank", "composite_score"], ascending=[True, False])
    .drop(columns=["_hs2_rank"])
    .reset_index(drop=True)
)

# Pre-format trade as string (avoids NumberColumn formatting issues)
table_df["trade_fmt"] = table_df["global_export_value"].apply(format_dollars)

col_config = {
    "hs2_name": st.column_config.TextColumn("HS2 Chapter", width="medium"),
    "hs4_code": st.column_config.TextColumn("HS4 Code", width="small"),
    "name_short": st.column_config.TextColumn("Industry", width="large"),
    "composite_score": st.column_config.ProgressColumn(
        "Composite", min_value=0, max_value=100, format="%.0f"
    ),
    "feasibility_score": st.column_config.ProgressColumn(
        "Feasibility", min_value=0, max_value=100, format="%.0f"
    ),
    "attractiveness_score": st.column_config.ProgressColumn(
        "Attractiveness", min_value=0, max_value=100, format="%.0f"
    ),
    "trade_fmt": st.column_config.TextColumn("Global Trade"),
}

show_cols = [c for c in [
    "hs2_name", "hs4_code", "name_short",
    "composite_score", "feasibility_score", "attractiveness_score",
    "trade_fmt",
] if c in table_df.columns]

st.dataframe(
    table_df[show_cols],
    column_config=col_config,
    use_container_width=True,
    height=550,
)

download_csv(
    table_df[show_cols],
    "powershoring_summary_top30.csv",
    "No Prior scenario, standard thresholds, 60F/40A",
)
