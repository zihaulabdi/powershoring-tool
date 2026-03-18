"""Comparison — Compare user-saved scenarios side by side."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_treemap, download_csv, format_dollars,
    get_hs2_color_map,
)

st.set_page_config(page_title="Comparison", layout="wide")
st.title("Comparison")
st.markdown("Compare your saved scenarios side by side.")

hs2_color_map = get_hs2_color_map()

# ============================================================
# LOAD SAVED SCENARIOS (user-generated only)
# ============================================================
saved = st.session_state.get("saved_scenarios", {})

# Normalize: saved_scenarios values can be dict {"products": df, ...} or bare df
available = {}
scenario_descs = {}
for name, data in saved.items():
    if isinstance(data, dict) and "products" in data:
        available[name] = data["products"]
        if data.get("desc"):
            scenario_descs[name] = data["desc"]
    elif isinstance(data, pd.DataFrame):
        available[name] = data

if len(available) < 2:
    st.warning(
        "You need at least **two saved scenarios** to compare. "
        "Go to **Likelihood & Prioritization**, adjust the weights, "
        "then use **Save Scenario** in the sidebar to save each scenario you want to compare."
    )
    if available:
        st.info(f"You have {len(available)} saved scenario(s) so far: {', '.join(available.keys())}")
    else:
        st.info("No saved scenarios yet.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
scenario_names = list(available.keys())

RANK_VARS = [
    "global_export_value", "amount_carriers", "amount_electric_energy",
    "electricity_share", "rca_mar", "pci", "density_mar", "cog_mar",
    "cbam_score", "vulnerability_score", "export_cagr_2012_2023",
    "inv_hhi", "weighted_degree", "likelihood_score",
    "feasibility_score", "attractiveness_score", "composite_score",
    "product_distance",
]

with st.sidebar:
    st.header("Scenario Selection")
    name_a = st.selectbox("Scenario A", scenario_names, index=0)
    name_b = st.selectbox("Scenario B", scenario_names,
                          index=min(1, len(scenario_names) - 1))

    st.divider()
    st.header("View Options")
    view_by = st.radio(
        "Aggregate by",
        ["Products (HS6)", "Industries (HS2)", "Green Value Chain"],
        help="How to group products in the comparison",
    )

    df_sample = available[name_a]
    rank_vars_available = [v for v in RANK_VARS if v in df_sample.columns]
    rank_var = st.selectbox(
        "Rank / size by", rank_vars_available, index=0,
        format_func=lambda x: VARIABLE_LABELS.get(x, x),
        key="comp_rank_var",
    )

df_a = available[name_a].copy()
df_b = available[name_b].copy()

df_a["_hs"] = df_a["hs_product_code"].astype(str)
df_b["_hs"] = df_b["hs_product_code"].astype(str)

set_a = set(df_a["_hs"])
set_b = set(df_b["_hs"])
overlap = set_a & set_b
only_a = set_a - set_b
only_b = set_b - set_a

# ============================================================
# SCENARIO DESCRIPTIONS
# ============================================================
desc_a = scenario_descs.get(name_a)
desc_b = scenario_descs.get(name_b)
if desc_a or desc_b:
    col_da, col_db = st.columns(2)
    with col_da:
        if desc_a:
            st.caption(f"**{name_a}:** {desc_a}")
    with col_db:
        if desc_b:
            st.caption(f"**{name_b}:** {desc_b}")

# ============================================================
# KPI ROW
# ============================================================
st.markdown("### Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**{name_a}**")
    st.metric("Products", f"{len(df_a):,}")
    st.metric("Global Trade", format_dollars(df_a["global_export_value"].sum()))
    st.metric("HS2 Chapters", df_a["hs2_code"].nunique() if "hs2_code" in df_a.columns else "—")

with col2:
    st.markdown("**Overlap**")
    st.metric("Shared Products", f"{len(overlap):,}")
    overlap_pct = 100 * len(overlap) / max(len(set_a | set_b), 1)
    st.metric("Overlap %", f"{overlap_pct:.1f}%")
    st.metric("Only in A", f"{len(only_a):,}")
    st.metric("Only in B", f"{len(only_b):,}")

with col3:
    st.markdown(f"**{name_b}**")
    st.metric("Products", f"{len(df_b):,}")
    st.metric("Global Trade", format_dollars(df_b["global_export_value"].sum()))
    st.metric("HS2 Chapters", df_b["hs2_code"].nunique() if "hs2_code" in df_b.columns else "—")


# ============================================================
# HELPER
# ============================================================
def aggregate_by(df, view, metric):
    agg_func = "sum" if metric in ("global_export_value", "value_mar") else "mean"
    if view == "Products (HS6)":
        return df
    elif view == "Industries (HS2)":
        return df.groupby(["hs2_code", "hs2_name"]).agg(
            value=(metric, agg_func),
            n_products=("hs_product_code", "count"),
        ).reset_index().rename(columns={"hs2_name": "label"})
    elif view == "Green Value Chain":
        col = "green_topic" if "green_topic" in df.columns else "hs2_name"
        tmp = df.copy()
        tmp[col] = tmp[col].fillna("").replace("", "No green topic")
        return tmp.groupby(col).agg(
            value=(metric, agg_func),
            n_products=("hs_product_code", "count"),
        ).reset_index().rename(columns={col: "label"})
    return df


# ============================================================
# TABS: Side-by-Side | Overlap Table
# ============================================================
tab_side, tab_overlap = st.tabs(["Side-by-Side", "Overlap Table"])

# --- SIDE-BY-SIDE ---
with tab_side:
    st.markdown("#### Treemaps")
    col_l, col_r = st.columns(2)

    if view_by in ("Products (HS6)", "Industries (HS2)"):
        with col_l:
            st.markdown(f"**{name_a}**")
            fig_tm_a = make_treemap(
                df_a, rank_var,
                title=f"{name_a} — {VARIABLE_LABELS.get(rank_var, rank_var)}",
                color_map=hs2_color_map,
            )
            st.plotly_chart(fig_tm_a, use_container_width=True)
        with col_r:
            st.markdown(f"**{name_b}**")
            fig_tm_b = make_treemap(
                df_b, rank_var,
                title=f"{name_b} — {VARIABLE_LABELS.get(rank_var, rank_var)}",
                color_map=hs2_color_map,
            )
            st.plotly_chart(fig_tm_b, use_container_width=True)

    elif view_by == "Green Value Chain":
        all_topics = sorted(set(
            list(df_a["green_topic"].fillna("").replace("", "No green topic").unique()) +
            list(df_b["green_topic"].fillna("").replace("", "No green topic").unique())
        ) if "green_topic" in df_a.columns else [])
        green_cmap = {}
        _ci = 0
        for t in all_topics:
            if t == "No green topic":
                green_cmap[t] = GREY
            else:
                green_cmap[t] = GL_PALETTE_EXT[_ci % len(GL_PALETTE_EXT)]
                _ci += 1

        for col_slot, (sname, sdf) in zip([col_l, col_r], [(name_a, df_a), (name_b, df_b)]):
            with col_slot:
                st.markdown(f"**{sname}**")
                tmp = sdf.copy()
                if "green_topic" in tmp.columns:
                    tmp["green_topic"] = tmp["green_topic"].fillna("").replace("", "No green topic")
                    agg = tmp.groupby("green_topic").agg(
                        value=(rank_var, "sum" if rank_var == "global_export_value" else "mean"),
                        n_products=("hs_product_code", "count"),
                    ).reset_index()
                    fig_gv = px.treemap(
                        agg, path=["green_topic"], values="value",
                        color="green_topic", color_discrete_map=green_cmap,
                        title=f"{sname} — Green Value Chain",
                    )
                    fig_gv.update_layout(template=GL_TEMPLATE, height=400,
                                         margin=dict(t=50, l=10, r=10, b=10))
                    st.plotly_chart(fig_gv, use_container_width=True)
                else:
                    st.info("No green_topic column available.")

    # Bar charts — always HS2 chapter level, ranked by number of HS6 products, top 5
    st.markdown("#### Top 5 HS2 Chapters by Number of Products")
    col_bl, col_br = st.columns(2)

    for col_slot, (sname, sdf, color) in zip(
        [col_bl, col_br],
        [(name_a, df_a, MOROCCO_RED), (name_b, df_b, GL_PALETTE_EXT[1])],
    ):
        with col_slot:
            st.markdown(f"**{sname}**")
            hs2_counts = (
                sdf.groupby("hs2_name")["hs_product_code"]
                .count()
                .reset_index()
                .rename(columns={"hs_product_code": "n_products"})
                .nlargest(5, "n_products")
                .sort_values("n_products", ascending=True)
            )
            fig_bar = px.bar(
                hs2_counts, x="n_products", y="hs2_name", orientation="h",
                color_discrete_sequence=[color],
            )
            fig_bar.update_layout(
                template=GL_TEMPLATE, height=280, showlegend=False,
                xaxis_title="Number of HS6 products", yaxis_title="",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Feasibility vs Attractiveness scatter (if scores exist)
    has_scores = (
        "feasibility_score" in df_a.columns and "attractiveness_score" in df_a.columns and
        "feasibility_score" in df_b.columns and "attractiveness_score" in df_b.columns
    )
    if has_scores:
        st.markdown("#### Feasibility vs. Attractiveness")
        col_sl, col_sr = st.columns(2)
        for col_slot, (sname, sdf, color) in zip(
            [col_sl, col_sr],
            [(name_a, df_a, MOROCCO_RED), (name_b, df_b, GL_PALETTE_EXT[1])],
        ):
            with col_slot:
                st.markdown(f"**{sname}**")
                fig_sc = px.scatter(
                    sdf, x="feasibility_score", y="attractiveness_score",
                    size="global_export_value" if "global_export_value" in sdf.columns else None,
                    color="hs2_name" if "hs2_name" in sdf.columns else None,
                    hover_name="description" if "description" in sdf.columns else None,
                    color_discrete_map=hs2_color_map,
                    size_max=25,
                )
                fig_sc.update_layout(
                    template=GL_TEMPLATE, height=450,
                    xaxis_title="Feasibility", yaxis_title="Attractiveness",
                    showlegend=False,
                )
                st.plotly_chart(fig_sc, use_container_width=True)

# --- OVERLAP TABLE ---
with tab_overlap:
    cols_to_show = ["hs_product_code", "description", "hs2_name", "global_export_value"]
    score_cols = ["likelihood_score", "feasibility_score", "attractiveness_score", "composite_score"]
    for sc in score_cols:
        if sc in df_a.columns or sc in df_b.columns:
            cols_to_show.append(sc)
    cols_to_show = list(dict.fromkeys(cols_to_show))

    cols_a = [c for c in cols_to_show if c in df_a.columns]
    cols_b = [c for c in cols_to_show if c in df_b.columns]

    merge_df = pd.merge(
        df_a[cols_a + ["_hs"]],
        df_b[cols_b + ["_hs"]],
        on="_hs", how="outer", suffixes=(f" ({name_a})", f" ({name_b})"),
        indicator=True,
    )
    merge_df["Status"] = merge_df["_merge"].map({
        "both": "In both",
        "left_only": f"Only in {name_a}",
        "right_only": f"Only in {name_b}",
    })
    merge_df = merge_df.drop(columns=["_merge", "_hs"])

    status_order = {"In both": 0, f"Only in {name_a}": 1, f"Only in {name_b}": 2}
    merge_df["_sort"] = merge_df["Status"].map(status_order)
    merge_df = merge_df.sort_values("_sort").drop(columns=["_sort"])

    st.dataframe(merge_df, use_container_width=True, height=500)
    st.markdown(
        f"**{len(merge_df)} total unique products** — "
        f"{len(overlap)} shared, {len(only_a)} only in {name_a}, {len(only_b)} only in {name_b}"
    )
    download_csv(merge_df, "powershoring_comparison.csv",
                 f"Comparison: {name_a} vs {name_b}")
