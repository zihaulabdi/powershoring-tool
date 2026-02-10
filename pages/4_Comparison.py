"""Comparison — Compare scenarios and shortlists side by side."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_treemap, download_csv, format_dollars,
    percentile_rank,
)

st.set_page_config(page_title="Comparison", layout="wide")
st.title("Scenario Comparison")
st.markdown("Compare different scenarios and shortlists side by side.")

# ============================================================
# BUILD AVAILABLE SCENARIOS
# ============================================================
available = {}

if st.session_state.get("filtered_products") is not None:
    available["Stage 1: Filtered"] = st.session_state.filtered_products
if st.session_state.get("likelihood_products") is not None:
    available["Stage 2a: Likelihood"] = st.session_state.likelihood_products
if st.session_state.get("prioritized_products") is not None:
    available["Stage 2b: Prioritized"] = st.session_state.prioritized_products

# Add saved scenarios
scenario_descs = {}  # name → description string (likelihood weights etc.)
for name, data in st.session_state.get("saved_scenarios", {}).items():
    if isinstance(data, dict) and "products" in data:
        available[f"Saved: {name}"] = data["products"]
        if data.get("desc"):
            scenario_descs[f"Saved: {name}"] = data["desc"]
    elif isinstance(data, pd.DataFrame):
        available[f"Saved: {name}"] = data

if len(available) < 2:
    st.warning(
        "You need at least **2 scenarios** to compare. "
        "Run Stage 1 (Filtering) and Stage 2a (Likelihood) first, "
        "or save scenarios from Stage 2b (Prioritization)."
    )
    st.info(f"Currently available: {list(available.keys()) if available else 'None'}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
scenario_names = list(available.keys())

# Numeric vars for ranking
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
    view_by = st.radio("Aggregate by",
                       ["Products (HS6)", "Industries (HS2)", "NAICS", "Green Value Chain"],
                       help="How to group products in the comparison")

    # Rank / size variable
    df_sample = available[name_a]
    rank_vars_available = [v for v in RANK_VARS if v in df_sample.columns]
    rank_var = st.selectbox("Rank / size by", rank_vars_available,
                            index=0,
                            format_func=lambda x: VARIABLE_LABELS.get(x, x),
                            key="comp_rank_var")

    # Quadrant filter — always shown, sourced from live prioritization
    st.divider()
    st.header("Quadrant Filter")
    all_quadrants = ["Top Priorities", "Strategic Bets", "Low-Hanging Fruit", "Deprioritize"]

    # Pull quadrant mapping from live prioritized_products (dynamic — updates when user re-runs Prioritization)
    prio_df = st.session_state.get("prioritized_products")
    has_quadrant_source = prio_df is not None and "quadrant" in prio_df.columns
    if has_quadrant_source:
        quadrant_filter = st.multiselect("Include quadrants", all_quadrants, default=all_quadrants,
                                         help="Filter by quadrant from Stage 2b Prioritization (updates dynamically)")
        # Build lookup: hs_product_code → quadrant from the live prioritization
        quadrant_lookup = prio_df.set_index("hs_product_code")["quadrant"].to_dict()
    else:
        st.info("Run **Stage 2b: Prioritization** (Multidimensional mode) first to assign quadrants.")
        quadrant_filter = None
        quadrant_lookup = {}

df_a = available[name_a].copy()
df_b = available[name_b].copy()

# Apply quadrant filter dynamically using the live prioritization lookup
if quadrant_filter is not None and quadrant_lookup:
    # Map quadrant from live prioritization onto each scenario's products
    df_a["_quadrant"] = df_a["hs_product_code"].map(quadrant_lookup)
    df_b["_quadrant"] = df_b["hs_product_code"].map(quadrant_lookup)
    # Keep products that are in a selected quadrant (or have no quadrant assignment)
    df_a = df_a[df_a["_quadrant"].isin(quadrant_filter)]
    df_b = df_b[df_b["_quadrant"].isin(quadrant_filter)]
    df_a = df_a.drop(columns=["_quadrant"])
    df_b = df_b.drop(columns=["_quadrant"])

# Ensure hs_product_code is string for set operations
df_a["_hs"] = df_a["hs_product_code"].astype(str)
df_b["_hs"] = df_b["hs_product_code"].astype(str)

set_a = set(df_a["_hs"])
set_b = set(df_b["_hs"])
overlap = set_a & set_b
only_a = set_a - set_b
only_b = set_b - set_a

# ============================================================
# SCENARIO SUBTITLES (show likelihood weights if available)
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
# KPI ROW  (always side-by-side)
# ============================================================
st.markdown("### Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**{name_a}**")
    st.metric("Products", f"{len(df_a):,}")
    st.metric("Global Trade", format_dollars(df_a["global_export_value"].sum()))
    st.metric("HS2 Chapters", df_a["hs2_code"].nunique() if "hs2_code" in df_a.columns else "-")

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
    st.metric("HS2 Chapters", df_b["hs2_code"].nunique() if "hs2_code" in df_b.columns else "-")

# ============================================================
# HELPER: aggregate by chosen dimension
# ============================================================
def aggregate_by(df, view, metric):
    """Aggregate df according to the chosen view, summing/averaging the metric."""
    agg_func = "sum" if metric in ("global_export_value", "value_mar") else "mean"
    if view == "Products (HS6)":
        return df  # no aggregation
    elif view == "Industries (HS2)":
        return df.groupby(["hs2_code", "hs2_name"]).agg(
            value=(metric, agg_func),
            n_products=("hs_product_code", "count"),
        ).reset_index().rename(columns={"hs2_name": "label"})
    elif view == "NAICS":
        if "naics_code" not in df.columns:
            # Try to use hs2_name as fallback
            return df.groupby("hs2_name").agg(
                value=(metric, agg_func),
                n_products=("hs_product_code", "count"),
            ).reset_index().rename(columns={"hs2_name": "label"})
        return df.groupby("naics_code").agg(
            value=(metric, agg_func),
            n_products=("hs_product_code", "count"),
        ).reset_index().rename(columns={"naics_code": "label"})
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
# TABS
# ============================================================
tab_side, tab_radar, tab_overlap = st.tabs(
    ["Side-by-Side", "Radar Chart", "Overlap Table"]
)

# --- SIDE-BY-SIDE ---
with tab_side:
    # --- Build unified HS2 color map for consistent treemap colors ---
    all_hs2_names = sorted(set(
        list(df_a["hs2_name"].dropna().unique()) + list(df_b["hs2_name"].dropna().unique())
    ))
    hs2_color_map = {name: GL_PALETTE_EXT[i % len(GL_PALETTE_EXT)] for i, name in enumerate(all_hs2_names)}

    # --- TREEMAPS (always side-by-side) ---
    st.markdown("#### Treemaps")
    col_l, col_r = st.columns(2)

    if view_by == "Products (HS6)" or view_by == "Industries (HS2)":
        # HS2 treemaps with consistent colors
        with col_l:
            st.markdown(f"**{name_a}**")
            fig_tm_a = make_treemap(df_a, rank_var,
                                    title=f"{name_a} — {VARIABLE_LABELS.get(rank_var, rank_var)}",
                                    color_map=hs2_color_map)
            st.plotly_chart(fig_tm_a, use_container_width=True)
        with col_r:
            st.markdown(f"**{name_b}**")
            fig_tm_b = make_treemap(df_b, rank_var,
                                    title=f"{name_b} — {VARIABLE_LABELS.get(rank_var, rank_var)}",
                                    color_map=hs2_color_map)
            st.plotly_chart(fig_tm_b, use_container_width=True)

    elif view_by == "Green Value Chain":
        # Build unified green_topic color map
        all_green_topics = sorted(set(
            list(df_a["green_topic"].fillna("").replace("", "No green topic").unique()) +
            list(df_b["green_topic"].fillna("").replace("", "No green topic").unique())
        ) if "green_topic" in df_a.columns and "green_topic" in df_b.columns else [])
        green_color_map = {}
        _ci = 0
        for t in all_green_topics:
            if t == "No green topic":
                green_color_map[t] = GREY
            else:
                green_color_map[t] = GL_PALETTE_EXT[_ci % len(GL_PALETTE_EXT)]
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
                        color="green_topic",
                        color_discrete_map=green_color_map if green_color_map else None,
                        color_discrete_sequence=GL_PALETTE_EXT if not green_color_map else None,
                        title=f"{sname} — Green Value Chain",
                    )
                    fig_gv.update_layout(template=GL_TEMPLATE, height=400, margin=dict(t=50, l=10, r=10, b=10))
                    st.plotly_chart(fig_gv, use_container_width=True)
                else:
                    st.info("No green_topic column available.")

    else:
        # NAICS — fall back to bar charts
        for col_slot, (sname, sdf) in zip([col_l, col_r], [(name_a, df_a), (name_b, df_b)]):
            with col_slot:
                st.markdown(f"**{sname}**")
                agg_df = aggregate_by(sdf, view_by, rank_var)
                top15 = agg_df.nlargest(15, "value").sort_values("value", ascending=True)
                fig_bar = px.bar(top15, x="value", y="label", orientation="h",
                                 color_discrete_sequence=[MOROCCO_RED if sname == name_a else GL_PALETTE_EXT[1]])
                fig_bar.update_layout(template=GL_TEMPLATE, height=400, showlegend=False,
                                      xaxis_title=VARIABLE_LABELS.get(rank_var, rank_var), yaxis_title="")
                st.plotly_chart(fig_bar, use_container_width=True)

    # --- BAR CHARTS (always side-by-side) ---
    st.markdown("#### Bar Charts")
    col_bl, col_br = st.columns(2)

    agg_a = aggregate_by(df_a, view_by, rank_var)
    agg_b = aggregate_by(df_b, view_by, rank_var)

    if view_by != "Products (HS6)":
        top_n_comp = 15
        with col_bl:
            st.markdown(f"**{name_a}**")
            top_a = agg_a.nlargest(top_n_comp, "value").sort_values("value", ascending=True)
            fig_ba = px.bar(top_a, x="value", y="label", orientation="h",
                            color_discrete_sequence=[MOROCCO_RED])
            fig_ba.update_layout(template=GL_TEMPLATE, height=400, showlegend=False,
                                 xaxis_title=VARIABLE_LABELS.get(rank_var, rank_var), yaxis_title="")
            st.plotly_chart(fig_ba, use_container_width=True)

        with col_br:
            st.markdown(f"**{name_b}**")
            top_b = agg_b.nlargest(top_n_comp, "value").sort_values("value", ascending=True)
            fig_bb = px.bar(top_b, x="value", y="label", orientation="h",
                            color_discrete_sequence=[GL_PALETTE_EXT[1]])
            fig_bb.update_layout(template=GL_TEMPLATE, height=400, showlegend=False,
                                 xaxis_title=VARIABLE_LABELS.get(rank_var, rank_var), yaxis_title="")
            st.plotly_chart(fig_bb, use_container_width=True)
    else:
        # Product-level: top N products bar chart
        top_n_comp = 20
        for col_slot, (sname, sdf, color) in zip(
            [col_bl, col_br],
            [(name_a, df_a, MOROCCO_RED), (name_b, df_b, GL_PALETTE_EXT[1])],
        ):
            with col_slot:
                st.markdown(f"**{sname}** — Top {top_n_comp}")
                top_p = sdf.nlargest(top_n_comp, rank_var).sort_values(rank_var, ascending=True)
                top_p["_label"] = top_p["description"].fillna("").str[:50]
                fig_bp = px.bar(top_p, x=rank_var, y="_label", orientation="h",
                                color_discrete_sequence=[color])
                fig_bp.update_layout(template=GL_TEMPLATE, height=max(400, top_n_comp * 22),
                                     showlegend=False,
                                     xaxis_title=VARIABLE_LABELS.get(rank_var, rank_var), yaxis_title="")
                st.plotly_chart(fig_bp, use_container_width=True)

    # --- SIDE-BY-SIDE SCATTER (if prioritization scores exist) ---
    has_scores = ("feasibility_score" in df_a.columns and "attractiveness_score" in df_a.columns and
                  "feasibility_score" in df_b.columns and "attractiveness_score" in df_b.columns)
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
                    color_discrete_sequence=GL_PALETTE_EXT,
                    size_max=25,
                )
                fig_sc.update_layout(
                    template=GL_TEMPLATE, height=450,
                    xaxis_title="Feasibility", yaxis_title="Attractiveness",
                    showlegend=False,
                )
                st.plotly_chart(fig_sc, use_container_width=True)

# --- RADAR CHART ---
with tab_radar:
    st.markdown("Compare individual products across multiple dimensions.")

    radar_vars = [
        "amount_electric_energy", "amount_carriers", "global_export_value",
        "rca_mar", "pci", "density_mar",
    ]
    radar_vars = [v for v in radar_vars if v in df_a.columns or v in df_b.columns]

    if not radar_vars:
        st.warning("Not enough numeric variables for radar chart.")
    else:
        combined = pd.concat([df_a, df_b]).drop_duplicates(subset=["_hs"])
        product_options = combined.apply(
            lambda r: f"{r['_hs']} — {str(r.get('description', ''))[:50]}", axis=1
        ).tolist()[:200]

        selected_products = st.multiselect(
            "Select products to compare (up to 5)", product_options,
            default=product_options[:min(3, len(product_options))],
            max_selections=5,
        )

        if selected_products:
            selected_hs = [p.split(" — ")[0] for p in selected_products]
            radar_df = combined[combined["_hs"].isin(selected_hs)].copy()

            for v in radar_vars:
                if v in radar_df.columns:
                    vmin = combined[v].min()
                    vmax = combined[v].max()
                    if vmax > vmin:
                        radar_df[f"{v}_norm"] = (radar_df[v] - vmin) / (vmax - vmin) * 100
                    else:
                        radar_df[f"{v}_norm"] = 50

            fig = go.Figure()
            for i, (_, row) in enumerate(radar_df.iterrows()):
                desc = f"{row['_hs']} — {str(row.get('description', ''))[:30]}"
                values = [row.get(f"{v}_norm", 50) for v in radar_vars]
                values.append(values[0])

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[VARIABLE_LABELS.get(v, v) for v in radar_vars] + [VARIABLE_LABELS.get(radar_vars[0], radar_vars[0])],
                    fill="toself",
                    name=desc,
                    line_color=GL_PALETTE_EXT[i % len(GL_PALETTE_EXT)],
                    opacity=0.6,
                ))

            fig.update_layout(
                template=GL_TEMPLATE,
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=500,
                title="Product Comparison — Radar Chart",
            )
            st.plotly_chart(fig, use_container_width=True)

# --- OVERLAP TABLE ---
with tab_overlap:
    cols_to_show = ["hs_product_code", "description", "hs2_name", "global_export_value"]
    score_cols = ["likelihood_score", "feasibility_score", "attractiveness_score", "composite_score", "quadrant"]
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
    st.markdown(f"**{len(merge_df)} total unique products** — "
                f"{len(overlap)} shared, {len(only_a)} only in A, {len(only_b)} only in B")
    quadrant_desc = f", quadrants={quadrant_filter}" if quadrant_filter is not None else ""
    download_csv(merge_df, "powershoring_comparison.csv",
                 f"Comparison: {name_a} vs {name_b}{quadrant_desc}")
