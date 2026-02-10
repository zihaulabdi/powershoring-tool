"""Stage 2b: Prioritization — Rank products by feasibility and attractiveness."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_scatter, download_csv, format_dollars,
    percentile_rank, weighted_score,
)

st.set_page_config(page_title="Stage 2b: Prioritization", layout="wide")
st.title("Stage 2b: Prioritization")
st.markdown("Rank products by feasibility (Morocco capabilities) and attractiveness (market opportunity).")

# ============================================================
# LOAD DATA — scenario picker
# ============================================================
df_all = load_data()

# Build input options: current likelihood + saved scenarios
_input_options = {}
if st.session_state.get("likelihood_products") is not None:
    _input_options["Stage 2a: Likelihood (current)"] = {
        "products": st.session_state.likelihood_products,
        "desc": None,
    }
if st.session_state.get("filtered_products") is not None:
    _input_options["Stage 1: Filtered (current)"] = {
        "products": st.session_state.filtered_products,
        "desc": None,
    }
for _sname, _sdata in st.session_state.get("saved_scenarios", {}).items():
    if isinstance(_sdata, dict) and "products" in _sdata:
        _input_options[f"Saved: {_sname}"] = _sdata
    elif isinstance(_sdata, pd.DataFrame):
        _input_options[f"Saved: {_sname}"] = {"products": _sdata, "desc": None}

if _input_options:
    with st.sidebar:
        st.header("Input Scenario")
        _chosen = st.selectbox("Use products from:", list(_input_options.keys()), index=0, key="prio_input")
    _chosen_data = _input_options[_chosen]
    df = _chosen_data["products"].copy()
    _desc = _chosen_data.get("desc")
    if _desc:
        st.success(f"Using **{len(df):,}** products from **{_chosen}** — {_desc}")
    else:
        st.success(f"Using **{len(df):,}** products from **{_chosen}**.")
else:
    df = df_all.copy()
    st.warning("No prior filtering applied — using full product universe.")

# Variables available for axis selection
NUMERIC_VARS = [
    "rca_mar", "density_mar", "cog_mar", "pci",
    "global_export_value", "export_cagr_2012_2023",
    "amount_carriers", "amount_electric_energy", "electricity_share",
    "vulnerability_score", "cbam_score", "eu_import_share",
    "inv_hhi", "hhi", "n_exporting_countries",
    "weighted_degree", "eigenvector_centrality",
    "value_mar", "likelihood_score", "product_distance",
]
NUMERIC_VARS = [v for v in NUMERIC_VARS if v in df.columns]

COLOR_VARS = ["hs2_name", "cbam_flag", "green_supply_chain_flag", "green_topic"]
COLOR_VARS = [v for v in COLOR_VARS if v in df.columns]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Prioritization Mode")
    mode = st.radio("Mode", ["Bi-dimensional", "Multidimensional"], index=1)

    if mode == "Bi-dimensional":
        st.subheader("Axis Variables")
        x_var = st.selectbox("X-axis", NUMERIC_VARS,
                             index=NUMERIC_VARS.index("rca_mar") if "rca_mar" in NUMERIC_VARS else 0,
                             format_func=lambda x: VARIABLE_LABELS.get(x, x))
        y_var = st.selectbox("Y-axis", NUMERIC_VARS,
                             index=NUMERIC_VARS.index("global_export_value") if "global_export_value" in NUMERIC_VARS else 1,
                             format_func=lambda x: VARIABLE_LABELS.get(x, x))
        size_var = st.selectbox("Size", ["uniform"] + NUMERIC_VARS,
                                index=(NUMERIC_VARS.index("global_export_value") + 1) if "global_export_value" in NUMERIC_VARS else 0,
                                format_func=lambda x: VARIABLE_LABELS.get(x, x) if x != "uniform" else "Uniform")
        color_var = st.selectbox("Color", COLOR_VARS,
                                 format_func=lambda x: VARIABLE_LABELS.get(x, x))

        st.subheader("Quadrant Lines")
        line_method = st.radio("Threshold method", ["Median", "Mean", "Percentile", "None"], index=0)
        if line_method == "Percentile":
            line_pctile = st.slider("Percentile for lines", 10, 90, 50, 5, key="line_pctile")

        label_top = st.slider("Label top N", 0, 30, 5, key="prio_label",
                              help="Top N by 50/50 composite of X and Y axes")

    else:  # Multidimensional
        st.subheader("Feasibility Weights")
        f_rca = st.slider("Morocco RCA", 0, 100, 25, key="f_rca",
                          help="Revealed comparative advantage — existing export strength")
        f_density = st.slider("Morocco Density", 0, 100, 30, key="f_density",
                              help="Proximity to existing capabilities in product space")
        f_hhi = st.slider("Market fragmentation (1/HHI)", 0, 100, 25, key="f_hhi",
                          help="Lower concentration = easier to enter")
        f_dist = st.slider("Avg. Trade Distance (km)", 0, 100, 20, key="f_dist",
                           help="Short-distance products are more transport-sensitive — better powershoring candidates")

        st.subheader("Attractiveness Weights")
        a_market = st.slider("Market size", 0, 100, 25, key="a_market",
                             help="Global export value")
        a_growth = st.slider("Market growth", 0, 100, 20, key="a_growth",
                             help="Export CAGR 2012-2023")
        a_cog = st.slider("COG (opportunity gain)", 0, 100, 20, key="a_cog",
                          help="Contribution to future diversification")
        a_pci = st.slider("Product complexity", 0, 100, 15, key="a_pci",
                          help="Higher complexity = more value added")
        a_spillover = st.slider("Spillover potential", 0, 100, 20, key="a_spillover",
                                help="Network centrality — benefits to related industries")

        color_multi = st.selectbox("Color by", COLOR_VARS, key="color_multi",
                                   format_func=lambda x: VARIABLE_LABELS.get(x, x))
        label_top_multi = st.slider("Label top N", 0, 30, 10, key="prio_label_multi")



# ============================================================
# HELPER: prepare color column for categorical flags
# ============================================================
def prepare_color(plot_df, col):
    """Return (df_copy, color_col_name) with proper categorical handling."""
    out = plot_df.copy()
    if col == "cbam_flag":
        out["_color"] = out[col].map({1: "CBAM", 0: "Not CBAM"}).fillna("Not CBAM")
        return out, "_color"
    elif col == "green_supply_chain_flag":
        out["_color"] = out[col].map({1: "Green SC", 0: "Not Green SC"}).fillna("Not Green SC")
        return out, "_color"
    elif col == "green_topic":
        out["_color"] = out[col].fillna("").replace("", "No green topic")
        return out, "_color"
    return out, col


def color_map_for(col, plot_df, color_col):
    """Return color_discrete_map or color_discrete_sequence kwargs."""
    if col == "cbam_flag":
        return dict(color_discrete_map={"CBAM": MOROCCO_RED, "Not CBAM": GREY})
    elif col == "green_supply_chain_flag":
        return dict(color_discrete_map={"Green SC": GL_PALETTE_EXT[2], "Not Green SC": GREY})
    elif col == "green_topic":
        topics = sorted(plot_df[color_col].unique())
        cmap = {}
        color_idx = 0
        for t in topics:
            if t == "No green topic":
                cmap[t] = GREY
            else:
                cmap[t] = GL_PALETTE_EXT[color_idx % len(GL_PALETTE_EXT)]
                color_idx += 1
        return dict(color_discrete_map=cmap)
    else:
        return dict(color_discrete_sequence=GL_PALETTE_EXT)


# ============================================================
# BI-DIMENSIONAL MODE
# ============================================================
if mode == "Bi-dimensional":
    # Compute threshold lines
    if line_method == "Median":
        threshold_x = df[x_var].median()
        threshold_y = df[y_var].median()
    elif line_method == "Mean":
        threshold_x = df[x_var].mean()
        threshold_y = df[y_var].mean()
    elif line_method == "Percentile":
        threshold_x = df[x_var].quantile(line_pctile / 100)
        threshold_y = df[y_var].quantile(line_pctile / 100)
    else:
        threshold_x = None
        threshold_y = None

    # Prepare color
    plot_df, color_col = prepare_color(df, color_var)
    ckwargs = color_map_for(color_var, plot_df, color_col)

    # Build scatter
    size_arg = {}
    if size_var and size_var != "uniform":
        plot_df["_size"] = plot_df[size_var].clip(lower=0).fillna(0)
        size_arg = dict(size="_size", size_max=30)

    fig = px.scatter(
        plot_df, x=x_var, y=y_var,
        color=color_col,
        **ckwargs,
        **size_arg,
        hover_name="description" if "description" in plot_df.columns else None,
        hover_data={c: True for c in ["hs_product_code", "hs2_name", "global_export_value"] if c in plot_df.columns},
        title=f"{VARIABLE_LABELS.get(y_var, y_var)} vs. {VARIABLE_LABELS.get(x_var, x_var)}",
    )

    if threshold_x is not None:
        fig.add_vline(x=threshold_x, line_dash="dash", line_color="grey", opacity=0.5)
    if threshold_y is not None:
        fig.add_hline(y=threshold_y, line_dash="dash", line_color="grey", opacity=0.5)

    # Label top N by 50/50 composite of X and Y (percentile ranked)
    if label_top > 0:
        df["_x_pctile"] = percentile_rank(df[x_var].fillna(0))
        df["_y_pctile"] = percentile_rank(df[y_var].fillna(0))
        df["_composite_xy"] = (df["_x_pctile"] + df["_y_pctile"]) / 2
        top_prods = df.nlargest(label_top, "_composite_xy")
        for _, row in top_prods.iterrows():
            desc = str(row.get("description", ""))[:40]
            fig.add_annotation(
                x=row[x_var], y=row[y_var],
                text=desc, showarrow=True, arrowhead=2,
                font=dict(size=9),
            )
        df.drop(columns=["_x_pctile", "_y_pctile", "_composite_xy"], inplace=True, errors="ignore")

    # Quadrant labels
    if threshold_x is not None and threshold_y is not None:
        x_range = df[x_var].max() - df[x_var].min()
        y_range = df[y_var].max() - df[y_var].min()
        for qlabel, xpos, ypos in [
            ("High Y, Low X", threshold_x - x_range * 0.25, threshold_y + y_range * 0.35),
            ("High Y, High X", threshold_x + x_range * 0.25, threshold_y + y_range * 0.35),
            ("Low Y, Low X", threshold_x - x_range * 0.25, threshold_y - y_range * 0.35),
            ("Low Y, High X", threshold_x + x_range * 0.25, threshold_y - y_range * 0.35),
        ]:
            fig.add_annotation(
                x=xpos, y=ypos, text=qlabel,
                showarrow=False, font=dict(size=11, color="grey"), opacity=0.6,
            )

    fig.update_layout(
        template=GL_TEMPLATE,
        height=600,
        xaxis_title=VARIABLE_LABELS.get(x_var, x_var),
        yaxis_title=VARIABLE_LABELS.get(y_var, y_var),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table with all relevant columns
    st.markdown("---")
    display_cols = [
        "hs_product_code", "description", "hs2_name",
        x_var, y_var,
        "global_export_value", "amount_carriers", "amount_electric_energy",
        "rca_mar", "density_mar", "pci", "cog_mar",
        "cbam_flag", "cbam_score", "vulnerability_score",
        "export_cagr_2012_2023", "inv_hhi",
        "likelihood_score",
    ]
    display_cols = list(dict.fromkeys([c for c in display_cols if c in df.columns]))
    st.dataframe(
        df[display_cols].sort_values(y_var, ascending=False),
        use_container_width=True, height=400,
    )
    download_csv(df[display_cols], "powershoring_bidimensional.csv",
                 f"Bi-dimensional: X={x_var}, Y={y_var}")

# ============================================================
# MULTIDIMENSIONAL MODE
# ============================================================
else:
    # Compute feasibility
    feas_weights = {"rca_pctile": f_rca, "density_pctile": f_density, "hhi_pctile": f_hhi, "distance_pctile": f_dist}
    feas_components = {}
    feas_components["rca_pctile"] = percentile_rank(df["rca_mar"].fillna(0)) if "rca_mar" in df.columns else pd.Series(50, index=df.index)
    feas_components["density_pctile"] = percentile_rank(df["density_mar"].fillna(0)) if "density_mar" in df.columns else pd.Series(50, index=df.index)
    feas_components["hhi_pctile"] = percentile_rank(df["inv_hhi"].fillna(0)) if "inv_hhi" in df.columns else pd.Series(50, index=df.index)
    # Lower trade distance = more transport-sensitive = better powershoring candidate, so negate
    feas_components["distance_pctile"] = percentile_rank(-df["product_distance"].fillna(df["product_distance"].max() or 0)) if "product_distance" in df.columns else pd.Series(50, index=df.index)

    df["feasibility_score"] = weighted_score(df, feas_components, feas_weights)

    # Compute attractiveness
    attr_weights = {
        "market_size_pctile": a_market,
        "growth_pctile": a_growth,
        "cog_pctile": a_cog,
        "pci_pctile": a_pci,
        "spillover_pctile": a_spillover,
    }
    attr_components = {}
    attr_components["market_size_pctile"] = percentile_rank(df["global_export_value"].fillna(0)) if "global_export_value" in df.columns else pd.Series(50, index=df.index)
    attr_components["growth_pctile"] = percentile_rank(df["export_cagr_2012_2023"].fillna(0)) if "export_cagr_2012_2023" in df.columns else pd.Series(50, index=df.index)
    attr_components["cog_pctile"] = percentile_rank(df["cog_mar"].fillna(0)) if "cog_mar" in df.columns else pd.Series(50, index=df.index)
    attr_components["pci_pctile"] = percentile_rank(df["pci"].fillna(0)) if "pci" in df.columns else pd.Series(50, index=df.index)
    attr_components["spillover_pctile"] = percentile_rank(df["weighted_degree"].fillna(0)) if "weighted_degree" in df.columns else pd.Series(50, index=df.index)

    df["attractiveness_score"] = weighted_score(df, attr_components, attr_weights)

    # Composite score (equal weight to feasibility and attractiveness)
    df["composite_score"] = (df["feasibility_score"] + df["attractiveness_score"]) / 2

    # Assign quadrant labels
    feas_med = df["feasibility_score"].median()
    attr_med = df["attractiveness_score"].median()
    conditions = [
        (df["feasibility_score"] >= feas_med) & (df["attractiveness_score"] >= attr_med),
        (df["feasibility_score"] < feas_med) & (df["attractiveness_score"] >= attr_med),
        (df["feasibility_score"] >= feas_med) & (df["attractiveness_score"] < attr_med),
        (df["feasibility_score"] < feas_med) & (df["attractiveness_score"] < attr_med),
    ]
    quadrant_names = ["Top Priorities", "Strategic Bets", "Low-Hanging Fruit", "Deprioritize"]
    df["quadrant"] = "Deprioritize"
    for cond, qname in zip(conditions, quadrant_names):
        df.loc[cond, "quadrant"] = qname

    # Store component percentiles
    for k, v in feas_components.items():
        df[k] = v
    for k, v in attr_components.items():
        df[k] = v

    # Save to session state
    st.session_state.prioritized_products = df

    # --- KPI ROW ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products", f"{len(df):,}")
    col2.metric("Avg Feasibility", f"{df['feasibility_score'].mean():.1f}")
    col3.metric("Avg Attractiveness", f"{df['attractiveness_score'].mean():.1f}")
    col4.metric("Avg Composite", f"{df['composite_score'].mean():.1f}")

    # --- SCATTER: FEASIBILITY VS ATTRACTIVENESS ---
    st.markdown("### Feasibility vs. Attractiveness")

    # Prepare color for multidimensional
    plot_df_m, color_col_m = prepare_color(df, color_multi)
    ckwargs_m = color_map_for(color_multi, plot_df_m, color_col_m)

    plot_df_m["_size"] = plot_df_m["global_export_value"].clip(lower=0).fillna(0) if "global_export_value" in plot_df_m.columns else 1

    fig = px.scatter(
        plot_df_m, x="feasibility_score", y="attractiveness_score",
        size="_size", size_max=30,
        color=color_col_m,
        **ckwargs_m,
        hover_name="description" if "description" in plot_df_m.columns else None,
        hover_data={c: True for c in ["hs_product_code", "hs2_name", "quadrant", "composite_score"] if c in plot_df_m.columns},
        title="Feasibility vs. Attractiveness",
    )

    fig.add_vline(x=feas_med, line_dash="dash", line_color="grey", opacity=0.5)
    fig.add_hline(y=attr_med, line_dash="dash", line_color="grey", opacity=0.5)

    # Quadrant labels
    quadrant_label_positions = [
        ("Strategic Bets", feas_med * 0.5, attr_med + (100 - attr_med) * 0.5),
        ("Top Priorities", feas_med + (100 - feas_med) * 0.5, attr_med + (100 - attr_med) * 0.5),
        ("Deprioritize", feas_med * 0.5, attr_med * 0.5),
        ("Low-Hanging Fruit", feas_med + (100 - feas_med) * 0.5, attr_med * 0.5),
    ]
    for qlabel, xpos, ypos in quadrant_label_positions:
        fig.add_annotation(
            x=xpos, y=ypos, text=f"<b>{qlabel}</b>",
            showarrow=False, font=dict(size=13, color="grey"), opacity=0.5,
        )

    # Label top products
    if label_top_multi > 0:
        top_prods = df.nlargest(label_top_multi, "composite_score")
        for _, row in top_prods.iterrows():
            desc = str(row.get("description", ""))[:35]
            fig.add_annotation(
                x=row["feasibility_score"], y=row["attractiveness_score"],
                text=desc, showarrow=True, arrowhead=2, font=dict(size=9),
            )

    fig.update_layout(
        xaxis_title="Feasibility Score",
        yaxis_title="Attractiveness Score",
        template=GL_TEMPLATE,
        height=600,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- TABS ---
    tab_table, tab_feas, tab_attr = st.tabs(
        ["Ranked Table", "Feasibility Breakdown", "Attractiveness Breakdown"]
    )

    with tab_table:
        display_cols = [
            "hs_product_code", "description", "hs2_name", "quadrant",
            "composite_score", "feasibility_score", "attractiveness_score",
            "rca_pctile", "density_pctile", "hhi_pctile", "distance_pctile",
            "market_size_pctile", "growth_pctile", "cog_pctile",
            "pci_pctile", "spillover_pctile",
            "global_export_value", "rca_mar", "pci",
        ]
        display_cols = [c for c in display_cols if c in df.columns]

        ranked = df[display_cols].sort_values("composite_score", ascending=False).reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked.index.name = "Rank"

        st.dataframe(ranked.head(500), use_container_width=True, height=500)
        st.markdown(f"**{len(df)} products** ranked")
        download_csv(df, "powershoring_prioritized_products.csv",
                     f"Multidimensional: F(rca={f_rca},density={f_density},hhi={f_hhi},distance={f_dist}) "
                     f"A(market={a_market},growth={a_growth},cog={a_cog},pci={a_pci},spillover={a_spillover})")

    # --- FEASIBILITY BREAKDOWN ---
    with tab_feas:
        st.markdown("**Feasibility Score Composition — Top 30**")
        top30 = df.nlargest(30, "feasibility_score")
        feas_labels = {"rca_pctile": "Morocco RCA", "density_pctile": "Morocco Density", "hhi_pctile": "Market Fragmentation", "distance_pctile": "Avg. Trade Distance"}
        total_fw = sum(feas_weights.values()) or 1

        bar_data = []
        for _, row in top30.iterrows():
            label = f"{row.get('hs_product_code', '')} — {str(row.get('description', ''))[:45]}"
            for comp_name, comp_label in feas_labels.items():
                bar_data.append({
                    "product": label,
                    "component": comp_label,
                    "contribution": row[comp_name] * feas_weights[comp_name] / total_fw,
                })

        bar_df = pd.DataFrame(bar_data)
        product_order = top30.sort_values("feasibility_score", ascending=True).apply(
            lambda r: f"{r.get('hs_product_code', '')} — {str(r.get('description', ''))[:45]}", axis=1
        ).tolist()

        fig_f = px.bar(
            bar_df, x="contribution", y="product", color="component",
            orientation="h", color_discrete_sequence=GL_PALETTE_EXT[:4],
            category_orders={"product": product_order},
        )
        fig_f.update_layout(
            template=GL_TEMPLATE, height=max(500, 25 * len(top30)),
            xaxis_title="Weighted Score", yaxis_title="",
            margin=dict(l=350),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig_f, use_container_width=True)

    # --- ATTRACTIVENESS BREAKDOWN ---
    with tab_attr:
        st.markdown("**Attractiveness Score Composition — Top 30**")
        top30a = df.nlargest(30, "attractiveness_score")
        attr_labels = {
            "market_size_pctile": "Market Size",
            "growth_pctile": "Market Growth",
            "cog_pctile": "COG",
            "pci_pctile": "Complexity (PCI)",
            "spillover_pctile": "Spillover Potential",
        }
        total_aw = sum(attr_weights.values()) or 1

        bar_data = []
        for _, row in top30a.iterrows():
            label = f"{row.get('hs_product_code', '')} — {str(row.get('description', ''))[:45]}"
            for comp_name, comp_label in attr_labels.items():
                bar_data.append({
                    "product": label,
                    "component": comp_label,
                    "contribution": row[comp_name] * attr_weights[comp_name] / total_aw,
                })

        bar_df = pd.DataFrame(bar_data)
        product_order = top30a.sort_values("attractiveness_score", ascending=True).apply(
            lambda r: f"{r.get('hs_product_code', '')} — {str(r.get('description', ''))[:45]}", axis=1
        ).tolist()

        fig_a = px.bar(
            bar_df, x="contribution", y="product", color="component",
            orientation="h", color_discrete_sequence=GL_PALETTE_EXT[:5],
            category_orders={"product": product_order},
        )
        fig_a.update_layout(
            template=GL_TEMPLATE, height=max(500, 25 * len(top30a)),
            xaxis_title="Weighted Score", yaxis_title="",
            margin=dict(l=350),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig_a, use_container_width=True)
