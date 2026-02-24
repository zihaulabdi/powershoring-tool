"""Stage 2b: Prioritization — Rank products by feasibility and attractiveness."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, download_csv, format_dollars,
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

COLOR_VARS = ["top_n", "hs2_name", "cbam_flag", "green_supply_chain_flag", "green_topic"]
COLOR_VARS = [v for v in COLOR_VARS if v in df.columns or v == "top_n"]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Feasibility vs Attractiveness")
    feas_pct = st.slider("Feasibility weight (%)", 0, 100, 60, 5, key="fa_balance",
                         help="Remaining weight goes to attractiveness")
    attr_pct = 100 - feas_pct
    st.caption(f"**{feas_pct}% Feasibility / {attr_pct}% Attractiveness**")

    st.divider()
    st.header("Feasibility Weights")
    f_rca = st.slider("Morocco RCA", 0, 100, 20, key="f_rca",
                      help="Revealed comparative advantage — existing export strength")
    f_density = st.slider("Morocco Density", 0, 100, 50, key="f_density",
                          help="Proximity to existing capabilities in product space")
    f_hhi = st.slider("Market fragmentation (1/HHI)", 0, 100, 15, key="f_hhi",
                      help="Lower concentration = easier to enter")
    f_dist = st.slider("Avg. Trade Distance (km)", 0, 100, 15, key="f_dist",
                       help="Short-distance products are more transport-sensitive — better powershoring candidates")

    st.divider()
    st.header("Attractiveness Weights")
    a_market = st.slider("Market size", 0, 100, 15, key="a_market",
                         help="Global export value")
    a_growth = st.slider("Market growth", 0, 100, 15, key="a_growth",
                         help="Export CAGR 2012-2023")
    a_cog = st.slider("COG (opportunity gain)", 0, 100, 30, key="a_cog",
                      help="Contribution to future diversification")
    a_pci = st.slider("Product complexity", 0, 100, 30, key="a_pci",
                      help="Higher complexity = more value added")
    a_spillover = st.slider("Spillover potential", 0, 100, 10, key="a_spillover",
                            help="Network centrality — benefits to related industries")

    st.divider()
    _color_labels = {**VARIABLE_LABELS, "top_n": "Top N (by composite score)"}
    color_var = st.selectbox("Color by", COLOR_VARS, key="color_multi",
                             format_func=lambda x: _color_labels.get(x, x))
    label_top = st.slider("Label top N", 0, 50, 30, key="prio_label_multi")


# ============================================================
# HELPER: prepare color column for categorical flags
# ============================================================
def prepare_color(plot_df, col, top_n_count=30):
    """Return (df_copy, color_col_name) with proper categorical handling."""
    out = plot_df.copy()
    if col == "top_n":
        top_codes = set(out.nlargest(top_n_count, "composite_score").index)
        out["_color"] = out.index.map(lambda i: f"Top {top_n_count}" if i in top_codes else "Other")
        return out, "_color"
    elif col == "cbam_flag":
        out["_color"] = out[col].map({1: "CBAM", 0: "Not CBAM"}).fillna("Not CBAM")
        return out, "_color"
    elif col == "green_supply_chain_flag":
        out["_color"] = out[col].map({1: "Green SC", 0: "Not Green SC"}).fillna("Not Green SC")
        return out, "_color"
    elif col == "green_topic":
        out["_color"] = out[col].fillna("").replace("", "No green topic")
        return out, "_color"
    return out, col


def color_map_for(col, plot_df, color_col, top_n_count=30):
    """Return color_discrete_map or color_discrete_sequence kwargs."""
    if col == "top_n":
        return dict(color_discrete_map={f"Top {top_n_count}": MOROCCO_RED, "Other": GREY})
    elif col == "cbam_flag":
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
# COMPUTE SCORES
# ============================================================
# Feasibility
feas_weights = {"rca_pctile": f_rca, "density_pctile": f_density, "hhi_pctile": f_hhi, "distance_pctile": f_dist}
feas_components = {}
feas_components["rca_pctile"] = percentile_rank(df["rca_mar"].fillna(0)) if "rca_mar" in df.columns else pd.Series(50, index=df.index)
feas_components["density_pctile"] = percentile_rank(df["density_mar"].fillna(0)) if "density_mar" in df.columns else pd.Series(50, index=df.index)
feas_components["hhi_pctile"] = percentile_rank(df["inv_hhi"].fillna(0)) if "inv_hhi" in df.columns else pd.Series(50, index=df.index)
feas_components["distance_pctile"] = percentile_rank(-df["product_distance"].fillna(df["product_distance"].max() or 0)) if "product_distance" in df.columns else pd.Series(50, index=df.index)

df["feasibility_score"] = weighted_score(df, feas_components, feas_weights)

# Attractiveness
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

# Composite score based on slider balance
df["composite_score"] = (feas_pct / 100) * df["feasibility_score"] + (attr_pct / 100) * df["attractiveness_score"]

# Store component percentiles
for k, v in feas_components.items():
    df[k] = v
for k, v in attr_components.items():
    df[k] = v

# Save to session state
st.session_state.prioritized_products = df

# ============================================================
# KPI ROW
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Products", f"{len(df):,}")
col2.metric("Avg Feasibility", f"{df['feasibility_score'].mean():.1f}")
col3.metric("Avg Attractiveness", f"{df['attractiveness_score'].mean():.1f}")
col4.metric(f"Avg Composite ({feas_pct}F/{attr_pct}A)", f"{df['composite_score'].mean():.1f}")

# ============================================================
# SCATTER: FEASIBILITY VS ATTRACTIVENESS
# ============================================================
st.markdown("### Feasibility vs. Attractiveness")

plot_df, color_col = prepare_color(df, color_var, top_n_count=label_top)
ckwargs = color_map_for(color_var, plot_df, color_col, top_n_count=label_top)

plot_df["_size"] = plot_df["global_export_value"].clip(lower=0).fillna(0) if "global_export_value" in plot_df.columns else 1

# Category order: render "Other" first so Top N dots appear on top
cat_order = {}
if color_var == "top_n":
    cat_order = {color_col: ["Other", f"Top {label_top}"]}

fig = px.scatter(
    plot_df, x="feasibility_score", y="attractiveness_score",
    size="_size", size_max=30,
    color=color_col,
    **ckwargs,
    category_orders=cat_order,
    hover_name="description" if "description" in plot_df.columns else None,
    hover_data={c: True for c in ["hs_product_code", "hs2_name", "composite_score"] if c in plot_df.columns},
    title=f"Feasibility vs. Attractiveness ({feas_pct}F / {attr_pct}A)",
)

# Label top products
if label_top > 0:
    top_prods = df.nlargest(label_top, "composite_score")
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

# ============================================================
# TABS
# ============================================================
tab_table, tab_feas, tab_attr = st.tabs(
    ["Ranked Table", "Feasibility Breakdown", "Attractiveness Breakdown"]
)

with tab_table:
    display_cols = [
        "hs_product_code", "description", "hs2_name",
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
                 f"Balance: {feas_pct}F/{attr_pct}A | "
                 f"F(rca={f_rca},density={f_density},hhi={f_hhi},distance={f_dist}) "
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
