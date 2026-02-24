"""Shared utilities for the Powershoring interactive tool."""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ============================================================
# PATHS
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MASTER_DATA = os.path.join(DATA_DIR, "master_product_data.parquet")
HS4_DESCRIPTIONS = os.path.join(DATA_DIR, "hs4_descriptions.csv")

# ============================================================
# GROWTH LAB COLOR PALETTE
# ============================================================
GL_PALETTE = [
    "#c22229", "#204B82", "#8AB920", "#CB6843",
    "#157972", "#9E1C66", "#FA777E", "#A8DADC",
]
GL_PALETTE_EXT = GL_PALETTE + ["#FFC857", "#119DA4", "#E76F51", "#6A0572"]

MOROCCO_RED = "#c22229"
GREY = "#c3c7c5"

# Named variable labels for display
VARIABLE_LABELS = {
    "amount_carriers": "Energy Intensity (Carriers, MJ/$)",
    "amount_electric_energy": "Electricity Intensity (MJ/$)",
    "amount_fuel_energy": "Fuel Energy Intensity (MJ/$)",
    "amount_scope3": "Scope 3 Energy (MJ/$)",
    "electricity_share": "Electricity Share (%)",
    "global_export_value": "Global Export Value ($)",
    "export_cagr_2012_2023": "Export CAGR 2012-2023",
    "n_exporting_countries": "# Exporting Countries",
    "rca_mar": "Morocco RCA",
    "density_mar": "Morocco Density",
    "cog_mar": "Morocco COG",
    "value_mar": "Morocco Export Value ($)",
    "pci": "Product Complexity Index",
    "vulnerability_score": "Incumbent Vulnerability",
    "cbam_flag": "CBAM Covered",
    "cbam_score": "CBAM Score",
    "eu_import_share": "EU Import Share",
    "inv_hhi": "Market Fragmentation (1/HHI)",
    "hhi": "Market Concentration (HHI)",
    "weighted_degree": "Spillover Potential (Centrality)",
    "weighted_degree_quintile": "Centrality Quintile",
    "eigenvector_centrality": "Eigenvector Centrality",
    "product_distance": "Avg. Trade Distance (km)",
}


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load the master product dataset."""
    df = pd.read_parquet(MASTER_DATA)
    return df


# ============================================================
# PLOTLY THEME
# ============================================================
def get_gl_template():
    """Return a Plotly template matching Growth Lab style."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Source Sans Pro, sans-serif", size=14),
        colorway=GL_PALETTE_EXT,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True, linecolor="black"),
        yaxis=dict(showgrid=True, gridcolor="#eee", showline=True, linecolor="black"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return template


GL_TEMPLATE = get_gl_template()


# ============================================================
# CHART HELPERS
# ============================================================
def make_treemap(df, value_col, label_col="hs2_name", color_col="hs2_code", title="",
                 color_map=None):
    """Create an interactive Plotly treemap at HS2 level.

    Args:
        color_map: Optional dict mapping hs2_name → color hex for consistent coloring
                   across multiple treemaps.
    """
    agg = df.groupby(["hs2_code", "hs2_name"]).agg(
        value=(value_col, "sum"),
        n_products=("hs_product_code", "count"),
    ).reset_index()
    agg = agg.sort_values("value", ascending=False)

    color_kwargs = {}
    if color_map:
        color_kwargs["color_discrete_map"] = color_map
    else:
        color_kwargs["color_discrete_sequence"] = GL_PALETTE_EXT

    fig = px.treemap(
        agg,
        path=["hs2_name"],
        values="value",
        color="hs2_name",
        **color_kwargs,
        title=title,
        custom_data=["hs2_code", "n_products"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>HS2: %{customdata[0]}<br>Products: %{customdata[1]}<extra></extra>",
        textinfo="label+percent root",
    )
    fig.update_layout(
        template=GL_TEMPLATE,
        height=500,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


def make_bar_chart(df, value_col, label_col="description", n=20, title=""):
    """Create a horizontal bar chart of top N products."""
    top = df.nlargest(n, value_col).sort_values(value_col, ascending=True)

    fig = px.bar(
        top,
        x=value_col,
        y=label_col,
        orientation="h",
        title=title,
        color_discrete_sequence=[MOROCCO_RED],
    )
    fig.update_layout(
        template=GL_TEMPLATE,
        height=max(400, n * 25),
        yaxis_title="",
        xaxis_title=VARIABLE_LABELS.get(value_col, value_col),
        margin=dict(l=300),
    )
    return fig


def make_scatter(df, x_col, y_col, size_col=None, color_col=None,
                 hover_cols=None, title="", threshold_x=None, threshold_y=None):
    """Create an interactive scatter plot."""
    plot_df = df.copy()

    size_arg = {}
    if size_col and size_col != "uniform":
        plot_df["_size"] = plot_df[size_col].clip(lower=0).fillna(0)
        size_arg = dict(size="_size", size_max=30)

    color_arg = {}
    if color_col:
        color_arg = dict(color=color_col, color_discrete_sequence=GL_PALETTE_EXT)

    hover_data = {c: True for c in (hover_cols or ["description", "hs2_name"])}

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        **size_arg,
        **color_arg,
        hover_name="description" if "description" in plot_df.columns else None,
        hover_data=hover_data,
        title=title,
    )

    if threshold_x is not None:
        fig.add_vline(x=threshold_x, line_dash="dash", line_color="grey", opacity=0.5)
    if threshold_y is not None:
        fig.add_hline(y=threshold_y, line_dash="dash", line_color="grey", opacity=0.5)

    fig.update_layout(
        template=GL_TEMPLATE,
        height=550,
        xaxis_title=VARIABLE_LABELS.get(x_col, x_col),
        yaxis_title=VARIABLE_LABELS.get(y_col, y_col),
    )
    return fig


# ============================================================
# SCORING HELPERS
# ============================================================
def percentile_rank(series):
    """Convert a series to 0-100 percentile ranks."""
    return series.rank(pct=True, na_option="bottom") * 100


def weighted_score(df, components, weights):
    """Compute weighted composite score from percentile-ranked components."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return pd.Series(0, index=df.index)
    score = sum(weights[k] * components[k] for k in components if k in weights)
    return score / total_weight


# ============================================================
# DOWNLOAD HELPER
# ============================================================
def download_csv(df, filename, filter_description=""):
    """Provide a CSV download button."""
    export = df.copy()
    if filter_description:
        export["filter_description"] = filter_description
    csv = export.to_csv(index=False)
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def format_dollars(value):
    """Format dollar values for display."""
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"


# ============================================================
# SCENARIO DEFINITIONS & SCORING
# ============================================================
SCENARIO_DEFS = {
    "No Prior": {
        "weights": {"carriers": 25, "elec": 25, "vuln": 25, "cbam": 25, "growth": 0},
        "pre_filter": None,
        "desc": "Equal weight baseline — no assumption about which driver matters most.",
    },
    "Electricity Intensity": {
        "weights": {"carriers": 0, "elec": 100, "vuln": 0, "cbam": 0, "growth": 0},
        "pre_filter": None,
        "desc": "Pure electricity cost mechanism — industries with highest electricity share of costs.",
    },
    "Fuel + Electricity": {
        "weights": {"carriers": 50, "elec": 50, "vuln": 0, "cbam": 0, "growth": 0},
        "pre_filter": None,
        "desc": "Both fuel and electricity intensity — captures current and emerging electrification.",
    },
    "CBAM Dominant": {
        "weights": {"carriers": 0, "elec": 0, "vuln": 0, "cbam": 100, "growth": 0},
        "pre_filter": "cbam",
        "desc": "EU regulatory pressure — pre-filtered to CBAM-covered products only.",
    },
    "Disruption Opportunity": {
        "weights": {"carriers": 0, "elec": 0, "vuln": 100, "cbam": 0, "growth": 0},
        "pre_filter": None,
        "desc": "Incumbent vulnerability — where current exporters are most energy-deficit.",
    },
    "Market Growth": {
        "weights": {"carriers": 0, "elec": 0, "vuln": 0, "cbam": 0, "growth": 100},
        "pre_filter": None,
        "desc": "Fastest-growing global trade — expanding markets with easier entry.",
    },
}

DEFAULT_FEAS_WEIGHTS = {"rca": 20, "density": 50, "hhi": 15, "distance": 15}
DEFAULT_ATTR_WEIGHTS = {"market_size": 15, "growth": 15, "cog": 30, "pci": 30, "spillover": 10}


def run_scenario_scoring(df, likelihood_weights, feas_weights, attr_weights):
    """Run full pipeline: likelihood → top 50% filter → F/A scoring.

    Args:
        df: Filtered product DataFrame (Stage 1 output).
        likelihood_weights: Dict with keys carriers, elec, vuln, cbam, growth.
        feas_weights: Dict with keys rca, density, hhi, distance.
        attr_weights: Dict with keys market_size, growth, cog, pci, spillover.

    Returns:
        Scored DataFrame with likelihood, feasibility, attractiveness, lhf, sb scores
        and all component percentiles.
    """
    d = df.copy()

    # Likelihood components
    comps = {}
    comps["carriers"] = percentile_rank(d["amount_carriers"].fillna(0))
    comps["elec"] = percentile_rank(d["amount_electric_energy"].fillna(0))
    comps["vuln"] = (
        percentile_rank(-d["vulnerability_score"].fillna(0))
        if "vulnerability_score" in d.columns
        else pd.Series(50, index=d.index)
    )
    comps["cbam"] = (
        percentile_rank(d["cbam_score"].fillna(0))
        if "cbam_score" in d.columns
        else pd.Series(50, index=d.index)
    )
    comps["growth"] = percentile_rank(d["export_cagr_2012_2023"].fillna(0))

    active = {k: v for k, v in likelihood_weights.items() if v > 0}
    if not active:
        d["likelihood_score"] = 50.0
    else:
        total = sum(active.values())
        d["likelihood_score"] = sum(active[k] * comps[k] for k in active) / total

    # Top 50% by likelihood
    cutoff = d["likelihood_score"].quantile(0.5)
    sel = d[d["likelihood_score"] >= cutoff].copy()

    # Feasibility
    fc = {}
    fc["rca"] = percentile_rank(sel["rca_mar"].fillna(0))
    fc["density"] = percentile_rank(sel["density_mar"].fillna(0))
    fc["hhi"] = percentile_rank(sel["inv_hhi"].fillna(0))
    fc["distance"] = percentile_rank(-sel["product_distance"].fillna(sel["product_distance"].median()))
    ftotal = sum(feas_weights.values())
    sel["feasibility_score"] = sum(feas_weights[k] * fc[k] for k in feas_weights) / ftotal
    for k, v in fc.items():
        sel[f"feas_{k}"] = v

    # Attractiveness
    ac = {}
    ac["market_size"] = percentile_rank(sel["global_export_value"].fillna(0))
    ac["growth"] = percentile_rank(sel["export_cagr_2012_2023"].fillna(0))
    ac["cog"] = percentile_rank(sel["cog_mar"].fillna(0))
    ac["pci"] = percentile_rank(sel["pci"].fillna(0))
    ac["spillover"] = percentile_rank(sel["weighted_degree"].fillna(0))
    atotal = sum(attr_weights.values())
    sel["attractiveness_score"] = sum(attr_weights[k] * ac[k] for k in attr_weights) / atotal
    for k, v in ac.items():
        sel[f"attr_{k}"] = v

    # Composite score: 60% Feasibility + 40% Attractiveness
    sel["lhf_score"] = 0.60 * sel["feasibility_score"] + 0.40 * sel["attractiveness_score"]

    return sel


def _load_hs4_descriptions():
    """Load HS4 description lookup from CSV."""
    if os.path.exists(HS4_DESCRIPTIONS):
        hs4_df = pd.read_csv(HS4_DESCRIPTIONS, dtype=str)
        return dict(zip(hs4_df["hs4_code"], hs4_df["hs4_description"]))
    return {}

_HS4_DESC_LOOKUP = _load_hs4_descriptions()


def aggregate_to_hs4(sel):
    """Aggregate scored HS6 products to HS4 level using trade-weighted averages.

    Returns DataFrame with one row per HS4 code.
    """
    sel = sel.copy()
    sel["hs4_code"] = sel["hs_product_code"].astype(str).str.zfill(6).str[:4]

    score_cols = [c for c in sel.columns if c.endswith("_score") or c.startswith("feas_") or c.startswith("attr_")]

    def _tw_mean(g, col):
        w = g["global_export_value"].fillna(0)
        return np.average(g[col], weights=w) if w.sum() > 0 else g[col].mean()

    rows = []
    for (hs4, hs2n), g in sel.groupby(["hs4_code", "hs2_name"]):
        desc = _HS4_DESC_LOOKUP.get(hs4, "")
        if not desc:
            desc = f"{hs2n} ({hs4})"
        r = {
            "hs4_code": hs4, "hs2_name": hs2n, "description": desc,
            "n_products": len(g), "global_export_value": g["global_export_value"].sum(),
            "rca_mar": g["rca_mar"].mean(), "density_mar": g["density_mar"].mean(),
            "elec_int": _tw_mean(g, "amount_electric_energy"),
            "cbam_flag": g["cbam_flag"].max() if "cbam_flag" in g.columns else 0,
            "green_flag": g["green_supply_chain_flag"].max() if "green_supply_chain_flag" in g.columns else 0,
        }
        for col in score_cols:
            if col in g.columns:
                r[col] = _tw_mean(g, col)
        rows.append(r)

    return pd.DataFrame(rows)
