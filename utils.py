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
