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
CODE_DIR = os.path.dirname(APP_DIR)
PROJECT_DIR = os.path.dirname(CODE_DIR)

# The same utils.py is used in two places:
#   1. Local development: Powershoring/02_Code/app/
#   2. Deployment repo:   /Users/zia226/powershoring-tool/
# Prefer the local project data when present; otherwise fall back to the
# deployed app's data/ folder. This avoids hand-editing path logic during sync.
LOCAL_DATA_DIR = os.path.join(PROJECT_DIR, "01_Data")
DEPLOY_DATA_DIR = os.path.join(APP_DIR, "data")
DATA_DIR = LOCAL_DATA_DIR if os.path.exists(os.path.join(LOCAL_DATA_DIR, "master_product_data.parquet")) else DEPLOY_DATA_DIR
MASTER_DATA = os.path.join(DATA_DIR, "master_product_data.parquet")
HS4_DESCRIPTIONS = os.path.join(DATA_DIR, "hs4_descriptions.csv")

METHODOLOGY_VERSION = "2026-04-28_four_scenario_fixed_fa_universe_v1"
FA_PERCENTILE_UNIVERSE = "stage1_candidate_pool"

DEFAULT_ENERGY_PERCENTILE = 0.75
DEFAULT_ELEC_PERCENTILE = 0.50
DEFAULT_TRADE_PERCENTILE = 0.15
DEFAULT_FILTER_LOGIC = "OR"

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
    "export_cagr_2012_2023": "Export CAGR 2012-2024",
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


def _normalize_percentile(value):
    """Accept either 0-1 or 0-100 percentile inputs and return 0-1."""
    return value / 100 if value > 1 else value


def _energy_threshold(series, percentile):
    """Compute an energy threshold on observed values; 0th percentile is 0.

    Missing energy values mean "no USEEIO mapping", not observed zero use. We
    therefore compute positive thresholds on non-missing values, but treat
    missing values as zero in pass/fail masks. At the 0th percentile, the
    threshold is forced to zero so missing-energy products are visible.
    """
    pct = _normalize_percentile(percentile)
    if pct <= 0:
        return 0.0
    observed = series.dropna()
    if observed.empty:
        return 0.0
    return float(observed.quantile(pct))


def get_stage1_thresholds(
    df_all,
    energy_percentile=DEFAULT_ENERGY_PERCENTILE,
    elec_percentile=DEFAULT_ELEC_PERCENTILE,
    trade_percentile=DEFAULT_TRADE_PERCENTILE,
):
    """Return canonical Stage 1 threshold values."""
    return {
        "energy_percentile": _normalize_percentile(energy_percentile),
        "elec_percentile": _normalize_percentile(elec_percentile),
        "trade_percentile": _normalize_percentile(trade_percentile),
        "energy_threshold": _energy_threshold(df_all["amount_carriers"], energy_percentile),
        "elec_threshold": _energy_threshold(df_all["amount_electric_energy"], elec_percentile),
        "trade_threshold": float(df_all["global_export_value"].quantile(_normalize_percentile(trade_percentile))),
        "missing_energy_policy": "thresholds_on_observed_values__missing_as_zero_for_pass_fail",
    }


def build_stage1_energy_mask(df, thresholds, filter_logic=DEFAULT_FILTER_LOGIC):
    """Build the canonical Stage 1 energy pass/fail mask."""
    carriers = df["amount_carriers"].fillna(0)
    elec = df["amount_electric_energy"].fillna(0)
    carrier_pass = carriers >= thresholds["energy_threshold"]
    elec_pass = elec >= thresholds["elec_threshold"]
    if str(filter_logic).upper() == "AND":
        return carrier_pass & elec_pass
    return carrier_pass | elec_pass


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


def inject_custom_css():
    """Inject custom CSS for typography and layout consistency."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, p, label, .stSelectbox, .stRadio {
        font-family: 'Source Sans Pro', sans-serif !important;
        line-height: 1.45;
    }
    h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.01em; }
    h2 { font-size: 1.35rem; font-weight: 600; }
    h3 { font-size: 1.1rem; font-weight: 600; }
    .stMarkdown p { font-size: 15px; line-height: 1.5; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SCENARIO DEFINITIONS & SCORING
# ============================================================
SCENARIO_DEFS = {
    "No Prior": {
        "weights": {"fuel": 0, "elec": 33, "vuln": 33, "cbam": 33, "growth": 0},
        "pre_filter": None,
        "desc": "Equal weight on electricity intensity, incumbent vulnerability, and CBAM exposure. No prior assumption about which relocation driver dominates.",
    },
    "Electricity Cost": {
        "weights": {"fuel": 0, "elec": 100, "vuln": 0, "cbam": 0, "growth": 0},
        "pre_filter": None,
        "desc": "Pure electricity cost mechanism. Selects industries where electricity is the largest share of production costs.",
    },
    "Carbon Regulation": {
        "weights": {"fuel": 0, "elec": 0, "vuln": 0, "cbam": 100, "growth": 0},
        "pre_filter": "cbam",
        "desc": "EU carbon border pressure. Pre-filtered to CBAM-covered products, ranked by combined energy intensity and EU market exposure.",
    },
    "Disruption Opportunity": {
        "weights": {"fuel": 0, "elec": 0, "vuln": 100, "cbam": 0, "growth": 0},
        "pre_filter": None,
        "desc": "Disruption opportunity. Where current top exporters are most energy-deficit and therefore most vulnerable to powershoring competition.",
    },
}

DEFAULT_FEAS_WEIGHTS = {"rca": 20, "density": 50, "hhi": 15, "distance": 15}
DEFAULT_ATTR_WEIGHTS = {"market_size": 15, "growth": 15, "cog": 30, "pci": 30, "spillover": 10}


# ============================================================
# DEFAULT STAGE 1 EXCLUSIONS (single source of truth)
# ============================================================
# Imported by both the interactive app and the batch chart pipeline so that
# the published figures, the chapter, and the live tool always agree on what
# the candidate universe is.
#
# Imported by:
#   - app/pages/1_Filtering.py    (sidebar toggle exposes legacy fallback)
#   - app/pages/5_Summary.py      (via _default_stage1_filter)
#   - 02_Code/scenario_analysis_simplified.py
#   - 02_Code/robustness_analysis.py

# Legacy default: HS 01-27 (non-manufacturing + extractive) -- preserved so
# old analyses can be reproduced exactly via the sidebar toggle.
EXCLUDE_HS2_LEGACY = list(range(1, 28))  # chapters 1..27 inclusive

# Current default: legacy plus chapters whose location is not driven by
# electricity cost.
#   HS 68 -- Articles of stone, cement, asbestos: heavy and local; energy is
#            upstream in already-excluded cement (HS 25).
#   HS 71 -- Precious stones and metals: location driven by vault security,
#            refining licenses, and craft skills, not industrial electricity.
EXCLUDE_HS2_DEFAULT = EXCLUDE_HS2_LEGACY + [68, 71]

# Raw agricultural fibers at the HS4 heading level. The HS classification
# puts these in textile chapters, but they are agricultural commodities --
# location is determined by climate and biology, not by industrial cost.
#   5001 silk-worm cocoons; 5101 wool, not carded or combed;
#   5201 cotton, not carded or combed; 5301 raw or retted flax.
EXCLUDE_HS4_RAW_FIBERS = [5001, 5101, 5201, 5301]


def apply_default_exclusions(df, use_legacy=False):
    """Apply Stage 1 chapter and raw-fiber exclusions.

    Args:
        df: DataFrame with `hs2_code` and `hs4_code` columns.
        use_legacy: If True, exclude only HS 01-27 (legacy behaviour, for
            reproducing pre-2026-04 analyses). If False (default), also
            exclude HS 68, HS 71, and raw fiber HS4 headings 5001/5101/5201/5301.

    Returns:
        Filtered DataFrame copy.
    """
    out = df.copy()
    out["hs2_code"] = out["hs2_code"].astype(int)
    out["hs4_code"] = out["hs4_code"].astype(int)

    excluded_hs2 = EXCLUDE_HS2_LEGACY if use_legacy else EXCLUDE_HS2_DEFAULT
    out = out[~out["hs2_code"].isin(excluded_hs2)]

    if not use_legacy:
        out = out[~out["hs4_code"].isin(EXCLUDE_HS4_RAW_FIBERS)]

    return out


def apply_stage1_filter(
    df_all,
    energy_percentile=DEFAULT_ENERGY_PERCENTILE,
    elec_percentile=DEFAULT_ELEC_PERCENTILE,
    trade_percentile=DEFAULT_TRADE_PERCENTILE,
    filter_logic=DEFAULT_FILTER_LOGIC,
    use_legacy=False,
    cbam_only=False,
    green_only=False,
    green_topics=None,
    rca_threshold=0.0,
    return_metadata=False,
):
    """Apply the canonical Stage 1 candidate-universe filter.

    Thresholds are computed on the full universe. Positive energy thresholds
    are computed on observed USEEIO values, while pass/fail masks use missing
    energy as zero. This keeps 0th-percentile controls inclusive without
    lowering non-zero energy thresholds because of unmapped products.
    """
    thresholds = get_stage1_thresholds(
        df_all,
        energy_percentile=energy_percentile,
        elec_percentile=elec_percentile,
        trade_percentile=trade_percentile,
    )

    filtered = apply_default_exclusions(df_all, use_legacy=use_legacy)
    energy_mask = build_stage1_energy_mask(filtered, thresholds, filter_logic=filter_logic)
    trade_mask = filtered["global_export_value"] >= thresholds["trade_threshold"]
    filtered = filtered[energy_mask & trade_mask].copy()

    filter_parts = []
    if use_legacy:
        filter_parts.append("Legacy exclusion (HS 01-27)")
    else:
        filter_parts.append("Default exclusion (HS 01-27, 68, 71 + raw fibers)")
    filter_parts.append(
        f"Energy >= {thresholds['energy_percentile']*100:.0f}th pct "
        f"{str(filter_logic).upper()} Electricity >= {thresholds['elec_percentile']*100:.0f}th pct"
    )
    filter_parts.append(f"Trade >= {format_dollars(thresholds['trade_threshold'])} ({thresholds['trade_percentile']*100:.0f}th pct)")

    if cbam_only:
        filtered = filtered[filtered["cbam_flag"] == 1].copy()
        filter_parts.append("CBAM only")
    if green_only and green_topics:
        filtered = filtered[filtered["green_supply_chain_flag"] == 1].copy()
        filtered = filtered[filtered["green_topic"].apply(lambda x: any(t in str(x) for t in green_topics))].copy()
        filter_parts.append(f"Green SC: {', '.join(green_topics)}")
    if rca_threshold and rca_threshold > 0:
        filtered = filtered[filtered["rca_mar"] >= rca_threshold].copy()
        filter_parts.append(f"Morocco RCA >= {rca_threshold}")

    metadata = {
        "methodology_version": METHODOLOGY_VERSION,
        "thresholds": thresholds,
        "filter_logic": str(filter_logic).upper(),
        "use_legacy_exclusions": bool(use_legacy),
        "description": " | ".join(filter_parts),
        "n_products": int(len(filtered)),
        "global_export_value": float(filtered["global_export_value"].sum()),
    }
    if return_metadata:
        return filtered, metadata
    return filtered


def add_likelihood_scores(df, likelihood_weights):
    """Add scenario-specific likelihood score and component percentiles."""
    d = df.copy()

    comps = {}
    comps["fuel"] = percentile_rank(d["amount_fuel_energy"].fillna(0))
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

    for k, v in comps.items():
        d[f"like_{k}"] = v

    return d


def add_feasibility_attractiveness_scores(df, feas_weights, attr_weights, reference_df=None):
    """Add F/A scores using a fixed reference universe for percentile ranks.

    For cross-scenario work, `reference_df` should be the full Stage 1
    candidate pool. Scenario-specific likelihood filters then gate products,
    but F/A ranks stay comparable across scenarios.
    """
    out = df.copy()
    ref = out if reference_df is None else reference_df.copy()

    # If a caller passes rows outside the reference universe, include them so
    # reindexing below cannot silently create NaN ranks.
    missing_idx = out.index.difference(ref.index)
    if len(missing_idx) > 0:
        ref = pd.concat([ref, out.loc[missing_idx]], axis=0)

    fc = {}
    fc["rca"] = percentile_rank(ref["rca_mar"].fillna(0))
    fc["density"] = percentile_rank(ref["density_mar"].fillna(0))
    fc["hhi"] = percentile_rank(ref["inv_hhi"].fillna(0))
    fc["distance"] = percentile_rank(-ref["product_distance"].fillna(ref["product_distance"].median()))

    ftotal = sum(feas_weights.values()) or 1
    for k, v in fc.items():
        out[f"feas_{k}"] = v.reindex(out.index)
    out["feasibility_score"] = sum(feas_weights[k] * out[f"feas_{k}"] for k in feas_weights) / ftotal

    ac = {}
    ac["market_size"] = percentile_rank(ref["global_export_value"].fillna(0))
    ac["growth"] = percentile_rank(ref["export_cagr_2012_2023"].fillna(0))
    ac["cog"] = percentile_rank(ref["cog_mar"].fillna(0))
    ac["pci"] = percentile_rank(ref["pci"].fillna(0))
    ac["spillover"] = percentile_rank(ref["weighted_degree"].fillna(ref["weighted_degree"].median()))

    atotal = sum(attr_weights.values()) or 1
    for k, v in ac.items():
        out[f"attr_{k}"] = v.reindex(out.index)
    out["attractiveness_score"] = sum(attr_weights[k] * out[f"attr_{k}"] for k in attr_weights) / atotal

    # Backward-compatible aliases for the combined app page/table.
    out["rca_pctile"] = out["feas_rca"]
    out["density_pctile"] = out["feas_density"]
    out["hhi_pctile"] = out["feas_hhi"]
    out["distance_pctile"] = out["feas_distance"]
    out["market_size_pctile"] = out["attr_market_size"]
    out["attr_growth_pctile"] = out["attr_growth"]
    out["cog_pctile"] = out["attr_cog"]
    out["pci_pctile"] = out["attr_pci"]
    out["spillover_pctile"] = out["attr_spillover"]

    return out


def run_scenario_scoring(df, likelihood_weights, feas_weights, attr_weights, fa_reference_df=None):
    """Run full pipeline: likelihood → top 50% filter → F/A scoring.

    Args:
        df: Filtered product DataFrame (Stage 1 output).
        likelihood_weights: Dict with keys fuel, elec, vuln, cbam, growth.
        feas_weights: Dict with keys rca, density, hhi, distance.
        attr_weights: Dict with keys market_size, growth, cog, pci, spillover.
        fa_reference_df: Universe used for F/A percentile ranks. Pass the full
            Stage 1 pool for cross-scenario comparability.

    Returns:
        Scored DataFrame with likelihood, feasibility, attractiveness scores
        and all component percentiles.
    """
    d = add_likelihood_scores(df, likelihood_weights)
    d = add_feasibility_attractiveness_scores(
        d,
        feas_weights,
        attr_weights,
        reference_df=fa_reference_df if fa_reference_df is not None else d,
    )

    # Top 50% by likelihood
    cutoff = d["likelihood_score"].quantile(0.5)
    sel = d[d["likelihood_score"] >= cutoff].copy()
    return sel


def _load_hs4_descriptions():
    """Load HS4 description lookup from CSV."""
    if os.path.exists(HS4_DESCRIPTIONS):
        hs4_df = pd.read_csv(HS4_DESCRIPTIONS, dtype=str)
        return dict(zip(hs4_df["hs4_code"], hs4_df["hs4_description"]))
    return {}

_HS4_DESC_LOOKUP = _load_hs4_descriptions()


# ============================================================
# DEFAULT STAGE 1 FILTER
# ============================================================
def _default_stage1_filter(df_all, use_legacy=False):
    """Apply default Stage 1 filter to the full dataset.

    Thresholds: energy >= 75th pct OR electricity >= 50th pct, trade >= 15th pct.
    Exclusions are applied via `apply_default_exclusions`. By default this
    excludes HS 01-27 plus HS 68, HS 71, and raw fiber HS4 headings; pass
    `use_legacy=True` to fall back to HS 01-27 only.

    Returns raw filtered DataFrame (not scored).
    """
    return apply_stage1_filter(df_all, use_legacy=use_legacy)


def run_default_scenario(df_all):
    """Run No Prior scenario with default thresholds. Returns HS4-aggregated top 15."""
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
    return hs4.nlargest(15, "composite_score").reset_index(drop=True)


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
        name_short = desc[:40] if desc else f"{hs2n} ({hs4})"
        atlas_sect = g["atlas_section"].iloc[0] if "atlas_section" in g.columns else hs2n
        r = {
            "hs4_code": hs4, "hs2_name": hs2n, "atlas_section": atlas_sect,
            "description": desc, "name_short": name_short,
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
