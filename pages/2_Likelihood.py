"""Stage 2a: Likelihood — Score how likely products are to powershore."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_treemap,
    download_csv, format_dollars, percentile_rank, weighted_score,
)

st.set_page_config(page_title="Stage 2a: Likelihood", layout="wide")
st.title("Stage 2a: Likelihood")
st.markdown("Score products by how likely they are to powershore — energy intensity, incumbent vulnerability, and CBAM exposure.")

# ============================================================
# LOAD DATA
# ============================================================
df_all = load_data()

# Use filtered products from Stage 1 if available
if st.session_state.get("filtered_products") is not None:
    df = st.session_state.filtered_products.copy()
    st.success(f"Using **{len(df):,}** filtered products from Stage 1.")
else:
    df = df_all.copy()
    st.warning("No Stage 1 filtering applied — using full product universe. Go to **Stage 1: Filtering** first.")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("Likelihood Weights")
    st.markdown("Adjust weights for each likelihood dimension (will be normalized to 100%).")

    w_carriers = st.slider("Energy (Fuel+Electricity) Intensity", 0, 100, 25, key="w_carriers",
                           help="Higher = products with high total energy intensity (fuel + electricity) score higher")
    w_elec = st.slider("Electricity intensity", 0, 100, 25, key="w_elec",
                       help="Higher = products with high electricity usage per $ score higher")
    w_vuln = st.slider("Incumbent vulnerability", 0, 100, 25, key="w_vuln",
                       help="Higher = products where incumbents are energy-deficit score higher")
    w_cbam = st.slider("CBAM exposure", 0, 100, 25, key="w_cbam",
                       help="Higher = CBAM-covered products with EU exposure score higher")

    total_w = w_carriers + w_elec + w_vuln + w_cbam
    if total_w == 0:
        st.error("At least one weight must be > 0")
        st.stop()

    st.caption(f"**Effective weights:** Carriers {w_carriers/total_w*100:.0f}% | "
               f"Electricity {w_elec/total_w*100:.0f}% | "
               f"Vulnerability {w_vuln/total_w*100:.0f}% | "
               f"CBAM {w_cbam/total_w*100:.0f}%")

    st.divider()
    st.header("Selection Cutoff")
    selection_method = st.radio("Selection method", ["Top %", "Top N", "Score threshold"])

    if selection_method == "Top %":
        top_pct = st.slider("Top % of products", 10, 100, 50, 5)
    elif selection_method == "Top N":
        top_n = st.number_input("Top N products", 10, len(df), min(200, len(df)), 10)
    else:
        score_thresh = st.slider("Minimum likelihood score", 0.0, 100.0, 50.0, 1.0)

    st.divider()
    st.header("Save Scenario")
    scenario_name = st.text_input("Scenario name", "", key="likelihood_scenario_name")
    if st.button("Save current scenario") and scenario_name:
        if "saved_scenarios" not in st.session_state:
            st.session_state.saved_scenarios = {}
        # Will be populated after scores are computed below
        st.session_state["_pending_scenario_save"] = scenario_name

# ============================================================
# COMPUTE LIKELIHOOD SCORES
# ============================================================
weights = {
    "carriers_pctile": w_carriers,
    "elec_pctile": w_elec,
    "vulnerability_pctile": w_vuln,
    "cbam_pctile": w_cbam,
}

# Build component percentiles
components = {}
components["carriers_pctile"] = percentile_rank(df["amount_carriers"].fillna(0))
components["elec_pctile"] = percentile_rank(df["amount_electric_energy"].fillna(0))
# Vulnerability: more negative = more vulnerable for incumbents = better for powershoring
if "vulnerability_score" in df.columns:
    components["vulnerability_pctile"] = percentile_rank(-df["vulnerability_score"].fillna(0))
else:
    components["vulnerability_pctile"] = pd.Series(50, index=df.index)
if "cbam_score" in df.columns:
    components["cbam_pctile"] = percentile_rank(df["cbam_score"].fillna(0))
else:
    components["cbam_pctile"] = pd.Series(50, index=df.index)

df["likelihood_score"] = weighted_score(df, components, weights)
for k, v in components.items():
    df[k] = v

# Apply selection cutoff
if selection_method == "Top %":
    cutoff = df["likelihood_score"].quantile(1 - top_pct / 100)
    selected = df[df["likelihood_score"] >= cutoff]
elif selection_method == "Top N":
    selected = df.nlargest(top_n, "likelihood_score")
    cutoff = selected["likelihood_score"].min()
else:
    cutoff = score_thresh
    selected = df[df["likelihood_score"] >= cutoff]

# Save to session state
st.session_state.likelihood_products = selected

# Handle pending scenario save (button was pressed in sidebar before scores computed)
if st.session_state.get("_pending_scenario_save"):
    _sname = st.session_state.pop("_pending_scenario_save")
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = {}
    _weight_desc = (f"Energy(F+E) {w_carriers/total_w*100:.0f}%, "
                    f"Electricity {w_elec/total_w*100:.0f}%, "
                    f"Vulnerability {w_vuln/total_w*100:.0f}%, "
                    f"CBAM {w_cbam/total_w*100:.0f}%")
    st.session_state.saved_scenarios[_sname] = {
        "products": selected.copy(),
        "stage": "likelihood",
        "desc": f"Weights: {_weight_desc} | Cutoff: {cutoff:.1f} | {len(selected)} products",
    }
    st.success(f"Saved scenario: **{_sname}** ({len(selected)} products)")

# ============================================================
# KPI ROW
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Selected Products", f"{len(selected):,}")
col2.metric("% of Filtered", f"{100 * len(selected) / max(len(df), 1):.1f}%")
col3.metric("Avg Score", f"{selected['likelihood_score'].mean():.1f}")
col4.metric("Global Trade", format_dollars(selected["global_export_value"].sum()))
col5.metric("Score Cutoff", f"{cutoff:.1f}")

# ============================================================
# WEIGHT VISUALIZATION
# ============================================================
weight_labels = {
    "carriers_pctile": "Energy (Fuel+Electricity) Intensity",
    "elec_pctile": "Electricity Intensity",
    "vulnerability_pctile": "Incumbent Vulnerability",
    "cbam_pctile": "CBAM Exposure",
}
fig_w = go.Figure(go.Bar(
    x=[weights[k] / total_w * 100 for k in weights],
    y=[weight_labels[k] for k in weights],
    orientation="h",
    marker_color=[GL_PALETTE_EXT[i] for i in range(len(weights))],
    text=[f"{weights[k]/total_w*100:.0f}%" for k in weights],
    textposition="auto",
))
fig_w.update_layout(
    template=GL_TEMPLATE, height=200,
    xaxis_title="Weight (%)", yaxis_title="",
    margin=dict(t=10, b=30),
    showlegend=False,
)
st.plotly_chart(fig_w, use_container_width=True)

# ============================================================
# TABS
# ============================================================
tab_dist, tab_treemap, tab_table, tab_breakdown = st.tabs(
    ["Score Distribution", "Industry Treemap", "Ranked Table", "Component Breakdown"]
)

# --- SCORE DISTRIBUTION ---
with tab_dist:
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(
        x=df["likelihood_score"], name="All filtered",
        marker_color=GREY, opacity=0.6, nbinsx=50,
    ))
    fig_d.add_trace(go.Histogram(
        x=selected["likelihood_score"], name="Selected",
        marker_color=MOROCCO_RED, opacity=0.7, nbinsx=50,
    ))
    fig_d.add_vline(x=cutoff, line_dash="dash", line_color="black",
                    annotation_text=f"Cutoff: {cutoff:.1f}")
    fig_d.update_layout(
        template=GL_TEMPLATE, barmode="overlay", height=400,
        xaxis_title="Likelihood Score", yaxis_title="Count",
        title="Distribution of Likelihood Scores",
    )
    st.plotly_chart(fig_d, use_container_width=True)

# --- INDUSTRY TREEMAP (HS2 of selected products) ---
with tab_treemap:
    st.markdown("**What industries are selected after likelihood scoring?**")
    treemap_metric = st.radio(
        "Size by:", ["global_export_value", "amount_carriers", "amount_electric_energy"],
        format_func=lambda x: VARIABLE_LABELS.get(x, x),
        horizontal=True,
        key="likelihood_treemap_metric",
    )
    fig_tm = make_treemap(
        selected, treemap_metric,
        title=f"Selected Products by HS2 — {VARIABLE_LABELS.get(treemap_metric, treemap_metric)}",
    )
    st.plotly_chart(fig_tm, use_container_width=True)

# --- RANKED TABLE ---
with tab_table:
    display_cols = [
        "hs_product_code", "description", "hs2_name",
        "likelihood_score", "carriers_pctile", "elec_pctile",
        "vulnerability_pctile", "cbam_pctile",
        "global_export_value", "amount_electric_energy",
        "cbam_flag", "rca_mar",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    ranked = df[display_cols].sort_values("likelihood_score", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"

    st.dataframe(
        ranked.head(500),
        use_container_width=True,
        height=500,
    )
    st.markdown(f"**{len(selected)} products** selected (score >= {cutoff:.1f})")
    download_csv(selected, "powershoring_likelihood_products.csv",
                 f"Likelihood cutoff={cutoff:.1f}, weights: carriers={w_carriers}, elec={w_elec}, vuln={w_vuln}, cbam={w_cbam}")

# --- COMPONENT BREAKDOWN ---
with tab_breakdown:
    st.markdown("**Score Composition — Top 30 Products**")
    top30 = selected.nlargest(30, "likelihood_score")

    # Build stacked bar data
    bar_data = []
    for _, row in top30.iterrows():
        desc = str(row.get("description", ""))[:50]
        hs = str(row.get("hs_product_code", ""))
        label = f"{hs} — {desc}"
        for comp_name, comp_label in weight_labels.items():
            contribution = row[comp_name] * weights[comp_name] / total_w
            bar_data.append({
                "product": label,
                "component": comp_label,
                "contribution": contribution,
                "total_score": row["likelihood_score"],
            })

    bar_df = pd.DataFrame(bar_data)
    # Sort products by total score
    product_order = top30.sort_values("likelihood_score", ascending=True).apply(
        lambda r: f"{r.get('hs_product_code', '')} — {str(r.get('description', ''))[:50]}", axis=1
    ).tolist()

    fig_b = px.bar(
        bar_df, x="contribution", y="product", color="component",
        orientation="h",
        color_discrete_sequence=GL_PALETTE_EXT[:4],
        title="Score Composition — Top 30 Products",
        category_orders={"product": product_order},
    )
    fig_b.update_layout(
        template=GL_TEMPLATE,
        height=max(500, len(top30) * 25),
        xaxis_title="Weighted Score Contribution",
        yaxis_title="",
        margin=dict(l=350),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_b, use_container_width=True)
