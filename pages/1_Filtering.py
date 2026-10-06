"""Step 1: define the candidate pool by energy and trade thresholds."""

import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui  # noqa: E402
import utils as U  # noqa: E402

df = U.load_data()

# ============================================================
# SIDEBAR: thresholds (persist across pages)
# ============================================================
sb = st.sidebar
sb.header("Energy thresholds")
energy_pct = ui.persist(sb.slider, "Total energy intensity: minimum percentile", "s1_energy", 75,
                        min_value=0, max_value=100, step=5, format="%dth",
                        help="Percentile of fuel + electricity use per $ of output, across all products.")
elec_pct = ui.persist(sb.slider, "Electricity intensity: minimum percentile", "s1_elec", 50,
                      min_value=0, max_value=100, step=5, format="%dth",
                      help="Percentile of electricity use per $ of output, across all products.")
logic = ui.persist(sb.radio, "A product passes if it clears", "s1_logic", "Either threshold (OR)",
                   options=["Either threshold (OR)", "Both thresholds (AND)"])

sb.header("Trade threshold")
trade_pct = ui.persist(sb.slider, "World trade: minimum percentile", "s1_trade", 15,
                       min_value=0, max_value=50, step=5, format="%dth",
                       help="Removes thinly traded products, which are hard to build an export industry on.")

with sb.expander("More filters"):
    legacy = ui.persist(st.checkbox, "Legacy exclusions (HS 01–27 only)", "s1_legacy", False,
                        help="Default also excludes stone articles (HS 68), precious metals (HS 71) and raw "
                             "fibres (HS 5001, 5101, 5201, 5301). Tick to reproduce pre-April 2026 results.")
    cbam_only = ui.persist(st.checkbox, "CBAM-covered products only", "s1_cbam", False)
    green_only = ui.persist(st.checkbox, "Green supply chain products only", "s1_green", False)
    topics = []
    if green_only:
        all_topics = sorted([t for t in df["green_topic"].dropna().unique() if t])
        topics = ui.persist(st.multiselect, "Supply chains", "s1_topics", all_topics, options=all_topics)
    rca_min = ui.persist(st.slider, "Minimum Morocco RCA", "s1_rca", 0.0,
                         min_value=0.0, max_value=5.0, step=0.1,
                         help="Above 0 keeps only products Morocco already exports.")

if sb.button("Reset step 1 to defaults"):
    ui.reset_settings("s1_")
    st.rerun()

# ============================================================
# COMPUTE
# ============================================================
pool, meta = ui.candidate_pool()
th = meta["thresholds"]
n_cbam = int((pool["cbam_flag"] == 1).sum())

# ============================================================
# HEADER
# ============================================================
ui.page_header(
    "Step 1 of 3 · Candidate pool",
    "Which products are energy-intensive enough to matter?",
    lede=(
        "Powershoring only matters for products where energy is a large share of costs. This step keeps "
        "manufactured products that use a lot of energy or electricity per dollar of output, and that are "
        f"traded in meaningful volumes. Of {len(df):,} products, <b>{len(pool):,}</b> pass the current thresholds."
    ),
)

if not ui.stage1_is_default():
    ui.note("You have changed the defaults. Steps 2 and 3 use these settings. "
            "The Shortlist page always uses the defaults.", accent=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Candidate products", f"{len(pool):,}")
k2.metric("Share of all products", f"{100 * len(pool) / len(df):.0f}%")
k3.metric("World exports", ui.fmt_usd(pool["global_export_value"].sum()))
k4.metric("CBAM-covered", f"{n_cbam:,}")

logic_word = "and" if "AND" in logic else "or"
ui.small(
    f"Rule: energy ≥ {th['energy_threshold']:.1f} MJ/$ (top {100 - energy_pct}%) {logic_word} "
    f"electricity ≥ {th['elec_threshold']:.2f} MJ/$ (top {100 - elec_pct}%), and world trade ≥ "
    f"{ui.fmt_usd(th['trade_threshold'])}. Agriculture, food and extractive chapters (HS 01–27) are excluded"
    + ("." if legacy else ", along with HS 68, HS 71 and raw fibres.")
    + " Percentiles are computed across all products."
)

if len(pool) == 0:
    st.warning("No products pass these thresholds. Loosen them in the sidebar.")
    st.stop()

# ============================================================
# SECTOR COMPOSITION
# ============================================================
size_by = st.radio("Measure sectors by", ["Number of products", "World exports"], horizontal=True,
                   key="f_size_by")
if size_by == "Number of products":
    s = pool.groupby("hs2_name").size().sort_values(ascending=False)
    fmt, sub = ",.0f", "Number of HS6 products in the pool, top 15 HS2 chapters."
else:
    s = pool.groupby("hs2_name")["global_export_value"].sum().sort_values(ascending=False) / 1e9
    fmt, sub = ",.0f", "World exports of products in the pool, $ billion, top 15 HS2 chapters."
ui.chart_header("Which sectors make up the pool", sub)
top = s.head(15)
ui.plot(ui.hbar(top.index, top.values, value_fmt=fmt))
if len(s) > 15:
    ui.small(f"{len(s) - 15} smaller chapters not shown. The full list is in the table below.")

# ============================================================
# DISTRIBUTIONS
# ============================================================
with st.expander("Where the thresholds fall in the distribution"):
    d1, d2, d3 = st.columns(3)

    def _hist(all_vals, kept_vals, cut, title, xlab):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=all_vals, name="All products", marker_color=ui.WASH, nbinsx=50))
        fig.add_trace(go.Histogram(x=kept_vals, name="In pool", marker_color=ui.INK, nbinsx=50))
        if cut is not None:
            fig.add_vline(x=cut, line_dash="dot", line_color=ui.ACCENT)
        fig.update_layout(barmode="overlay", height=260, xaxis_title=xlab, yaxis_title="Products",
                          margin=dict(l=10, r=10, t=30, b=40))
        ui.chart_header(title)
        ui.plot(fig)

    # Log scale keeps the long right tail readable
    with d1:
        _hist(np.log10(df["amount_carriers"].dropna().clip(lower=1e-3)),
              np.log10(pool["amount_carriers"].dropna().clip(lower=1e-3)),
              np.log10(max(th["energy_threshold"], 1e-3)), "Total energy intensity", "log10 MJ per $")
    with d2:
        _hist(np.log10(df["amount_electric_energy"].dropna().clip(lower=1e-3)),
              np.log10(pool["amount_electric_energy"].dropna().clip(lower=1e-3)),
              np.log10(max(th["elec_threshold"], 1e-3)), "Electricity intensity", "log10 MJ per $")
    with d3:
        _hist(np.log10(df["global_export_value"].clip(lower=1)),
              np.log10(pool["global_export_value"].clip(lower=1)),
              np.log10(max(th["trade_threshold"], 1)), "World trade", "log10 $")
    ui.small("Grey: all products. Black: products in the pool. Dotted line: threshold. "
             "With OR logic, products below one energy line can still pass on the other.")

# ============================================================
# TABLE
# ============================================================
ui.chart_header("Products in the pool", "Sorted by world exports. Search with the magnifier icon above the table.")
tbl = pool.copy()
tbl["HS6"] = tbl["hs_product_code"].astype(str).str.zfill(6)
tbl["Product"] = ui.product_names(tbl, 90)
cols = ["HS6", "Product", "hs2_name", "amount_carriers", "amount_electric_energy",
        "global_export_value", "rca_mar", "cbam_flag"]
st.dataframe(
    tbl.sort_values("global_export_value", ascending=False)[cols],
    column_config={
        "HS6": st.column_config.TextColumn(width="small"),
        "Product": st.column_config.TextColumn(width="large"),
        "hs2_name": st.column_config.TextColumn("Sector", width="medium"),
        "amount_carriers": st.column_config.NumberColumn("Energy (MJ/$)", format="%.1f"),
        "amount_electric_energy": st.column_config.NumberColumn("Electricity (MJ/$)", format="%.2f"),
        "global_export_value": st.column_config.NumberColumn("World exports ($)", format="compact"),
        "rca_mar": st.column_config.NumberColumn("Morocco RCA", format="%.2f"),
        "cbam_flag": st.column_config.CheckboxColumn("CBAM"),
    },
    hide_index=True, width="stretch", height=460,
)
U.download_csv(pool, "powershoring_candidate_pool.csv", meta["description"])

st.divider()
st.page_link("pages/2_Likelihood_Prioritization.py", label="Next: score and rank these products  →")
