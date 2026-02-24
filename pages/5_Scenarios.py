"""Scenario Analysis — Cross-reference 6 likelihood scenarios to identify robust candidates."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data, GL_PALETTE_EXT, GL_TEMPLATE, MOROCCO_RED, GREY,
    VARIABLE_LABELS, make_treemap, download_csv, format_dollars,
    percentile_rank, weighted_score,
    SCENARIO_DEFS, DEFAULT_FEAS_WEIGHTS, DEFAULT_ATTR_WEIGHTS,
    run_scenario_scoring, aggregate_to_hs4,
)

st.set_page_config(page_title="Scenario Analysis", layout="wide")
st.title("Scenario Analysis")
st.markdown(
    "Cross-reference **6 likelihood scenarios** to identify robust powershoring candidates. "
    "Each scenario emphasizes a different driver of relocation — products appearing across "
    "multiple scenarios are the strongest candidates regardless of assumptions."
)

# ============================================================
# LOAD DATA & STAGE 1 FILTERING
# ============================================================
df_all = load_data()

# Use Stage 1 filtered products if available, otherwise apply defaults
if st.session_state.get("filtered_products") is not None:
    filtered = st.session_state.filtered_products.copy()
    filter_source = "Stage 1 filtering"
else:
    energy_thresh = df_all["amount_carriers"].quantile(0.75)
    elec_thresh = df_all["amount_electric_energy"].quantile(0.50)
    trade_thresh = df_all["global_export_value"].quantile(0.15)
    mask = pd.Series(True, index=df_all.index)
    mask &= df_all["hs2_code"].astype(int) >= 25
    mask &= ~df_all["hs2_code"].astype(int).isin([25, 26, 27])
    mask &= ((df_all["amount_carriers"] >= energy_thresh) | (df_all["amount_electric_energy"] >= elec_thresh))
    mask &= df_all["global_export_value"] >= trade_thresh
    filtered = df_all[mask].copy()
    filter_source = "default thresholds"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Scenarios")
    st.caption("Select which scenarios to include.")
    active_scenarios = {}
    for sname, sdef in SCENARIO_DEFS.items():
        if st.checkbox(sname, value=True, key=f"sc_{sname}", help=sdef["desc"]):
            active_scenarios[sname] = sdef

    st.divider()
    st.header("Feasibility vs Attractiveness")
    feas_pct = st.slider("Feasibility weight (%)", 0, 100, 60, 5, key="sc_fa_balance",
                         help="Remaining weight goes to attractiveness")
    attr_pct = 100 - feas_pct
    st.caption(f"**{feas_pct}% Feasibility / {attr_pct}% Attractiveness**")

    st.divider()
    st.header("Feasibility Weights")
    fw_density = st.slider("Product Space Proximity (Density)", 0, 100, 50, key="sc_fw_d")
    fw_rca = st.slider("Existing Capability (RCA)", 0, 100, 20, key="sc_fw_r")
    fw_hhi = st.slider("Market Fragmentation (1/HHI)", 0, 100, 15, key="sc_fw_h")
    fw_dist = st.slider("Transport Sensitivity (Distance)", 0, 100, 15, key="sc_fw_dist")
    feas_w = {"rca": fw_rca, "density": fw_density, "hhi": fw_hhi, "distance": fw_dist}
    fw_total = sum(feas_w.values())
    if fw_total == 0:
        st.error("At least one feasibility weight must be > 0")
        st.stop()
    st.caption(f"Density {fw_density/fw_total*100:.0f}% | RCA {fw_rca/fw_total*100:.0f}% | "
               f"HHI {fw_hhi/fw_total*100:.0f}% | Dist {fw_dist/fw_total*100:.0f}%")

    st.divider()
    st.header("Attractiveness Weights")
    aw_pci = st.slider("Product Complexity (PCI)", 0, 100, 30, key="sc_aw_p")
    aw_cog = st.slider("Diversification Value (COG)", 0, 100, 30, key="sc_aw_c")
    aw_mkt = st.slider("Market Size", 0, 100, 15, key="sc_aw_m")
    aw_grw = st.slider("Market Growth", 0, 100, 15, key="sc_aw_g")
    aw_spl = st.slider("Spillover Potential", 0, 100, 10, key="sc_aw_s")
    attr_w = {"market_size": aw_mkt, "growth": aw_grw, "cog": aw_cog, "pci": aw_pci, "spillover": aw_spl}
    aw_total = sum(attr_w.values())
    if aw_total == 0:
        st.error("At least one attractiveness weight must be > 0")
        st.stop()
    st.caption(f"PCI {aw_pci/aw_total*100:.0f}% | COG {aw_cog/aw_total*100:.0f}% | "
               f"Mkt {aw_mkt/aw_total*100:.0f}% | Grw {aw_grw/aw_total*100:.0f}% | "
               f"Spl {aw_spl/aw_total*100:.0f}%")

    st.divider()
    st.header("Display")

    top_n = st.slider("Top N per scenario", 10, 50, 30, 5, key="sc_topn")

    agg_level = st.radio("Aggregation", ["HS4 (4-digit)", "HS6 (product)"], key="sc_agg")

    st.divider()
    st.header("Send to Comparison")
    send_scenario = st.selectbox("Scenario to send", list(active_scenarios.keys()) or ["—"], key="sc_send")
    if st.button("Save for Comparison"):
        st.session_state["_pending_scenario_send"] = send_scenario

rank_col = "composite_score"

if not active_scenarios:
    st.warning("Select at least one scenario in the sidebar.")
    st.stop()

# ============================================================
# RUN ALL SCENARIOS
# ============================================================
scenario_results = {}  # name → {"selected": DF, "hs4": DF}
for sname, sdef in active_scenarios.items():
    input_df = filtered
    if sdef.get("pre_filter") == "cbam":
        input_df = filtered[filtered["cbam_flag"] == 1].copy()
    sel = run_scenario_scoring(input_df, sdef["weights"], feas_w, attr_w)
    # Compute composite with user's slider balance
    sel["composite_score"] = (feas_pct / 100) * sel["feasibility_score"] + (attr_pct / 100) * sel["attractiveness_score"]
    hs4 = aggregate_to_hs4(sel)
    scenario_results[sname] = {"selected": sel, "hs4": hs4}

# Handle pending scenario send
if st.session_state.get("_pending_scenario_send"):
    _sname = st.session_state.pop("_pending_scenario_send")
    if _sname in scenario_results:
        if "saved_scenarios" not in st.session_state:
            st.session_state.saved_scenarios = {}
        _res = scenario_results[_sname]
        st.session_state.saved_scenarios[f"Scenario: {_sname}"] = {
            "products": _res["selected"].copy(),
            "stage": "scenarios",
            "desc": f"{_sname} scenario, {feas_pct}F/{attr_pct}A, top {top_n}",
        }
        st.success(f"Saved **{_sname}** ({len(_res['selected'])} products) to Comparison.")

# ============================================================
# BUILD CROSS-SCENARIO DATA
# ============================================================
n_scenarios = len(active_scenarios)
top_sets = {}  # sname → set of top N codes
for sname, res in scenario_results.items():
    df_rank = res["hs4"] if "HS4" in agg_level else res["selected"]
    code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"
    top_sets[sname] = set(df_rank.nlargest(top_n, rank_col)[code_col])

# Count appearances
all_codes = set()
code_appearances = {}
for sname in active_scenarios:
    for h in top_sets[sname]:
        all_codes.add(h)
        if h not in code_appearances:
            code_appearances[h] = {"scenarios": [], "total": 0}
        code_appearances[h]["scenarios"].append(sname)

for h in code_appearances:
    code_appearances[h]["total"] = len(code_appearances[h]["scenarios"])

robust_count = sum(1 for v in code_appearances.values() if v["total"] >= 4)
unique_count = sum(1 for v in code_appearances.values() if v["total"] == 1)

# Jaccard overlaps
overlaps = []
snames = list(active_scenarios.keys())
for i, s1 in enumerate(snames):
    for s2 in snames[i+1:]:
        j = len(top_sets[s1] & top_sets[s2]) / max(len(top_sets[s1] | top_sets[s2]), 1)
        overlaps.append(j)
avg_overlap = np.mean(overlaps) * 100 if overlaps else 0

# ============================================================
# KPI ROW
# ============================================================
st.markdown(f"*Using {len(filtered):,} products from {filter_source}. "
            f"Running {n_scenarios} scenarios. Ranking: {feas_pct}% F / {attr_pct}% A.*")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Products", f"{len(all_codes):,}")
c2.metric("Robust (4+ scenarios)", f"{robust_count:,}")
c3.metric("Scenario-specific (1 only)", f"{unique_count:,}")
c4.metric("Avg Pairwise Overlap", f"{avg_overlap:.0f}%")

# ============================================================
# TABS
# ============================================================
tab_robust, tab_breakdown, tab_tree, tab_data = st.tabs(
    ["Robustness", "Breakdown", "Treemaps", "Data Table"]
)

# --- TAB 1: ROBUSTNESS HEATMAP ---
with tab_robust:
    st.markdown(f"**Which products appear across scenarios?** Showing products in 2+ scenario Top {top_n}s.")

    # Filter to 2+ appearances, build matrix
    robust_codes = sorted(
        [h for h, v in code_appearances.items() if v["total"] >= 2],
        key=lambda h: -code_appearances[h]["total"]
    )

    if not robust_codes:
        st.info("No products appear in 2+ scenarios. Try increasing Top N.")
    else:
        # Get descriptions
        desc_lookup = {}
        for sname, res in scenario_results.items():
            src = res["hs4"] if "HS4" in agg_level else res["selected"]
            code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"
            for _, row in src.iterrows():
                c = str(row[code_col])
                if c not in desc_lookup:
                    desc_lookup[c] = str(row.get("description", ""))

        y_labels = [f"HS {h} — {desc_lookup.get(h, '')[:50]}" for h in robust_codes]
        z_matrix = []

        for h in robust_codes:
            row_z = []
            for sname in active_scenarios:
                row_z.append(1 if h in top_sets[sname] else 0)
            z_matrix.append(row_z)

        colorscale = [[0.0, "#F5F5F5"], [1.0, MOROCCO_RED]]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix, x=list(active_scenarios.keys()), y=y_labels,
            colorscale=colorscale, zmin=0, zmax=1, showscale=False,
            hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
        ))
        fig_heat.update_layout(
            template=GL_TEMPLATE,
            height=max(500, len(robust_codes) * 22),
            xaxis=dict(side="top", tickangle=0),
            yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
            margin=dict(l=350, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# --- TAB 2: BREAKDOWN ---
with tab_breakdown:
    focus_scenario = st.selectbox("Focus on scenario:", list(active_scenarios.keys()), key="sc_breakdown_focus")
    res = scenario_results[focus_scenario]
    hs4_df = res["hs4"] if "HS4" in agg_level else res["selected"]
    code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"

    # Top N from the focused scenario
    breakdown_df = hs4_df.nlargest(top_n, rank_col).copy()

    feas_comp_cols = [c for c in ["feas_density", "feas_rca", "feas_hhi", "feas_distance"] if c in breakdown_df.columns]
    attr_comp_cols = [c for c in ["attr_pci", "attr_cog", "attr_market_size", "attr_growth", "attr_spillover"] if c in breakdown_df.columns]

    def _make_breakdown_chart(df, sort_col, comp_cols, comp_labels, comp_weights, palette, title):
        """Build a normalized 100% stacked bar chart showing weighted contribution."""
        ranked = df.sort_values(sort_col, ascending=False)
        bar_data = []
        for _, row in ranked.iterrows():
            lbl = f"HS {row[code_col]} — {str(row.get('description', ''))[:40]}"
            weighted_vals = {c: max(row.get(c, 0), 0) * comp_weights.get(c, 1) for c in comp_cols}
            total = sum(weighted_vals.values())
            for c in comp_cols:
                pct = (weighted_vals[c] / total * 100) if total > 0 else 0
                bar_data.append({"product": lbl, "component": comp_labels.get(c, c), "value": pct})
        bdf = pd.DataFrame(bar_data)
        product_order = ranked.sort_values(sort_col, ascending=True).apply(
            lambda r: f"HS {r[code_col]} — {str(r.get('description', ''))[:40]}", axis=1).tolist()
        fig = px.bar(bdf, x="value", y="product", color="component", orientation="h",
                     color_discrete_sequence=palette, title=title)
        fig.update_layout(template=GL_TEMPLATE, height=max(500, len(ranked) * 28),
                          xaxis_title="Contribution to Score (%)", yaxis_title="",
                          margin=dict(l=350, b=80), legend=dict(orientation="h", y=-0.15),
                          xaxis=dict(range=[0, 100]))
        fig.update_yaxes(categoryorder="array", categoryarray=product_order,
                         dtick=1, tickfont=dict(size=11))
        return fig

    feas_labels = {"feas_density": "Density", "feas_rca": "RCA", "feas_hhi": "HHI", "feas_distance": "Distance"}
    attr_labels = {"attr_pci": "PCI", "attr_cog": "COG", "attr_market_size": "Mkt Size",
                   "attr_growth": "Growth", "attr_spillover": "Spillover"}

    # Map component columns to their user-chosen weights
    feas_comp_weights = {"feas_density": fw_density, "feas_rca": fw_rca, "feas_hhi": fw_hhi, "feas_distance": fw_dist}
    attr_comp_weights = {"attr_pci": aw_pci, "attr_cog": aw_cog, "attr_market_size": aw_mkt,
                         "attr_growth": aw_grw, "attr_spillover": aw_spl}
    # Composite weights: component weight × F or A share
    composite_weights = {c: w * feas_pct for c, w in feas_comp_weights.items()}
    composite_weights.update({c: w * attr_pct for c, w in attr_comp_weights.items()})

    # 3 panels
    st.markdown(f"#### Ranked by Composite Score ({feas_pct}F/{attr_pct}A)")
    all_comp_cols = feas_comp_cols + attr_comp_cols
    all_comp_labels = {**feas_labels, **attr_labels}
    all_palette = GL_PALETTE_EXT[:len(all_comp_cols)]
    fig_comp = _make_breakdown_chart(breakdown_df, rank_col, all_comp_cols, all_comp_labels,
                                     composite_weights, all_palette,
                                     f"Composite Score ({feas_pct}F/{attr_pct}A)")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Ranked by Feasibility")
    if feas_comp_cols:
        fig_fb = _make_breakdown_chart(breakdown_df, "feasibility_score", feas_comp_cols,
                                       feas_labels, feas_comp_weights, GL_PALETTE_EXT[:4],
                                       "Feasibility Breakdown")
        st.plotly_chart(fig_fb, use_container_width=True)

    st.markdown("#### Ranked by Attractiveness")
    if attr_comp_cols:
        fig_ab = _make_breakdown_chart(breakdown_df, "attractiveness_score", attr_comp_cols,
                                       attr_labels, attr_comp_weights, GL_PALETTE_EXT[4:9],
                                       "Attractiveness Breakdown")
        st.plotly_chart(fig_ab, use_container_width=True)


# --- TAB 3: TREEMAPS ---
with tab_tree:
    st.markdown(f"**HS2 Chapter composition** of each scenario's Top {top_n}.")

    # Build unified HS2 color map from ALL scenarios' full pools
    all_hs2_names = sorted(set(
        nm for res in scenario_results.values()
        for nm in res["hs4"]["hs2_name"].dropna().unique()
    ))
    _extended_palette = (
        GL_PALETTE_EXT
        + ["#264653", "#E9C46A", "#F4A261", "#2A9D8F", "#E63946",
           "#457B9D", "#1D3557", "#A8DADC", "#606C38", "#DDA15E"]
    )
    hs2_color_map = {nm: _extended_palette[i % len(_extended_palette)] for i, nm in enumerate(all_hs2_names)}

    # Render in 2×3 grid
    sname_list = list(active_scenarios.keys())
    for row_start in range(0, len(sname_list), 3):
        cols = st.columns(min(3, len(sname_list) - row_start))
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(sname_list):
                break
            sname = sname_list[idx]
            res = scenario_results[sname]
            top_df = res["hs4"].nlargest(top_n, rank_col).copy()

            # Aggregate to HS2 for treemap
            top_df["hs2_code"] = top_df["hs4_code"].str[:2]
            hs2_agg = top_df.groupby(["hs2_code", "hs2_name"]).agg(
                value=("global_export_value", "sum"),
                n_products=("hs4_code", "count"),
            ).reset_index()

            fig_tm = px.treemap(
                hs2_agg, path=["hs2_name"], values="value",
                color="hs2_name", color_discrete_map=hs2_color_map,
                title=sname,
            )
            fig_tm.update_traces(textinfo="label+percent root")
            fig_tm.update_layout(
                template=GL_TEMPLATE, height=350,
                title=dict(font=dict(size=13, color=MOROCCO_RED)),
                margin=dict(t=40, l=5, r=5, b=5),
                showlegend=False,
            )
            with col:
                st.plotly_chart(fig_tm, use_container_width=True)


# --- TAB 4: DATA TABLE ---
with tab_data:
    st.markdown(f"**Cross-scenario summary** — all products appearing in any scenario's Top {top_n}.")

    code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"

    cross_rows = []
    for h, info in code_appearances.items():
        meta = {
            "description": "", "hs2_name": "", "trade": 0,
            "feas": [], "attr": [], "composite": [], "likelihood": [],
            "feas_rca": [], "feas_density": [], "feas_hhi": [], "feas_distance": [],
            "attr_pci": [], "attr_cog": [], "attr_market_size": [], "attr_growth": [], "attr_spillover": [],
            "rca_mar": [], "density_mar": [], "elec_int": [],
            "cbam_flag": 0, "green_flag": 0,
        }
        for sname, res in scenario_results.items():
            src = res["hs4"] if "HS4" in agg_level else res["selected"]
            match = src[src[code_col].astype(str) == str(h)]
            if len(match) > 0:
                r = match.iloc[0]
                if not meta["description"]:
                    meta["description"] = str(r.get("description", ""))
                    meta["hs2_name"] = str(r.get("hs2_name", ""))
                meta["trade"] = max(meta["trade"], r.get("global_export_value", 0))
                meta["feas"].append(r.get("feasibility_score", 0))
                meta["attr"].append(r.get("attractiveness_score", 0))
                meta["composite"].append(r.get("composite_score", 0))
                meta["likelihood"].append(r.get("likelihood_score", 0))
                for k in ["feas_rca", "feas_density", "feas_hhi", "feas_distance",
                          "attr_pci", "attr_cog", "attr_market_size", "attr_growth", "attr_spillover"]:
                    meta[k].append(r.get(k, 0))
                for k in ["rca_mar", "density_mar", "elec_int"]:
                    meta[k].append(r.get(k, 0))
                meta["cbam_flag"] = max(meta["cbam_flag"], r.get("cbam_flag", 0))
                meta["green_flag"] = max(meta["green_flag"], r.get("green_flag", 0))

        _avg = lambda lst: np.mean(lst) if lst else 0

        cross_rows.append({
            "Code": h,
            "Description": meta["description"],
            "HS2 Chapter": meta["hs2_name"],
            "Appearances": info["total"],
            "Scenarios": ", ".join(info["scenarios"]),
            f"Composite ({feas_pct}F/{attr_pct}A)": round(_avg(meta["composite"]), 1),
            "Feasibility": round(_avg(meta["feas"]), 1),
            "Attractiveness": round(_avg(meta["attr"]), 1),
            "Likelihood": round(_avg(meta["likelihood"]), 1),
            "F: Density": round(_avg(meta["feas_density"]), 1),
            "F: RCA": round(_avg(meta["feas_rca"]), 1),
            "F: HHI": round(_avg(meta["feas_hhi"]), 1),
            "F: Distance": round(_avg(meta["feas_distance"]), 1),
            "A: PCI": round(_avg(meta["attr_pci"]), 1),
            "A: COG": round(_avg(meta["attr_cog"]), 1),
            "A: Market Size": round(_avg(meta["attr_market_size"]), 1),
            "A: Growth": round(_avg(meta["attr_growth"]), 1),
            "A: Spillover": round(_avg(meta["attr_spillover"]), 1),
            "Morocco RCA": round(_avg(meta["rca_mar"]), 2),
            "Morocco Density": round(_avg(meta["density_mar"]), 3),
            "Elec Intensity": round(_avg(meta["elec_int"]), 2),
            "Global Trade": meta["trade"],
            "CBAM": int(meta["cbam_flag"]),
            "Green SC": int(meta["green_flag"]),
        })

    cross_df = pd.DataFrame(cross_rows).sort_values(
        ["Appearances", f"Composite ({feas_pct}F/{attr_pct}A)"], ascending=[False, False]
    ).reset_index(drop=True)
    cross_df.index = cross_df.index + 1
    cross_df.index.name = "Rank"

    st.dataframe(cross_df, use_container_width=True, height=600)
    st.markdown(f"**{len(cross_df)} unique products** across {n_scenarios} scenarios")
    download_csv(cross_df, "powershoring_cross_scenario.csv",
                 f"Scenarios: {', '.join(active_scenarios.keys())} | {feas_pct}F/{attr_pct}A | Top {top_n}")
