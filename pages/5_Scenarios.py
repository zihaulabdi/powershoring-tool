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
    send_scenario = st.selectbox("Scenario to send", list(active_scenarios.keys()), key="sc_send")
    if st.button("Save for Comparison"):
        st.session_state["_pending_scenario_send"] = send_scenario

# Fixed ranking: 60% Feasibility + 40% Attractiveness
rank_col = "lhf_score"

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
            "desc": f"{_sname} scenario, 60F/40A, top {top_n}",
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
            f"Running {n_scenarios} scenarios.*")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Products", f"{len(all_codes):,}")
c2.metric("Robust (4+ scenarios)", f"{robust_count:,}")
c3.metric("Scenario-specific (1 only)", f"{unique_count:,}")
c4.metric("Avg Pairwise Overlap", f"{avg_overlap:.0f}%")

# ============================================================
# TABS
# ============================================================
tab_robust, tab_quad, tab_tree, tab_ocp, tab_data = st.tabs(
    ["Robustness", "Quadrants", "Treemaps", "OCP Lens", "Data Table"]
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
                    desc_lookup[c] = str(row.get("description", ""))[:40]

        y_labels = [f"HS {h} — {desc_lookup.get(h, '')}" for h in robust_codes]
        z_matrix = []
        text_matrix = []

        for h in robust_codes:
            row_z = []
            row_t = []
            for sname in active_scenarios:
                if h in top_sets[sname]:
                    row_z.append(1)
                    row_t.append("Top N")
                else:
                    row_z.append(0)
                    row_t.append("")
            z_matrix.append(row_z)
            text_matrix.append(row_t)

        colorscale = [[0.0, "#F5F5F5"], [1.0, MOROCCO_RED]]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix, x=list(active_scenarios.keys()), y=y_labels,
            text=text_matrix, texttemplate="%{text}", textfont=dict(size=9, color="white"),
            colorscale=colorscale, zmin=0, zmax=1, showscale=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        ))
        fig_heat.update_layout(
            template=GL_TEMPLATE,
            height=max(500, len(robust_codes) * 22),
            xaxis=dict(side="top", tickangle=0),
            yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
            margin=dict(l=350, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("**Legend:** Red = In Top N | Light = Not in Top N")


# --- TAB 2: QUADRANTS ---
with tab_quad:
    focus_scenario = st.selectbox("Focus on scenario:", list(active_scenarios.keys()), key="sc_quad_focus")
    res = scenario_results[focus_scenario]
    hs4_df = res["hs4"] if "HS4" in agg_level else res["selected"]
    code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"

    f_med = hs4_df["feasibility_score"].median()
    a_med = hs4_df["attractiveness_score"].median()

    hs4_df = hs4_df.copy()
    hs4_df["quadrant"] = "Deprioritize"
    hs4_df.loc[(hs4_df["feasibility_score"] >= f_med) & (hs4_df["attractiveness_score"] >= a_med), "quadrant"] = "Top Priorities"
    hs4_df.loc[(hs4_df["feasibility_score"] >= f_med) & (hs4_df["attractiveness_score"] < a_med), "quadrant"] = "Low-Hanging Fruit"
    hs4_df.loc[(hs4_df["feasibility_score"] < f_med) & (hs4_df["attractiveness_score"] >= a_med), "quadrant"] = "Strategic Bets"

    # Top 10 to label
    composite = 0.5 * hs4_df["feasibility_score"] + 0.5 * hs4_df["attractiveness_score"]
    top10_idx = composite.nlargest(10).index

    q_colors = {"Top Priorities": MOROCCO_RED, "Low-Hanging Fruit": "#8AB920",
                "Strategic Bets": "#204B82", "Deprioritize": GREY}

    trade_col = "global_export_value"
    hs4_df["_size"] = np.sqrt(hs4_df[trade_col].clip(lower=1) / 1e8).clip(4, 40)

    fig_q = go.Figure()
    for q, color in q_colors.items():
        mask = hs4_df["quadrant"] == q
        subset = hs4_df[mask]
        fig_q.add_trace(go.Scatter(
            x=subset["feasibility_score"], y=subset["attractiveness_score"],
            mode="markers", name=q,
            marker=dict(size=subset["_size"], color=color, opacity=0.65, line=dict(width=0.5, color="white")),
            text=subset.apply(lambda r: f"HS {r[code_col]} — {str(r.get('description', ''))[:30]}<br>"
                              f"F={r['feasibility_score']:.0f} A={r['attractiveness_score']:.0f}<br>"
                              f"Trade: {format_dollars(r[trade_col])}", axis=1),
            hoverinfo="text",
        ))

    # Label top 10
    for idx in top10_idx:
        row = hs4_df.loc[idx]
        fig_q.add_annotation(
            x=row["feasibility_score"], y=row["attractiveness_score"],
            text=f"HS {row[code_col]}", showarrow=True, arrowhead=0, arrowwidth=0.8,
            font=dict(size=8, color="#333"), ax=20, ay=-15,
        )

    fig_q.add_hline(y=a_med, line_dash="dot", line_color=GREY, line_width=1)
    fig_q.add_vline(x=f_med, line_dash="dot", line_color=GREY, line_width=1)

    fig_q.update_layout(
        template=GL_TEMPLATE,
        title=dict(text=f"{focus_scenario} — Feasibility vs Attractiveness", font=dict(color=MOROCCO_RED)),
        xaxis_title="Feasibility Score", yaxis_title="Attractiveness Score",
        height=600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(size=11)),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # Quadrant counts
    qc = hs4_df["quadrant"].value_counts()
    qcols = st.columns(4)
    for i, q in enumerate(["Top Priorities", "Low-Hanging Fruit", "Strategic Bets", "Deprioritize"]):
        qcols[i].metric(q, qc.get(q, 0))

    # Score breakdowns
    st.markdown("#### Score Composition — Top 20")
    top20 = hs4_df.nlargest(20, rank_col)

    feas_comp_cols = [c for c in ["feas_density", "feas_rca", "feas_hhi", "feas_distance"] if c in top20.columns]
    attr_comp_cols = [c for c in ["attr_pci", "attr_cog", "attr_market_size", "attr_growth", "attr_spillover"] if c in top20.columns]

    col_l, col_r = st.columns(2)
    with col_l:
        if feas_comp_cols:
            labels = {"feas_density": "Density", "feas_rca": "RCA", "feas_hhi": "HHI", "feas_distance": "Distance"}
            bar_data = []
            for _, row in top20.iterrows():
                lbl = f"HS {row[code_col]}"
                for c in feas_comp_cols:
                    bar_data.append({"product": lbl, "component": labels.get(c, c), "value": row.get(c, 0)})
            bdf = pd.DataFrame(bar_data)
            product_order = top20.sort_values("feasibility_score", ascending=True).apply(
                lambda r: f"HS {r[code_col]}", axis=1).tolist()
            fig_fb = px.bar(bdf, x="value", y="product", color="component", orientation="h",
                            color_discrete_sequence=GL_PALETTE_EXT[:4],
                            title="Feasibility Breakdown",
                            category_orders={"product": product_order})
            fig_fb.update_layout(template=GL_TEMPLATE, height=max(400, len(top20)*22),
                                 xaxis_title="Percentile Score", yaxis_title="",
                                 margin=dict(l=100), legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_fb, use_container_width=True)

    with col_r:
        if attr_comp_cols:
            labels = {"attr_pci": "PCI", "attr_cog": "COG", "attr_market_size": "Mkt Size",
                      "attr_growth": "Growth", "attr_spillover": "Spillover"}
            bar_data = []
            for _, row in top20.iterrows():
                lbl = f"HS {row[code_col]}"
                for c in attr_comp_cols:
                    bar_data.append({"product": lbl, "component": labels.get(c, c), "value": row.get(c, 0)})
            bdf = pd.DataFrame(bar_data)
            product_order = top20.sort_values("attractiveness_score", ascending=True).apply(
                lambda r: f"HS {r[code_col]}", axis=1).tolist()
            fig_ab = px.bar(bdf, x="value", y="product", color="component", orientation="h",
                            color_discrete_sequence=GL_PALETTE_EXT[4:9],
                            title="Attractiveness Breakdown",
                            category_orders={"product": product_order})
            fig_ab.update_layout(template=GL_TEMPLATE, height=max(400, len(top20)*22),
                                 xaxis_title="Percentile Score", yaxis_title="",
                                 margin=dict(l=100), legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_ab, use_container_width=True)


# --- TAB 3: TREEMAPS ---
with tab_tree:
    st.markdown(f"**HS2 Chapter composition** of each scenario's Top {top_n}.")

    # Build unified HS2 color map
    all_hs2_names = sorted(set(
        nm for res in scenario_results.values()
        for nm in res["hs4"]["hs2_name"].dropna().unique()
    ))
    hs2_color_map = {nm: GL_PALETTE_EXT[i % len(GL_PALETTE_EXT)] for i, nm in enumerate(all_hs2_names)}

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


# --- TAB 4: OCP LENS ---
with tab_ocp:
    st.markdown("**OCP ecosystem** — core products and strategic adjacencies in inorganic chemicals.")

    ocp_codes = {
        "2809": ("Phosphoric acid", "OCP Core", ""),
        "3105": ("Fertilizers (DAP/MAP)", "OCP Core", ""),
        "3103": ("Mineral phosphates", "OCP Core", ""),
        "2806": ("Hydrochloric acid (HCl)", "Extension", "Batteries, Solar"),
        "2847": ("Hydrogen peroxide (H2O2)", "Extension", ""),
        "2815": ("Caustic soda (NaOH)", "Extension", "Green H2, Carbon Capture"),
        "2818": ("Aluminium oxide (Al2O3)", "Extension", "Carbon Capture"),
        "2814": ("Ammonia (NH3)", "Greenfield", "Green H2, Fertilizers"),
        "2828": ("Hypochlorites", "Extension", "Water Treatment"),
    }

    # Pull from No Prior scenario (or first available)
    ref_scenario = "No Prior" if "No Prior" in scenario_results else list(scenario_results.keys())[0]
    ref_hs4 = scenario_results[ref_scenario]["hs4"]

    ocp_rows = []
    for code, (label, category, green_sc) in ocp_codes.items():
        row = ref_hs4[ref_hs4["hs4_code"] == code]
        if len(row) > 0:
            r = row.iloc[0]
            ocp_rows.append({
                "label": label, "category": category, "green_sc": green_sc,
                "feasibility": r["feasibility_score"], "attractiveness": r["attractiveness_score"],
                "trade": r["global_export_value"], "rca": r.get("rca_mar", 0),
            })
        else:
            # Product not in selected pool
            sub = filtered[filtered["hs_product_code"].astype(str).str.zfill(6).str[:4] == code]
            ocp_rows.append({
                "label": label, "category": category, "green_sc": green_sc,
                "feasibility": 0, "attractiveness": 0,
                "trade": sub["global_export_value"].sum() if len(sub) > 0 else 0,
                "rca": sub["rca_mar"].mean() if len(sub) > 0 else 0,
            })

    ocp_df = pd.DataFrame(ocp_rows).sort_values("feasibility", ascending=True)

    cat_colors = {"OCP Core": MOROCCO_RED, "Extension": "#204B82", "Greenfield": "#8AB920"}

    fig_ocp = go.Figure()
    for cat, color in cat_colors.items():
        subset = ocp_df[ocp_df["category"] == cat]
        fig_ocp.add_trace(go.Bar(
            y=subset["label"], x=subset["feasibility"], orientation="h",
            marker_color=color, name=cat,
            text=subset.apply(
                lambda r: f"F={r['feasibility']:.0f}  A={r['attractiveness']:.0f}  "
                          f"RCA={r['rca']:.1f}  {r['green_sc']}" if r['feasibility'] > 0
                else f"Not in pool  RCA={r['rca']:.1f}", axis=1),
            textposition="outside", textfont=dict(size=9),
        ))

    fig_ocp.update_layout(
        template=GL_TEMPLATE,
        title=dict(text="OCP Ecosystem — Feasibility of Adjacent Products", font=dict(color=MOROCCO_RED)),
        xaxis_title="Feasibility Score", barmode="stack",
        height=450, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=200, r=150, t=50, b=80),
    )
    st.plotly_chart(fig_ocp, use_container_width=True)

    # Ammonia vs Aluminium radar
    st.markdown("#### Why Aluminium ranks and Ammonia doesn't")
    col_l, col_r = st.columns([2, 1])

    alu_hs4 = filtered[filtered["hs_product_code"].astype(str).str.zfill(6).str[:4] == "7601"]
    amm_hs4 = filtered[filtered["hs_product_code"].astype(str).str.zfill(6).str[:4] == "2814"]

    if len(alu_hs4) > 0 and len(amm_hs4) > 0:
        dims = {
            "amount_electric_energy": "Electricity\nIntensity",
            "global_export_value": "Market\nSize",
            "rca_mar": "Morocco\nRCA",
            "density_mar": "Product Space\nProximity",
            "pci": "Product\nComplexity",
            "vulnerability_score": "Incumbent\nVulnerability",
        }

        def normalize(val, col):
            mn, mx = filtered[col].min(), filtered[col].max()
            if mx == mn:
                return 50
            n = (val - mn) / (mx - mn) * 100
            if col == "vulnerability_score":
                n = 100 - n
            return max(0, min(100, n))

        alu_vals = [normalize(alu_hs4[col].mean(), col) for col in dims]
        amm_vals = [normalize(amm_hs4[col].mean(), col) for col in dims]
        alu_vals.append(alu_vals[0])
        amm_vals.append(amm_vals[0])
        theta = list(dims.values()) + [list(dims.values())[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=alu_vals, theta=theta, fill="toself", name="Aluminium (7601)",
            line_color=MOROCCO_RED, fillcolor="rgba(194,34,41,0.15)", opacity=0.7,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=amm_vals, theta=theta, fill="toself", name="Ammonia (2814)",
            line_color="#204B82", fillcolor="rgba(32,75,130,0.15)", opacity=0.7,
        ))
        fig_radar.update_layout(
            template=GL_TEMPLATE,
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
            height=450, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            margin=dict(t=30, b=60),
        )
        with col_l:
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_r:
            st.markdown(
                "**Ammonia** incumbents (Qatar, Saudi Arabia, Trinidad) are "
                "**energy-surplus** countries — they are NOT vulnerable to displacement.\n\n"
                "**Aluminium** incumbents are in energy-deficit countries, "
                "electricity is a major cost driver, and Morocco has product space proximity.\n\n"
                "Green ammonia is a **greenfield bet**, not a powershoring opportunity."
            )


# --- TAB 5: DATA TABLE ---
with tab_data:
    st.markdown(f"**Cross-scenario summary** — all products appearing in any scenario's Top {top_n}.")

    code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"

    cross_rows = []
    for h, info in code_appearances.items():
        # Get metadata from first scenario that has it
        meta = {"description": "", "hs2_name": "", "trade": 0, "feas": [], "attr": []}
        for sname, res in scenario_results.items():
            src = res["hs4"] if "HS4" in agg_level else res["selected"]
            match = src[src[code_col].astype(str) == str(h)]
            if len(match) > 0:
                r = match.iloc[0]
                if not meta["description"]:
                    meta["description"] = str(r.get("description", ""))[:50]
                    meta["hs2_name"] = str(r.get("hs2_name", ""))
                meta["trade"] = max(meta["trade"], r.get("global_export_value", 0))
                meta["feas"].append(r.get("feasibility_score", 0))
                meta["attr"].append(r.get("attractiveness_score", 0))

        cross_rows.append({
            "Code": h,
            "Description": meta["description"],
            "HS2 Chapter": meta["hs2_name"],
            "Appearances": info["total"],
            "Scenarios": ", ".join(info["scenarios"]),
            "Avg Feasibility": np.mean(meta["feas"]) if meta["feas"] else 0,
            "Avg Attractiveness": np.mean(meta["attr"]) if meta["attr"] else 0,
            "Trade": meta["trade"],
        })

    cross_df = pd.DataFrame(cross_rows).sort_values(
        ["Appearances", "Avg Feasibility"], ascending=[False, False]
    ).reset_index(drop=True)
    cross_df.index = cross_df.index + 1
    cross_df.index.name = "Rank"

    # Format for display
    display_df = cross_df.copy()
    display_df["Trade"] = display_df["Trade"].apply(format_dollars)
    display_df["Avg Feasibility"] = display_df["Avg Feasibility"].round(0).astype(int)
    display_df["Avg Attractiveness"] = display_df["Avg Attractiveness"].round(0).astype(int)

    st.dataframe(display_df, use_container_width=True, height=600)
    st.markdown(f"**{len(cross_df)} unique products** across {n_scenarios} scenarios")
    download_csv(cross_df, "powershoring_cross_scenario.csv",
                 f"Scenarios: {', '.join(active_scenarios.keys())} | 60F/40A | Top {top_n}")
