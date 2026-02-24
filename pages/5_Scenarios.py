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
    send_scenario = st.selectbox("Scenario to send", list(active_scenarios.keys()) or ["—"], key="sc_send")
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


# --- TAB 2: BREAKDOWN ---
with tab_breakdown:
    focus_scenario = st.selectbox("Focus on scenario:", list(active_scenarios.keys()), key="sc_breakdown_focus")
    res = scenario_results[focus_scenario]
    hs4_df = res["hs4"] if "HS4" in agg_level else res["selected"]
    code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"

    hs4_df = hs4_df.copy()

    # Color by Top N
    top_n_codes = set(hs4_df.nlargest(top_n, rank_col)[code_col])
    hs4_df["_is_top"] = hs4_df[code_col].isin(top_n_codes)

    trade_col = "global_export_value"
    hs4_df["_size"] = np.sqrt(hs4_df[trade_col].clip(lower=1) / 1e8).clip(4, 40)

    # Scatter: grey for others, red for Top N (others first so red on top)
    fig_q = go.Figure()
    others = hs4_df[~hs4_df["_is_top"]]
    top = hs4_df[hs4_df["_is_top"]]

    fig_q.add_trace(go.Scatter(
        x=others["feasibility_score"], y=others["attractiveness_score"],
        mode="markers", name="Other",
        marker=dict(size=others["_size"], color=GREY, opacity=0.4,
                    line=dict(width=0.5, color="white")),
        text=others.apply(lambda r: f"HS {r[code_col]} — {str(r.get('description', ''))}<br>"
                          f"F={r['feasibility_score']:.0f} A={r['attractiveness_score']:.0f}<br>"
                          f"Trade: {format_dollars(r[trade_col])}", axis=1),
        hoverinfo="text",
    ))
    fig_q.add_trace(go.Scatter(
        x=top["feasibility_score"], y=top["attractiveness_score"],
        mode="markers", name=f"Top {top_n}",
        marker=dict(size=top["_size"], color=MOROCCO_RED, opacity=0.7,
                    line=dict(width=0.5, color="white")),
        text=top.apply(lambda r: f"HS {r[code_col]} — {str(r.get('description', ''))}<br>"
                       f"F={r['feasibility_score']:.0f} A={r['attractiveness_score']:.0f}<br>"
                       f"Trade: {format_dollars(r[trade_col])}", axis=1),
        hoverinfo="text",
    ))

    fig_q.update_layout(
        template=GL_TEMPLATE,
        title=dict(text=f"{focus_scenario} — Feasibility vs Attractiveness", font=dict(color=MOROCCO_RED)),
        xaxis_title="Feasibility Score", yaxis_title="Attractiveness Score",
        height=600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(size=11)),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # Score breakdowns — all products appearing in at least 1 scenario, ranked by score
    st.markdown("#### Score Composition — All Top N Products (ranked by composite score)")

    # Gather all codes that appear in at least one scenario
    all_scenario_codes = set()
    for s_codes in top_sets.values():
        all_scenario_codes |= s_codes

    # Filter current scenario's data to those codes
    breakdown_df = hs4_df[hs4_df[code_col].isin(all_scenario_codes)].copy()
    breakdown_df = breakdown_df.sort_values(rank_col, ascending=False)

    feas_comp_cols = [c for c in ["feas_density", "feas_rca", "feas_hhi", "feas_distance"] if c in breakdown_df.columns]
    attr_comp_cols = [c for c in ["attr_pci", "attr_cog", "attr_market_size", "attr_growth", "attr_spillover"] if c in breakdown_df.columns]

    col_l, col_r = st.columns(2)
    with col_l:
        if feas_comp_cols:
            labels = {"feas_density": "Density", "feas_rca": "RCA", "feas_hhi": "HHI", "feas_distance": "Distance"}
            bar_data = []
            for _, row in breakdown_df.iterrows():
                lbl = f"HS {row[code_col]}"
                for c in feas_comp_cols:
                    bar_data.append({"product": lbl, "component": labels.get(c, c), "value": row.get(c, 0)})
            bdf = pd.DataFrame(bar_data)
            product_order = breakdown_df.sort_values(rank_col, ascending=True).apply(
                lambda r: f"HS {r[code_col]}", axis=1).tolist()
            fig_fb = px.bar(bdf, x="value", y="product", color="component", orientation="h",
                            color_discrete_sequence=GL_PALETTE_EXT[:4],
                            title="Feasibility Breakdown",
                            category_orders={"product": product_order})
            fig_fb.update_layout(template=GL_TEMPLATE, height=max(400, len(breakdown_df) * 20),
                                 xaxis_title="Percentile Score", yaxis_title="",
                                 margin=dict(l=100), legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_fb, use_container_width=True)

    with col_r:
        if attr_comp_cols:
            labels = {"attr_pci": "PCI", "attr_cog": "COG", "attr_market_size": "Mkt Size",
                      "attr_growth": "Growth", "attr_spillover": "Spillover"}
            bar_data = []
            for _, row in breakdown_df.iterrows():
                lbl = f"HS {row[code_col]}"
                for c in attr_comp_cols:
                    bar_data.append({"product": lbl, "component": labels.get(c, c), "value": row.get(c, 0)})
            bdf = pd.DataFrame(bar_data)
            product_order = breakdown_df.sort_values(rank_col, ascending=True).apply(
                lambda r: f"HS {r[code_col]}", axis=1).tolist()
            fig_ab = px.bar(bdf, x="value", y="product", color="component", orientation="h",
                            color_discrete_sequence=GL_PALETTE_EXT[4:9],
                            title="Attractiveness Breakdown",
                            category_orders={"product": product_order})
            fig_ab.update_layout(template=GL_TEMPLATE, height=max(400, len(breakdown_df) * 20),
                                 xaxis_title="Percentile Score", yaxis_title="",
                                 margin=dict(l=100), legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_ab, use_container_width=True)


# --- TAB 3: TREEMAPS ---
with tab_tree:
    st.markdown(f"**HS2 Chapter composition** of each scenario's Top {top_n}.")

    # Build unified HS2 color map from ALL scenarios' full pools (not just Top N)
    # to ensure consistent and unique colors
    all_hs2_names = sorted(set(
        nm for res in scenario_results.values()
        for nm in res["hs4"]["hs2_name"].dropna().unique()
    ))
    # Use a wide palette to maximize color distinctiveness
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
        # Collect all scores from all scenarios that have this product
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
                meta["composite"].append(r.get("lhf_score", 0))
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
            "Composite (60F/40A)": round(_avg(meta["composite"]), 1),
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
        ["Appearances", "Composite (60F/40A)"], ascending=[False, False]
    ).reset_index(drop=True)
    cross_df.index = cross_df.index + 1
    cross_df.index.name = "Rank"

    st.dataframe(cross_df, use_container_width=True, height=600)
    st.markdown(f"**{len(cross_df)} unique products** across {n_scenarios} scenarios")
    download_csv(cross_df, "powershoring_cross_scenario.csv",
                 f"Scenarios: {', '.join(active_scenarios.keys())} | 60F/40A | Top {top_n}")
