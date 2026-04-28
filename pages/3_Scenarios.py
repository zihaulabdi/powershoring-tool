"""Scenario Analysis -- Cross-reference 4 likelihood scenarios to identify robust candidates."""

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
    inject_custom_css, _HS4_DESC_LOOKUP, _default_stage1_filter,
)

st.set_page_config(page_title="Scenario Analysis", layout="wide")
inject_custom_css()
st.title("Scenario Analysis")
st.markdown(
    "Cross-reference **4 likelihood scenarios** to identify robust powershoring candidates. "
    "Each scenario represents a different theory of why industries relocate to renewable-rich locations. "
    "Products appearing across multiple scenarios are the strongest candidates regardless of assumptions."
)

# ============================================================
# LOAD DATA & STAGE 1 FILTERING
# ============================================================
df_all = load_data()

if st.session_state.get("filtered_products") is not None:
    filtered = st.session_state.filtered_products.copy()
    filter_source = "Stage 1 filtering"
else:
    filtered = _default_stage1_filter(df_all)
    filter_source = "default thresholds"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.divider()
    st.header("Scenarios")
    st.caption("Select which scenarios to include.")
    active_scenarios = {}
    for sname, sdef in SCENARIO_DEFS.items():
        if st.checkbox(sname, value=True, key=f"sc_{sname}", help=sdef["desc"]):
            active_scenarios[sname] = sdef

    st.divider()
    st.header("Ranking")
    feas_pct = st.slider("Feasibility weight (%)", 0, 100, 60, 5, key="sc_fa_balance",
                         help="Share of composite score from feasibility. Remainder = attractiveness.")
    attr_pct = 100 - feas_pct
    st.caption(f"**{feas_pct}% Feasibility / {attr_pct}% Attractiveness**")

    top_n = st.slider("Top N per scenario", 10, 50, 30, 5, key="sc_topn")

    st.divider()
    st.header("Feasibility Weights")
    fw_density = st.slider("Capability proximity (Density)", 0, 100, 50, key="sc_fw_d")
    fw_rca     = st.slider("Existing capability (RCA)", 0, 100, 20, key="sc_fw_r")
    fw_hhi     = st.slider("Market openness (1/HHI)", 0, 100, 15, key="sc_fw_h")
    fw_dist    = st.slider("Transport sensitivity (Distance)", 0, 100, 15, key="sc_fw_dist")
    feas_w = {"rca": fw_rca, "density": fw_density, "hhi": fw_hhi, "distance": fw_dist}
    fw_total = sum(feas_w.values())
    if fw_total == 0:
        st.error("At least one feasibility weight must be > 0")
        st.stop()
    st.caption(f"Density {fw_density/fw_total*100:.0f}% | RCA {fw_rca/fw_total*100:.0f}% | "
               f"HHI {fw_hhi/fw_total*100:.0f}% | Dist {fw_dist/fw_total*100:.0f}%")

    st.divider()
    st.header("Attractiveness Weights")
    aw_pci = st.slider("Product complexity (PCI)", 0, 100, 30, key="sc_aw_p")
    aw_cog = st.slider("Diversification value (COG)", 0, 100, 30, key="sc_aw_c")
    aw_mkt = st.slider("Market size", 0, 100, 15, key="sc_aw_m")
    aw_grw = st.slider("Market growth", 0, 100, 15, key="sc_aw_g")
    aw_spl = st.slider("Spillover potential", 0, 100, 10, key="sc_aw_s")
    attr_w = {"market_size": aw_mkt, "growth": aw_grw, "cog": aw_cog, "pci": aw_pci, "spillover": aw_spl}
    aw_total = sum(attr_w.values())
    if aw_total == 0:
        st.error("At least one attractiveness weight must be > 0")
        st.stop()
    st.caption(f"PCI {aw_pci/aw_total*100:.0f}% | COG {aw_cog/aw_total*100:.0f}% | "
               f"Mkt {aw_mkt/aw_total*100:.0f}% | Grw {aw_grw/aw_total*100:.0f}% | "
               f"Spl {aw_spl/aw_total*100:.0f}%")

    st.divider()
    agg_level = st.radio("Aggregation", ["HS4 (4-digit)", "HS6 (product)"], key="sc_agg")

    st.divider()
    st.header("Send to Comparison")
    send_scenario = st.selectbox("Scenario to send", list(active_scenarios.keys()) or ["--"], key="sc_send")
    if st.button("Save for Comparison"):
        st.session_state["_pending_scenario_send"] = send_scenario

rank_col = "composite_score"

if not active_scenarios:
    st.warning("Select at least one scenario in the sidebar.")
    st.stop()

# ============================================================
# RUN ALL SCENARIOS
# ============================================================
scenario_results = {}
for sname, sdef in active_scenarios.items():
    input_df = filtered
    if sdef.get("pre_filter") == "cbam":
        input_df = filtered[filtered["cbam_flag"] == 1].copy()
    sel = run_scenario_scoring(input_df, sdef["weights"], feas_w, attr_w, fa_reference_df=filtered)
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
top_sets = {}
for sname, res in scenario_results.items():
    df_rank = res["hs4"] if "HS4" in agg_level else res["selected"]
    code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"
    top_sets[sname] = set(df_rank.nlargest(top_n, rank_col)[code_col])

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

robust_count = sum(1 for v in code_appearances.values() if v["total"] >= n_scenarios)
unique_count = sum(1 for v in code_appearances.values() if v["total"] == 1)

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
st.markdown(
    f"*{len(filtered):,} products from {filter_source}. "
    f"{n_scenarios} scenarios. "
    f"Composite ranking: {feas_pct}% Feasibility / {attr_pct}% Attractiveness.*"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Products", f"{len(all_codes):,}")
c2.metric(f"Robust (all {n_scenarios} scenarios)", f"{robust_count:,}")
c3.metric("Scenario-specific (1 only)", f"{unique_count:,}")
c4.metric("Avg Pairwise Overlap", f"{avg_overlap:.0f}%")

# ============================================================
# TABS (Robustness, Treemaps, Data Table -- Breakdown removed)
# ============================================================
tab_robust, tab_tree, tab_data = st.tabs(["Robustness", "Treemaps", "Data Table"])

# --- TAB 1: ROBUSTNESS HEATMAP ---
with tab_robust:
    st.markdown(
        f"**Which products appear across scenarios?** "
        f"Showing products in 2+ of the {n_scenarios} scenario Top {top_n}s. "
        f"Products in all {n_scenarios} scenarios are the most robust candidates."
    )

    n_all = sum(1 for v in code_appearances.values() if v["total"] >= n_scenarios)
    n_3plus = sum(1 for v in code_appearances.values() if v["total"] >= max(n_scenarios - 1, 2))
    if n_all > 0:
        st.success(
            f"**{n_all} {'industry' if n_all == 1 else 'industries'} appear in all {n_scenarios} scenarios** -- "
            f"the strongest candidates regardless of which relocation driver you believe in."
        )
    if n_3plus > n_all:
        st.info(f"{n_3plus} industries appear in {max(n_scenarios-1, 2)}+ scenarios.")

    robust_view = st.radio(
        "View level:", ["Product level", "HS2 Chapter level"],
        horizontal=True, key="robust_view_level",
    )

    if robust_view == "HS2 Chapter level":
        hs2_scenario_counts = {}
        hs2_total_appearances = {}
        for sname in active_scenarios:
            df_rank = scenario_results[sname]["hs4"] if "HS4" in agg_level else scenario_results[sname]["selected"]
            top_df = df_rank.nlargest(top_n, rank_col)
            if "HS4" in agg_level:
                top_df = top_df.copy()
                top_df["hs2_code"] = top_df["hs4_code"].str[:2]
            for hs2n in top_df["hs2_name"].dropna().unique():
                if hs2n not in hs2_scenario_counts:
                    hs2_scenario_counts[hs2n] = {}
                n_prods = len(top_df[top_df["hs2_name"] == hs2n])
                hs2_scenario_counts[hs2n][sname] = n_prods

        for hs2n, scens in hs2_scenario_counts.items():
            hs2_total_appearances[hs2n] = len(scens)

        robust_hs2 = sorted(
            [h for h, t in hs2_total_appearances.items() if t >= 2],
            key=lambda h: (-hs2_total_appearances[h], h)
        )

        if not robust_hs2:
            st.info("No HS2 chapters appear in 2+ scenarios. Try increasing Top N.")
        else:
            y_labels_hs2 = robust_hs2
            z_matrix_hs2 = []
            hover_text_hs2 = []
            for hs2n in robust_hs2:
                row_z = []
                row_hover = []
                for sname in active_scenarios:
                    cnt = hs2_scenario_counts.get(hs2n, {}).get(sname, 0)
                    row_z.append(cnt)
                    row_hover.append(f"<b>{hs2n}</b><br>{sname}: {cnt} products")
                z_matrix_hs2.append(row_z)
                hover_text_hs2.append(row_hover)

            z_max = max(max(row) for row in z_matrix_hs2) if z_matrix_hs2 else 1

            fig_heat_hs2 = go.Figure(data=go.Heatmap(
                z=z_matrix_hs2, x=list(active_scenarios.keys()), y=y_labels_hs2,
                colorscale=[[0.0, "#F5F5F5"], [0.01, "#fde0e0"], [0.5, "#e67478"], [1.0, MOROCCO_RED]],
                zmin=0, zmax=z_max, showscale=True,
                text=[[str(v) if v > 0 else "" for v in row] for row in z_matrix_hs2],
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                hovertext=hover_text_hs2,
                hovertemplate="%{hovertext}<extra></extra>",
            ))
            for i, hs2n in enumerate(robust_hs2):
                n_app = hs2_total_appearances[hs2n]
                fig_heat_hs2.add_annotation(
                    x=len(active_scenarios) - 0.3, y=i,
                    text=f"<b>{n_app}</b>", showarrow=False,
                    font=dict(size=13, color=MOROCCO_RED),
                    xanchor="left",
                )

            fig_heat_hs2.update_layout(
                template=GL_TEMPLATE,
                height=max(450, len(robust_hs2) * 35),
                xaxis=dict(side="top", tickangle=0, tickfont=dict(size=13)),
                yaxis=dict(autorange="reversed", tickfont=dict(size=13), dtick=1),
                margin=dict(l=350, r=60, t=50, b=30),
            )
            st.plotly_chart(fig_heat_hs2, use_container_width=True)
            st.caption(
                "Cell values = number of products from that HS2 chapter in each scenario's Top N. "
                "Numbers on right = total scenarios the chapter appears in."
            )

    else:
        robust_codes = sorted(
            [h for h, v in code_appearances.items() if v["total"] >= 2],
            key=lambda h: -code_appearances[h]["total"]
        )

        if not robust_codes:
            st.info("No products appear in 2+ scenarios. Try increasing Top N.")
        else:
            desc_lookup = {}
            for sname, res in scenario_results.items():
                src = res["hs4"] if "HS4" in agg_level else res["selected"]
                _code_col = "hs4_code" if "HS4" in agg_level else "hs_product_code"
                for _, row in src.iterrows():
                    c = str(row[_code_col])
                    if c not in desc_lookup:
                        desc_lookup[c] = str(row.get("description", ""))

            y_labels = []
            for h in robust_codes:
                desc = desc_lookup.get(h, "")
                if len(desc) > 45:
                    desc = desc[:43] + ".."
                n_app = code_appearances[h]["total"]
                y_labels.append(f"({n_app}) HS {h} - {desc}")

            z_matrix = []
            for h in robust_codes:
                row_z = [1 if h in top_sets[sname] else 0 for sname in active_scenarios]
                z_matrix.append(row_z)

            fig_heat = go.Figure(data=go.Heatmap(
                z=z_matrix, x=list(active_scenarios.keys()), y=y_labels,
                colorscale=[[0.0, "#F5F5F5"], [1.0, MOROCCO_RED]],
                zmin=0, zmax=1, showscale=False,
                hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
            ))
            fig_heat.update_layout(
                template=GL_TEMPLATE,
                height=max(500, len(robust_codes) * 28),
                xaxis=dict(side="top", tickangle=0, tickfont=dict(size=13)),
                yaxis=dict(autorange="reversed", tickfont=dict(size=12), dtick=1),
                margin=dict(l=420, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(f"Number in parentheses = scenarios the product appears in. Red = in Top {top_n}, grey = not.")

    # Download
    _code_col_dl = "hs4_code" if "HS4" in agg_level else "hs_product_code"
    cross_rows_dl = []
    for h, info in code_appearances.items():
        desc_dl = ""
        hs2_dl = ""
        trade_dl = 0
        composite_dl = []
        for sname, res in scenario_results.items():
            src = res["hs4"] if "HS4" in agg_level else res["selected"]
            match = src[src[_code_col_dl].astype(str) == str(h)]
            if len(match) > 0:
                r = match.iloc[0]
                if not desc_dl:
                    desc_dl = str(r.get("description", ""))
                    hs2_dl = str(r.get("hs2_name", ""))
                trade_dl = max(trade_dl, r.get("global_export_value", 0))
                composite_dl.append(r.get("composite_score", 0))
        cross_rows_dl.append({
            "Code": h,
            "Description": desc_dl,
            "HS2 Chapter": hs2_dl,
            "Appearances": info["total"],
            "Scenarios": ", ".join(info["scenarios"]),
            f"Composite ({feas_pct}F/{attr_pct}A)": round(np.mean(composite_dl) if composite_dl else 0, 1),
            "Global Trade": trade_dl,
        })
    cross_df_dl = pd.DataFrame(cross_rows_dl).sort_values(
        ["Appearances", f"Composite ({feas_pct}F/{attr_pct}A)"], ascending=[False, False]
    ).reset_index(drop=True)

    if len(cross_df_dl) > 0:
        download_csv(cross_df_dl.head(top_n), "powershoring_robust_candidates.csv",
                     f"Top {top_n} robust candidates -- {', '.join(active_scenarios.keys())}")


# --- TAB 2: TREEMAPS ---
with tab_tree:
    st.markdown(f"**HS2 Chapter composition** of each scenario's Top {top_n}.")

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


# --- TAB 3: DATA TABLE ---
with tab_data:
    st.markdown(f"**Cross-scenario summary** -- all products appearing in any scenario's Top {top_n}.")

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

        # Add HS4 description
        hs4_desc = _HS4_DESC_LOOKUP.get(str(h), "")

        cross_rows.append({
            "Code": h,
            "HS4 Description": hs4_desc if hs4_desc else meta["description"],
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
