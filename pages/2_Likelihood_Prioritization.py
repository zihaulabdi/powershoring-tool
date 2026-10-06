"""Step 2: score relocation likelihood, then rank on feasibility and attractiveness."""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui  # noqa: E402
import utils as U  # noqa: E402

LIKE_LABELS = {
    "fuel": "Fuel intensity",
    "elec": "Electricity intensity",
    "vuln": "Incumbent vulnerability",
    "cbam": "CBAM exposure",
    "growth": "Market growth",
}
CUSTOM = "Custom weights"

pool, meta = ui.candidate_pool()

# ============================================================
# SIDEBAR
# ============================================================
sb = st.sidebar
sb.header("Relocation theory")
theory = ui.persist(sb.radio, "Why do industries relocate?", "l_theory", "No Prior",
                    options=list(U.SCENARIO_DEFS.keys()) + [CUSTOM],
                    help="Each theory weights the likelihood components differently.")

if theory == CUSTOM:
    sb.caption("Set your own weights. They are rescaled to sum to 100%.")
    weights = {k: ui.persist(sb.slider, lab, f"l_w_{k}", 20 if k != "growth" else 0,
                             min_value=0, max_value=100, step=5)
               for k, lab in LIKE_LABELS.items()}
    pre_filter, default_share = None, 0.5
    if sum(weights.values()) == 0:
        sb.error("Give at least one likelihood component a weight above 0.")
        st.stop()
else:
    sdef = U.SCENARIO_DEFS[theory]
    sb.caption(sdef["desc"])
    weights = sdef["weights"]
    pre_filter, default_share = sdef.get("pre_filter"), sdef.get("likelihood_top_share", 0.5)

keep_pct = ui.persist(sb.slider, "Keep the most likely…", f"l_keep_{theory}", int(default_share * 100),
                      min_value=10, max_value=100, step=5, format="%d%%",
                      help="Share of products kept by likelihood score before ranking. "
                           "Carbon Regulation keeps 100% because the CBAM filter already narrows the pool.")

ranking = ui.ranking_controls()
feas_pct, top_n = ranking["feas_pct"], ranking["top_n"]

# ============================================================
# COMPUTE
# ============================================================
scored = pool[pool["cbam_flag"] == 1].copy() if pre_filter == "cbam" else pool.copy()
if len(scored) == 0:
    st.warning("No products to score. Loosen the step 1 filters.")
    st.stop()

scored = U.add_likelihood_scores(scored, weights)
# Feasibility and attractiveness are ranked against the whole candidate pool,
# so scores are comparable across theories.
scored = U.add_feasibility_attractiveness_scores(scored, ranking["feas_w"], ranking["attr_w"], reference_df=pool)
scored["composite_score"] = (feas_pct / 100) * scored["feasibility_score"] + (1 - feas_pct / 100) * scored["attractiveness_score"]

if keep_pct >= 100:
    kept = scored.copy()
else:
    cutoff = scored["likelihood_score"].quantile(1 - keep_pct / 100)
    kept = scored[scored["likelihood_score"] >= cutoff].copy()


kept["hs6"] = kept["hs_product_code"].astype(str).str.zfill(6)
kept["hs4"] = kept["hs6"].str[:4]
kept["industry"] = kept["hs4"].map(lambda c: ui.short_label(U._HS4_DESC_LOOKUP.get(c, c), 45))
kept["product"] = ui.product_names(kept)
top = kept.nlargest(top_n, "composite_score").copy()

# Save for comparison
sb.header("Save this shortlist")
# Key includes the theory and N so the suggested name updates when they change
save_name = sb.text_input("Name", value=f"{theory} · top {top_n}", key=f"l_save_name_{theory}_{top_n}")
if sb.button("Save for comparison", type="primary"):
    ui.save_shortlist(save_name, top,
                      f"{theory} theory · kept {keep_pct}% by likelihood · {feas_pct}F/{100 - feas_pct}A · top {top_n}")
    sb.success(f"Saved “{save_name}”. Open Compare to see it.")

# ============================================================
# HEADER
# ============================================================
ui.page_header(
    "Step 2 of 3",
    "Score and rank",
    lede=(
        "Each product in the candidate pool gets a <b>likelihood score</b> based on the relocation theory "
        "chosen in the sidebar. The least likely products are dropped. The rest are ranked on "
        "<b>feasibility</b> (Morocco's readiness to produce them) and <b>attractiveness</b> (the value of the market)."
    ),
)

w_total = sum(weights.values())
w_text = ", ".join(f"{LIKE_LABELS[k].lower()} {100 * v / w_total:.0f}%" for k, v in weights.items() if v > 0)
ui.small(
    f"Theory: <b>{theory}</b> ({w_text})"
    + (". Only CBAM-covered products are scored" if pre_filter == "cbam" else "")
    + f". Candidate pool from step 1: {len(pool):,} products."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Products scored", f"{len(scored):,}")
k2.metric(f"Kept (top {keep_pct}%)", f"{len(kept):,}")
k3.metric(f"Top {top_n}: industries", f"{top['hs4'].nunique()}")
k4.metric(f"Top {top_n}: world exports", ui.fmt_usd(top["global_export_value"].sum()))

# ============================================================
# SCATTER
# ============================================================
ui.chart_header(
    "Feasibility against attractiveness",
    f"Each dot is an HS6 product. <b style='color:{ui.ACCENT}'>Orange</b>: the Top {top_n} by composite score. "
    f"<b>Black</b>: other products kept by likelihood. <span style='color:{ui.MUTED}'>Grey</span>: dropped by "
    "likelihood. Dot size reflects world exports.",
)
plot_df = scored.copy()
plot_df["status"] = "Dropped by likelihood"
plot_df.loc[plot_df.index.isin(kept.index), "status"] = "Kept"
plot_df.loc[plot_df.index.isin(top.index), "status"] = f"Top {top_n}"
plot_df["size"] = 5 + 9 * (np.log10(plot_df["global_export_value"].clip(lower=1e6)) - 6) / 6
plot_df["label"] = ui.product_names(plot_df)
plot_df["hs6"] = plot_df["hs_product_code"].astype(str).str.zfill(6)

fig = go.Figure()
styles = {
    "Dropped by likelihood": dict(color=ui.WASH, line=ui.RULE),
    "Kept": dict(color=ui.INK, line=ui.INK),
    f"Top {top_n}": dict(color=ui.ACCENT, line="white"),
}
for status, sty in styles.items():
    d = plot_df[plot_df["status"] == status]
    fig.add_trace(go.Scatter(
        x=d["feasibility_score"], y=d["attractiveness_score"], mode="markers", name=status,
        marker=dict(size=d["size"], color=sty["color"], line=dict(color=sty["line"], width=0.6),
                    opacity=0.9 if status != "Kept" else 0.55),
        customdata=np.stack([d["label"], d["hs6"], d["likelihood_score"], d["global_export_value"] / 1e9], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>HS6 %{customdata[1]}<br>Feasibility %{x:.0f} · "
                       "Attractiveness %{y:.0f}<br>Likelihood %{customdata[2]:.0f}<br>"
                       "World exports $%{customdata[3]:.1f}B<extra></extra>"),
    ))
fig.update_layout(
    height=560, xaxis=dict(title="Feasibility (Morocco's readiness, percentile)", range=[0, 100]),
    yaxis=dict(title="Attractiveness (value of the market, percentile)", range=[0, 100],
               showgrid=True, gridcolor=ui.WASH, griddash="dot"),
)
ui.plot(fig)

# ============================================================
# TOP N TABLE
# ============================================================
ui.chart_header(
    f"Top {top_n} products",
    f"Ranked by composite score = {feas_pct}% feasibility + {100 - feas_pct}% attractiveness. "
    "All scores are percentile ranks within the candidate pool (0 to 100).",
)
t = top.reset_index(drop=True)
t.index = t.index + 1
t.index.name = "Rank"
st.dataframe(
    t[["hs6", "product", "hs2_name", "composite_score", "feasibility_score",
       "attractiveness_score", "global_export_value"]],
    column_config={
        "hs6": st.column_config.TextColumn("HS6", width="small"),
        "product": st.column_config.TextColumn("Product", width="medium"),
        "industry": st.column_config.TextColumn("Industry (HS4)", width="medium"),
        "hs2_name": st.column_config.TextColumn("Sector", width="small"),
        "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.0f"),
        "feasibility_score": st.column_config.NumberColumn("Feasibility", format="%.0f", width="small"),
        "attractiveness_score": st.column_config.NumberColumn("Attractiveness", format="%.0f", width="small"),
        "global_export_value": st.column_config.NumberColumn("World exports ($)", format="compact"),
    },
    width="stretch", height=min(36 * len(t) + 40, 640),
)

ui.chart_header(f"Top {top_n} by sector", "Number of products in each HS2 chapter.")
by_sector = top.groupby("hs2_name").size().sort_values(ascending=False)
ui.plot(ui.hbar(by_sector.index, by_sector.values))

# ============================================================
# COMPONENT DETAIL
# ============================================================
with st.expander("What drives each score (component percentiles for all kept products)"):
    comp_cols = {
        "hs6": "HS6", "product": "Product", "industry": "Industry (HS4)", "composite_score": "Composite",
        "likelihood_score": "Likelihood",
        "like_fuel": "L: fuel", "like_elec": "L: electricity", "like_vuln": "L: vulnerability",
        "like_cbam": "L: CBAM", "like_growth": "L: growth",
        "feas_density": "F: density", "feas_rca": "F: RCA", "feas_hhi": "F: openness", "feas_distance": "F: distance",
        "attr_pci": "A: PCI", "attr_cog": "A: COG", "attr_market_size": "A: market size",
        "attr_growth": "A: growth", "attr_spillover": "A: spillover",
    }
    detail = kept.sort_values("composite_score", ascending=False)[list(comp_cols)].rename(columns=comp_cols)
    st.dataframe(detail.round(0), hide_index=True, width="stretch", height=420)
    ui.small("L = likelihood component, F = feasibility component, A = attractiveness component. "
             "Each is a percentile rank (0 to 100).")
    U.download_csv(kept, "powershoring_scored_products.csv",
                   f"{theory} | kept {keep_pct}% | {feas_pct}F/{100 - feas_pct}A | {len(kept)} products")

st.divider()
st.page_link("pages/3_Scenarios.py", label="Next: compare all four theories  →")
