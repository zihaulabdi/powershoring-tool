"""Compare: put two saved shortlists side by side."""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui  # noqa: E402
import utils as U  # noqa: E402

ui.page_header(
    "Tools",
    "Compare",
    lede=(
        "Compare two saved results side by side. Save results with the button in the sidebar of "
        "<b>2 · Score and rank</b> or <b>3 · Scenarios</b>."
    ),
)


def add_theory_shortlists():
    """Save each theory's Top N under the user's current settings."""
    r = ui.current_ranking()
    results = ui.scenario_results(ranking=r)
    stamp = f"{r['feas_pct']}F/{100 - r['feas_pct']}A"
    for s, res in results.items():
        ui.save_shortlist(f"{s} · top {r['top_n']} · {stamp}", res["selected"].nlargest(r["top_n"], "composite_score"),
                          f"{s} theory · current settings · top {r['top_n']} products")


saved = ui.saved_shortlists()

# ============================================================
# EMPTY STATE
# ============================================================
if len(saved) < 2:
    ui.note(
        f"You have saved <b>{len(saved)}</b> shortlist{'s' if len(saved) != 1 else ''}. Two are needed. "
        "Save them from steps 2 and 3, or add the four theory shortlists produced by your current settings.",
        accent=True,
    )
    if st.button("Add the four theory shortlists (current settings)", type="primary"):
        add_theory_shortlists()
        st.rerun()
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
names = list(saved.keys())
sb = st.sidebar
sb.header("Shortlists")
name_a = sb.selectbox("Shortlist A", names, index=0, key="cmp_a")
name_b = sb.selectbox("Shortlist B", names, index=1 if len(names) > 1 else 0, key="cmp_b")
level_label = sb.radio("Compare at the level of", ["Products (HS6)", "Industries (HS4)"], key="cmp_level")
level = "HS4" if "HS4" in level_label else "HS6"

sb.header("Manage")
if sb.button("Add the four theory shortlists (current settings)"):
    add_theory_shortlists()
    st.rerun()
to_drop = sb.selectbox("Remove a shortlist", ["—"] + names, key="cmp_drop")
if to_drop != "—" and sb.button("Remove"):
    del st.session_state["saved_scenarios"][to_drop]
    st.rerun()

if name_a == name_b:
    st.info("Pick two different shortlists in the sidebar.")
    st.stop()


# ============================================================
# PREPARE
# ============================================================
def prep(df):
    """One row per code with description, sector, composite score and exports."""
    d = df.copy()
    d["hs6"] = d["hs_product_code"].astype(str).str.zfill(6)
    d["hs4"] = d["hs6"].str[:4]
    if "composite_score" not in d.columns:
        d["composite_score"] = float("nan")
    if level == "HS6":
        d["code"] = d["hs6"]
        d["label"] = ui.product_names(d)
        return d[["code", "label", "hs2_name", "composite_score", "global_export_value"]]
    g = d.groupby(["hs4", "hs2_name"]).agg(composite_score=("composite_score", "mean"),
                                           global_export_value=("global_export_value", "sum")).reset_index()
    g["code"] = g["hs4"]
    g["label"] = g["hs4"].map(lambda c: ui.short_label(U._HS4_DESC_LOOKUP.get(c, c), 60))
    return g[["code", "label", "hs2_name", "composite_score", "global_export_value"]]


A, B = prep(saved[name_a]["products"]), prep(saved[name_b]["products"])
both = set(A["code"]) & set(B["code"])
only_a, only_b = set(A["code"]) - both, set(B["code"]) - both
union = len(set(A["code"]) | set(B["code"]))

for nm, side in [(name_a, "A"), (name_b, "B")]:
    ui.small(f"<b>{side}: {nm}.</b> {saved[nm].get('desc', '')}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("In A", f"{len(A)}")
k2.metric("In B", f"{len(B)}")
k3.metric("In both", f"{len(both)}")
k4.metric("Overlap", f"{100 * len(both) / max(union, 1):.0f}%")

# ============================================================
# SECTOR COMPARISON
# ============================================================
ui.chart_header(
    "Sector composition",
    f"Number of {'products' if level == 'HS6' else 'industries'} from each HS2 chapter. "
    f"<b>Black</b>: A. <b style='color:{ui.ACCENT}'>Orange</b>: B.",
)
sec = pd.concat([A.groupby("hs2_name").size().rename("A"), B.groupby("hs2_name").size().rename("B")],
                axis=1).fillna(0)
sec = sec.loc[(sec["A"] + sec["B"]).sort_values(ascending=True).index]
fig = go.Figure()
fig.add_trace(go.Bar(y=sec.index, x=sec["A"], name="A", orientation="h", marker_color=ui.INK))
fig.add_trace(go.Bar(y=sec.index, x=sec["B"], name="B", orientation="h", marker_color=ui.ACCENT))
fig.update_layout(barmode="group", height=max(280, 46 * len(sec) + 60), bargap=0.3, bargroupgap=0.05,
                  xaxis=dict(title="Count", dtick=1 if sec.values.max() <= 10 else None),
                  yaxis=dict(automargin=True, showline=False, ticks=""))
ui.plot(fig)

# ============================================================
# ENTRY TABLE
# ============================================================
ui.chart_header("Every entry, side by side",
                "Composite scores come from each shortlist's own settings. A blank means the entry is not in that list.")
show = st.radio("Show", ["All", "In both", "Only in A", "Only in B"], horizontal=True, key="cmp_show")
m = A.merge(B[["code", "composite_score"]], on="code", how="outer", suffixes=(" A", " B"))
meta = pd.concat([A, B]).drop_duplicates("code").set_index("code")
m["label"] = m["code"].map(meta["label"])
m["hs2_name"] = m["code"].map(meta["hs2_name"])
m["global_export_value"] = m["code"].map(meta["global_export_value"])
m["status"] = m["code"].map(lambda c: "In both" if c in both else ("Only in A" if c in only_a else "Only in B"))
if show != "All":
    m = m[m["status"] == show]
m = m.sort_values(["status", "composite_score A", "composite_score B"], ascending=[True, False, False])
st.dataframe(
    m[["status", "code", "label", "hs2_name", "composite_score A", "composite_score B", "global_export_value"]],
    column_config={
        "status": st.column_config.TextColumn("Status", width="small"),
        "code": st.column_config.TextColumn(level, width="small"),
        "label": st.column_config.TextColumn("Description", width="medium"),
        "hs2_name": st.column_config.TextColumn("Sector", width="medium"),
        "composite_score A": st.column_config.NumberColumn("Composite in A", format="%.0f"),
        "composite_score B": st.column_config.NumberColumn("Composite in B", format="%.0f"),
        "global_export_value": st.column_config.NumberColumn("World exports ($)", format="compact"),
    },
    hide_index=True, width="stretch", height=520,
)
U.download_csv(m, "powershoring_comparison.csv", f"A: {name_a} | B: {name_b} | level {level}")
