"""Front-end helpers for the Powershoring tool.

Everything visual lives here so every page looks and behaves the same:
  - the editorial theme (fonts, colours, Plotly template), modelled on the
    UM6P Economic Complexity Unit site;
  - small layout components (page header, chart header, notes);
  - settings that persist across pages (Stage 1 thresholds, ranking weights);
  - cached computations shared by several pages.

Backend logic (filters, scoring) stays in utils.py.
"""

import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import utils as U

# ============================================================
# PALETTE (Economic Complexity Unit site)
# ============================================================
INK = "#1a1a1a"        # main text, bars
MUTED = "#6b6b6b"      # captions, secondary text
RULE = "#c8c8c2"       # lines, gridlines
WASH = "#ececea"       # light fills, empty heatmap cells
PAPER = "#f6f6f3"      # panel backgrounds
ACCENT = "#d84825"     # the single highlight colour
ACCENT_SOFT = "#f0b9a8"

SCENARIO_SHORT = {
    "No Prior": "No Prior",
    "Electricity Cost": "Electricity",
    "Carbon Regulation": "Carbon",
    "Disruption Opportunity": "Disruption",
}


# ============================================================
# THEME
# ============================================================
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea,
.stSelectbox, .stRadio, .stCheckbox, .stSlider, button {{
    font-family: 'Source Sans 3', -apple-system, 'Segoe UI', sans-serif !important;
}}
.stMarkdown p, .stMarkdown li {{ font-size: 16.5px; line-height: 1.6; color: {INK}; }}

h1, h2, h3, h4 {{
    font-family: 'Crimson Pro', Georgia, serif !important;
    font-weight: 600 !important; color: {INK} !important; letter-spacing: -0.005em;
}}
h1 {{ font-size: 2.6rem !important; line-height: 1.12 !important; padding-bottom: 0.2rem !important; }}
h2 {{ font-size: 1.75rem !important; margin-top: 1.6rem !important; }}
h3 {{ font-size: 1.35rem !important; }}

.block-container, [data-testid="stMainBlockContainer"] {{
    max-width: 1180px; padding-top: 5rem !important; padding-bottom: 4rem;
}}

/* Kicker: small mono label above titles */
.ecu-kicker {{
    font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
    letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED}; margin-bottom: 0.4rem;
}}
.ecu-byline {{ color: {MUTED}; font-size: 15px; margin: -0.3rem 0 1.2rem 0; }}
.ecu-byline span {{ color: {ACCENT}; }}
.ecu-lede {{ font-size: 18px; line-height: 1.6; color: {INK}; max-width: 780px; margin-bottom: 0.6rem; }}

/* Chart header: bold title with a black rule on the left, grey subtitle */
.ecu-chart {{ border-left: 3px solid {INK}; padding-left: 12px; margin: 1.6rem 0 0.6rem 0; }}
.ecu-chart .t {{ font-weight: 700; font-size: 17px; color: {INK}; }}
.ecu-chart .s {{ color: {MUTED}; font-size: 15px; line-height: 1.5; max-width: 760px; margin-top: 2px; }}

/* Note box */
.ecu-note {{
    background: {PAPER}; border: 1px solid {WASH}; border-radius: 3px;
    padding: 12px 16px; color: {INK}; font-size: 15.5px; line-height: 1.55; margin: 0.8rem 0;
}}
.ecu-note.accent {{ border-left: 3px solid {ACCENT}; }}

/* Numbered steps on the overview */
.ecu-step {{ border-top: 1px solid {RULE}; padding-top: 10px; }}
.ecu-step .n {{ font-family: 'JetBrains Mono', monospace; color: {ACCENT}; font-size: 13px; letter-spacing: 0.08em; }}
.ecu-step .h {{ font-family: 'Crimson Pro', Georgia, serif; font-size: 22px; font-weight: 600; margin: 2px 0 4px 0; }}
.ecu-step .b {{ color: {MUTED}; font-size: 15px; line-height: 1.5; }}
.ecu-step .k {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.06em; color: {ACCENT}; margin-top: 8px; }}

/* Metrics */
[data-testid="stMetricLabel"] p {{
    font-family: 'JetBrains Mono', monospace !important; font-size: 11.5px !important;
    letter-spacing: 0.1em; text-transform: uppercase; color: {MUTED} !important;
}}
[data-testid="stMetricValue"] {{ font-family: 'Crimson Pro', Georgia, serif !important; font-size: 2.1rem !important; color: {INK}; }}

/* Tabs */
.stTabs [data-baseweb="tab"] p {{ font-size: 15px; font-weight: 600; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {PAPER}; border-right: 1px solid {WASH}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
    letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED} !important;
    font-weight: 500 !important; margin-top: 0.8rem !important;
}}

/* Thin rules instead of heavy dividers */
hr {{ border-color: {WASH} !important; margin: 1.4rem 0 !important; }}

/* Footnotes and small print */
.ecu-small {{ color: {MUTED}; font-size: 13.5px; line-height: 1.5; }}
</style>
"""


def apply_theme():
    """Inject fonts and CSS, and register the Plotly template."""
    st.markdown(_CSS, unsafe_allow_html=True)
    pio.templates["ecu"] = _plotly_template()
    pio.templates.default = "ecu"


def _plotly_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Source Sans 3, Source Sans Pro, sans-serif", size=14, color=INK),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[INK, ACCENT, MUTED, RULE],
        xaxis=dict(showgrid=True, gridcolor=WASH, griddash="dot", zeroline=False,
                   showline=True, linecolor=RULE, ticks="outside", tickcolor=RULE,
                   title=dict(font=dict(size=13, color=MUTED))),
        yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=RULE,
                   ticks="outside", tickcolor=RULE, title=dict(font=dict(size=13, color=MUTED))),
        hoverlabel=dict(bgcolor="white", bordercolor=RULE, font=dict(family="Source Sans 3", color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=13, color=MUTED), title=dict(text="")),
        margin=dict(l=10, r=20, t=20, b=40),
    )
    return t


# ============================================================
# LAYOUT COMPONENTS
# ============================================================
def page_header(kicker, title, lede=None, byline=None):
    st.markdown(f"<div class='ecu-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    if byline:
        st.markdown(f"<div class='ecu-byline'>{byline}</div>", unsafe_allow_html=True)
    if lede:
        st.markdown(f"<div class='ecu-lede'>{lede}</div>", unsafe_allow_html=True)


def chart_header(title, subtitle=None):
    sub = f"<div class='s'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"<div class='ecu-chart'><div class='t'>{title}</div>{sub}</div>",
                unsafe_allow_html=True)


def note(text, accent=False):
    cls = "ecu-note accent" if accent else "ecu-note"
    st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)


def small(text):
    st.markdown(f"<div class='ecu-small'>{text}</div>", unsafe_allow_html=True)


def plot(fig, height=None):
    """Render a Plotly figure full width with the toolbar hidden."""
    if height:
        fig.update_layout(height=height)
    # Streamlit 1.50: plotly_chart is full width by default; extra keyword
    # arguments trigger a deprecation warning, so pass only `config`.
    st.plotly_chart(fig, config={"displayModeBar": False})


# ============================================================
# LABELS AND FORMATTING
# ============================================================
def short_label(text, max_len=58):
    """Shorten long HS descriptions for chart labels and tables."""
    s = str(text or "").strip()
    s = re.split(r"\s\(", s)[0].strip()          # drop parentheticals
    s = re.sub(r"\s*;\s*", ", ", s)               # "Aluminium; unwrought" -> "Aluminium, unwrought"
    if len(s) > max_len:
        s = s[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return s


fmt_usd = U.format_dollars


# ============================================================
# PERSISTENT SETTINGS
# ============================================================
# Streamlit forgets a widget's value when you leave its page. We keep a copy
# in st.session_state["_store"] and restore it when the page is shown again,
# so the same choice applies on every page.
def _store():
    return st.session_state.setdefault("_store", {})


def persist(widget, label, key, default, **kwargs):
    """Render `widget` (e.g. st.slider) with a value that survives page changes.

    The stored value is written into the widget's key before it is drawn, and
    an on_change callback copies every user change back into the store. The
    callback runs before the script reruns, so the store is never stale.
    """
    store = _store()
    store.setdefault(key, default)
    st.session_state[key] = store[key]

    def _remember():
        store[key] = st.session_state[key]

    return widget(label, key=key, on_change=_remember, **kwargs)


def reset_settings(prefix):
    """Forget every stored setting whose key starts with `prefix`."""
    for k in [k for k in _store() if k.startswith(prefix)]:
        del _store()[k]
    for k in [k for k in st.session_state.keys() if str(k).startswith(prefix)]:
        del st.session_state[k]


# ----- Stage 1 (candidate pool) -----
# Starting positions of the controls. They are not a recommended setting;
# every result in the tool follows from whatever the user chooses.
STAGE1_DEFAULTS = {
    "s1_energy": 75, "s1_elec": 50, "s1_trade": 15, "s1_logic": "Either threshold (OR)",
    "s1_legacy": False, "s1_cbam": False, "s1_green": False, "s1_topics": [], "s1_rca": 0.0,
}


def stage1_settings():
    s = {k: _store().get(k, v) for k, v in STAGE1_DEFAULTS.items()}
    return s


@st.cache_data(show_spinner=False)
def _pool_cached(energy, elec, trade, logic, legacy, cbam, green, topics, rca):
    df = U.load_data()
    return U.apply_stage1_filter(
        df, energy_percentile=energy, elec_percentile=elec, trade_percentile=trade,
        filter_logic=logic, use_legacy=legacy, cbam_only=cbam,
        green_only=green, green_topics=list(topics), rca_threshold=rca,
        return_metadata=True,
    )


def candidate_pool(settings=None):
    """Stage 1 pool for the given (or current) settings. Returns (df, meta)."""
    s = settings or stage1_settings()
    logic = "AND" if "AND" in s["s1_logic"] else "OR"
    df, meta = _pool_cached(s["s1_energy"], s["s1_elec"], s["s1_trade"], logic,
                            s["s1_legacy"], s["s1_cbam"], s["s1_green"],
                            tuple(s["s1_topics"]), s["s1_rca"])
    return df.copy(), meta


# ----- Ranking (feasibility / attractiveness) -----
RANK_DEFAULTS = {"r_feas_pct": 60, "r_top_n": 30}
FEAS_LABELS = {
    "density": ("Capability proximity (density)", "How close the product is to what Morocco already exports."),
    "rca": ("Existing export strength (RCA)", "Morocco's revealed comparative advantage in the product today."),
    "hhi": ("Market openness (1/HHI)", "How fragmented global supply is. Fragmented markets are easier to enter."),
    "distance": ("Short trade distance", "Products that travel shorter distances favour regional producers."),
}
ATTR_LABELS = {
    "pci": ("Product complexity (PCI)", "How much know-how the product embodies."),
    "cog": ("Diversification value (COG)", "How many new related products entry would open up for Morocco."),
    "market_size": ("Global market size", "Total world exports of the product."),
    "growth": ("Market growth", "Annual growth of world exports, 2012 to 2024."),
    "spillover": ("Spillover potential", "Centrality of the industry in the input-output network."),
}


def ranking_controls():
    """Shared sidebar block for the composite ranking. Returns a settings dict."""
    st.sidebar.header("Ranking")
    feas_pct = persist(st.sidebar.slider, "Weight on feasibility (%)", "r_feas_pct", 60,
                       min_value=0, max_value=100, step=5,
                       help="Composite score = this share × feasibility + the rest × attractiveness.")
    st.sidebar.caption(f"{feas_pct}% feasibility · {100 - feas_pct}% attractiveness")
    top_n = persist(st.sidebar.slider, "Size of each shortlist (Top N)", "r_top_n", 30,
                    min_value=10, max_value=50, step=5)

    with st.sidebar.expander("Feasibility components"):
        feas_w = {k: persist(st.slider, lab, f"r_f_{k}", U.DEFAULT_FEAS_WEIGHTS[k],
                             min_value=0, max_value=100, help=h)
                  for k, (lab, h) in FEAS_LABELS.items()}
    with st.sidebar.expander("Attractiveness components"):
        attr_w = {k: persist(st.slider, lab, f"r_a_{k}", U.DEFAULT_ATTR_WEIGHTS[k],
                             min_value=0, max_value=100, help=h)
                  for k, (lab, h) in ATTR_LABELS.items()}
    if sum(feas_w.values()) == 0 or sum(attr_w.values()) == 0:
        st.sidebar.error("Give at least one feasibility and one attractiveness component a weight above 0.")
        st.stop()
    if st.sidebar.button("Reset ranking controls", key="reset_rank"):
        reset_settings("r_")
        st.rerun()
    return {"feas_pct": feas_pct, "top_n": top_n, "feas_w": feas_w, "attr_w": attr_w}


def current_ranking():
    """Ranking settings as last chosen by the user (without drawing widgets)."""
    st_ = _store()
    return {
        "feas_pct": st_.get("r_feas_pct", 60),
        "top_n": st_.get("r_top_n", 30),
        "feas_w": {k: st_.get(f"r_f_{k}", U.DEFAULT_FEAS_WEIGHTS[k]) for k in FEAS_LABELS},
        "attr_w": {k: st_.get(f"r_a_{k}", U.DEFAULT_ATTR_WEIGHTS[k]) for k in ATTR_LABELS},
    }


def settings_summary(ranking=None):
    """One-line description of the current simulation settings."""
    s = stage1_settings()
    r = ranking or current_ranking()
    logic = "and" if "AND" in s["s1_logic"] else "or"
    extra = []
    if s["s1_legacy"]:
        extra.append("legacy exclusions")
    if s["s1_cbam"]:
        extra.append("CBAM only")
    if s["s1_green"]:
        extra.append("green supply chains only")
    if s["s1_rca"]:
        extra.append(f"Morocco RCA ≥ {s['s1_rca']}")
    pool = (f"energy ≥ {s['s1_energy']}th pct {logic} electricity ≥ {s['s1_elec']}th pct, "
            f"trade ≥ {s['s1_trade']}th pct" + (f" ({', '.join(extra)})" if extra else ""))
    return (f"<b>Your settings.</b> Candidate pool: {pool}. Ranking: {r['feas_pct']}% feasibility / "
            f"{100 - r['feas_pct']}% attractiveness, Top {r['top_n']}. Change them in step 1 and in the sidebar.")


# ============================================================
# CACHED SCENARIO RUNS
# ============================================================
@st.cache_data(show_spinner=False)
def _scenarios_cached(stage1_items, feas_items, attr_items, feas_share, names):
    pool, _ = candidate_pool(dict(stage1_items))
    return U.run_all_scenarios(pool, dict(feas_items), dict(attr_items), feas_share, list(names))


def _freeze(d):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items()))


def scenario_results(ranking=None, stage1=None, names=None):
    """Run the four scenarios on the current (or given) settings, cached."""
    r = ranking or {"feas_pct": 60, "feas_w": U.DEFAULT_FEAS_WEIGHTS, "attr_w": U.DEFAULT_ATTR_WEIGHTS}
    s1 = stage1 or stage1_settings()
    s1 = {k: (list(v) if isinstance(v, tuple) else v) for k, v in s1.items()}
    return _scenarios_cached(_freeze(s1), _freeze(r["feas_w"]), _freeze(r["attr_w"]),
                             r["feas_pct"] / 100, tuple(names or U.SCENARIO_DEFS.keys()))


# ============================================================
# SAVED SHORTLISTS (for the Compare page)
# ============================================================
def save_shortlist(name, products, description):
    st.session_state.setdefault("saved_scenarios", {})[name] = {
        "products": products.copy(), "desc": description,
    }


def saved_shortlists():
    return st.session_state.get("saved_scenarios", {})


# ============================================================
# CHART BUILDERS
# ============================================================
def hbar(labels, values, highlight=None, value_fmt=",.0f", hover=None, x_title="", height=None):
    """Ranked horizontal bar chart: black bars, optional accent highlight.

    `labels`/`values` are in display order (top first). `highlight` is a list
    of booleans the same length.
    """
    labels, values = list(labels)[::-1], list(values)[::-1]
    colors = [INK] * len(values)
    if highlight is not None:
        colors = [ACCENT if h else INK for h in list(highlight)[::-1]]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        hovertemplate=(hover or "<b>%{y}</b><br>%{x:" + value_fmt + "}") + "<extra></extra>",
        texttemplate="%{x:" + value_fmt + "}", textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color=MUTED),
    ))
    fig.update_layout(
        height=height or max(260, 28 * len(values) + 60),
        xaxis=dict(title=x_title, showticklabels=False, showgrid=False, showline=False, ticks=""),
        yaxis=dict(showline=False, ticks="", automargin=True),
        bargap=0.35, margin=dict(l=10, r=60, t=10, b=20),
    )
    return fig


def scenario_dots(table, names, code_label="code", score_col="composite_score"):
    """Dot matrix: one row per code, one column per scenario.

    Filled accent dot = in that scenario's Top N; hollow grey = not. The
    average composite score is printed on the right. `table` comes from
    utils.robustness_table and is already sorted (top row first).
    """
    rows = table.iloc[::-1]  # Plotly draws bottom-up
    y = [f"{short_label(d, 50)}  ·  {c}" for d, c in zip(rows["description"], rows[code_label])]
    fig = go.Figure()
    for j, s in enumerate(names):
        hit = rows[s].values
        fig.add_trace(go.Scatter(
            x=[j] * len(rows), y=y, mode="markers",
            marker=dict(size=15, color=[ACCENT if h else "white" for h in hit],
                        line=dict(color=[ACCENT if h else RULE for h in hit], width=1.5)),
            customdata=["in the shortlist" if h else "not in the shortlist" for h in hit],
            hovertemplate="<b>%{y}</b><br>" + s + ": %{customdata}<extra></extra>",
            showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[len(names) - 0.25] * len(rows), y=y, mode="text",
        text=[f"{v:.0f}" for v in rows[score_col]], textposition="middle right",
        textfont=dict(size=13, color=INK), hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=34 * len(rows) + 90,
        xaxis=dict(tickvals=list(range(len(names))) + [len(names) + 0.15],
                   ticktext=[SCENARIO_SHORT.get(s, s) for s in names] + ["Score"],
                   side="top", showgrid=False, showline=False, ticks="",
                   range=[-0.6, len(names) + 0.6], tickfont=dict(size=13, color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=WASH, griddash="dot", showline=False, ticks="",
                   automargin=True, tickfont=dict(size=13), range=[-0.7, len(rows) - 0.3]),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def count_heatmap(matrix, colorbar_title=""):
    """Heatmap of counts (rows x scenarios) in the accent ramp, zeros left blank."""
    z = matrix.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[SCENARIO_SHORT.get(c, c) for c in matrix.columns], y=list(matrix.index),
        colorscale=[[0, "white"], [0.001, "#fbe3db"], [1, ACCENT]], zmin=0,
        text=[[str(int(v)) if v else "" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=13), xgap=3, ygap=3, showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(
        height=30 * len(matrix) + 80,
        xaxis=dict(side="top", showline=False, ticks="", showgrid=False, tickfont=dict(color=MUTED)),
        yaxis=dict(autorange="reversed", showline=False, ticks="", automargin=True),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def product_names(df, max_len=60):
    """Short HS6 product names, falling back to the HS4 heading when blank."""
    hs4 = df["hs_product_code"].astype(str).str.zfill(6).str[:4]
    desc = df["description"].fillna("").astype(str)
    fallback = hs4.map(lambda c: U._HS4_DESC_LOOKUP.get(c, ""))
    desc = desc.where(desc.str.strip() != "", fallback)
    return desc.map(lambda d: short_label(d, max_len))
