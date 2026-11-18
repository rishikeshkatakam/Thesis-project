"""
iv_dashboard.py
===============
Streamlit interface for NIFTY implied-volatility research.
Run with:  streamlit run iv_dashboard.py

This file contains ONLY user-interface and plotting code. Every calculation
(filtering, binning, aggregation, bootstrap, skew, term structure, statistics)
lives in iv_analysis.py.
"""
from dataclasses import replace
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import iv_analysis as iva

st.set_page_config(page_title="NIFTY IV Research Dashboard", layout="wide")

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
OPT_COLORS = {"CE": "#1f77b4", "PE": "#d62728"}
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 3, "filename": "iv_figure"},
               "displaylogo": False}


# =============================================================================
# Data loading (cached; the source CSV is only ever read)
# =============================================================================
@st.cache_resource(show_spinner="Loading and preparing data...", max_entries=1)
def load_from_path(path: str, mtime: float):
    return iva.prepare_data(iva.load_csv(path))


@st.cache_resource(show_spinner="Loading and preparing data...", max_entries=1)
def load_from_upload(name: str, size: int, _file):
    _file.seek(0)
    return iva.prepare_data(iva.load_csv(_file))


@st.cache_resource(show_spinner="Filtering...", max_entries=6)
def get_filtered(_df, data_key: str, cfg_key: str, _cfg):
    return iva.apply_filters(_df, _cfg)


@st.cache_data(show_spinner="Aggregating...", max_entries=40)
def c_profile(_df, data_key, cfg_key, **kw):
    return iva.historical_profile(_df, **kw)


@st.cache_data(show_spinner="Aggregating...", max_entries=40)
def c_term(_df, data_key, cfg_key, **kw):
    return iva.term_structure(_df, **kw)


@st.cache_data(show_spinner="Aggregating...", max_entries=40)
def c_atm(_df, data_key, cfg_key, **kw):
    return iva.daily_atm_series(_df, **kw)


@st.cache_data(show_spinner="Aggregating...", max_entries=40)
def c_skew(_df, data_key, cfg_key, **kw):
    return iva.daily_skew(_df, **kw)


@st.cache_data(show_spinner="Summarising...", max_entries=40)
def c_summary(_df, data_key, cfg_key, by=None):
    return iva.summarize_iv(_df, by)


# =============================================================================
# Small UI helpers
# =============================================================================
def hex_to_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def style_figure(fig, title, xtitle, ytitle, height=560):
    fig.update_layout(template="simple_white", title=dict(text=title, x=0.01), height=height,
                      font=dict(size=14), legend=dict(orientation="h", y=-0.18),
                      margin=dict(l=60, r=20, t=70, b=60), hovermode="x unified")
    fig.update_xaxes(showline=True, mirror=True, ticks="outside", gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showline=True, mirror=True, ticks="outside", gridcolor="rgba(0,0,0,0.08)")
    return fig


def export_block(fig, table: pd.DataFrame, name: str, table_label="Exact aggregated values behind this figure"):
    """Underlying table + CSV / HTML / optional PNG downloads."""
    with st.expander(table_label):
        st.dataframe(table, width="stretch", height=300)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download table (CSV)", table.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=f"csv_{name}")
    c2.download_button("Download figure (HTML)", fig.to_html(include_plotlyjs="cdn").encode(), f"{name}.html",
                       "text/html", key=f"html_{name}")
    if c3.button("Prepare PNG (needs kaleido)", key=f"pngbtn_{name}"):
        try:
            png = fig.to_image(format="png", width=1200, height=700, scale=3)
            st.download_button("Download PNG", png, f"{name}.png", "image/png", key=f"png_{name}")
        except Exception as e:                                                      # noqa: BLE001
            st.warning(f"PNG export unavailable ({type(e).__name__}). Install kaleido, or use the camera "
                       "icon in the figure toolbar (exports a 3x PNG).")


def group_iter(df, group_cols):
    if not group_cols:
        yield (), df
        return
    for key, sub in df.groupby(list(group_cols), sort=False):
        yield (key if isinstance(key, tuple) else (key,)), sub


def curve_style(key, group_cols, idx, dash_map):
    color, dash = PALETTE[idx % len(PALETTE)], "solid"
    if "option_typ" in group_cols:
        color = OPT_COLORS.get(key[group_cols.index("option_typ")], color)
        others = tuple(k for c, k in zip(group_cols, key) if c != "option_typ")
        if others:
            dash = DASHES[dash_map.setdefault(others, len(dash_map)) % len(DASHES)]
    return color, dash


def plot_profile(profile, group_cols, title, show_iqr, show_ci):
    """Median curve (+ optional IQR and bootstrap-CI bands) and a lower panel with contributing dates."""
    group_cols = list(group_cols)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    dash_map = {}
    has_ci = "ci_low_pct" in profile.columns
    for i, (key, sub) in enumerate(group_iter(profile, group_cols)):
        sub = sub.sort_values("log_moneyness")
        name = " | ".join(map(str, key)) if key else "All selected observations"
        color, dash = curve_style(key, group_cols, i, dash_map)
        x = sub["log_moneyness"]
        if show_iqr:
            fig.add_trace(go.Scatter(x=x, y=sub["q75_pct"], mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip", legendgroup=name), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=sub["q25_pct"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=hex_to_rgba(color, 0.15), name=f"{name}: IQR across dates",
                                     legendgroup=name, hoverinfo="skip"), row=1, col=1)
        if show_ci and has_ci:
            fig.add_trace(go.Scatter(x=x, y=sub["ci_high_pct"], mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip", legendgroup=name), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=sub["ci_low_pct"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=hex_to_rgba(color, 0.35), name=f"{name}: bootstrap CI",
                                     legendgroup=name, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=sub["iv_central_pct"], mode="lines+markers", name=name, legendgroup=name,
                                 line=dict(color=color, dash=dash, width=2.5), marker=dict(size=6),
                                 customdata=sub[["n_dates", "n_obs"]].to_numpy(),
                                 hovertemplate="IV %{y:.2f}%<br>dates %{customdata[0]}, obs %{customdata[1]}"),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=x, y=sub["n_dates"], name=name, legendgroup=name, showlegend=False,
                             marker_color=hex_to_rgba(color, 0.6)), row=2, col=1)
    style_figure(fig, title, "", "")
    fig.update_xaxes(title_text="Log-moneyness ln(K/S)", row=2, col=1)
    fig.update_yaxes(title_text="Implied volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="# dates", row=2, col=1)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.28))
    fig.add_vline(x=0, line_dash="dot", line_color="grey")
    return fig


def need_data(df, what="figure") -> bool:
    if len(df) == 0:
        st.warning(f"No observations remain after the current filters, so the {what} cannot be drawn. "
                   "Relax a filter in the sidebar.")
        return False
    return True


def maturity_text(cfg, params):
    if cfg.custom_days is not None:
        return f"custom {cfg.custom_days[0]:g}-{cfg.custom_days[1]:g} days (POOLED)"
    if cfg.maturity_buckets is None:
        return "ALL maturities POOLED (not maturity-controlled)"
    tag = "compared separately" if params["compare_buckets"] else "POOLED"
    return ", ".join(cfg.maturity_buckets) + f" ({tag})"


def banner(cfg, params, df, extra=""):
    st.markdown(
        f"**Dates:** {cfg.start or 'start'} -> {cfg.end or 'end'} &nbsp;|&nbsp; **Maturity:** {maturity_text(cfg, params)} "
        f"&nbsp;|&nbsp; **Options:** {'/'.join(cfg.option_types)} &nbsp;|&nbsp; "
        f"**Observations:** {len(df):,} on {df['Date'].nunique():,} trading dates {extra}")
    if cfg.maturity_buckets is None and cfg.custom_days is None or (
            cfg.maturity_buckets and len(cfg.maturity_buckets) > 1 and not params["compare_buckets"]):
        st.caption("Pooling several maturities mixes different times to expiry: the curve is an average across "
                   "maturities, not a fixed-maturity smile.")


# =============================================================================
# Load data
# =============================================================================
st.title("NIFTY implied-volatility research dashboard")
st.sidebar.header("1. Data")
up = st.sidebar.file_uploader("Upload CSV (read-only; never overwritten)", type=["csv"])
path = st.sidebar.text_input("...or local file path (best for very large files)", value="")
st.sidebar.caption("Large uploads: start with `streamlit run iv_dashboard.py --server.maxUploadSize 2048`.")

try:
    if up is not None:
        df, report = load_from_upload(up.name, up.size, up)
        data_key = f"up:{up.name}:{up.size}"
    elif path.strip():
        if not os.path.isfile(path.strip()):
            st.error(f"File not found: {path}")
            st.stop()
        df, report = load_from_path(path.strip(), os.path.getmtime(path.strip()))
        data_key = f"path:{path.strip()}:{os.path.getmtime(path.strip())}"
    else:
        st.info("Upload the calculated-IV CSV (or enter its path) in the sidebar to begin.")
        st.stop()
except iva.SchemaError as e:
    st.error(f"The file does not match the expected schema. {e}")
    st.stop()
except Exception as e:                                                            # noqa: BLE001
    st.error(f"Could not read the file: {type(e).__name__}: {e}")
    st.stop()

td = report["trading_dates"]
if len(td) == 0:
    st.error("No valid trading dates could be parsed from the 'Date' column.")
    st.stop()

# ---- date-parsing report (never silent) --------------------------------------------------------------
dr, er = report["trade_date"], report["expiry_date"]
st.sidebar.success(f"{report['n_rows']:,} rows | trading dates {td.min():%d-%b-%Y} to {td.max():%d-%b-%Y}")
if dr["n_unparseable"] or dr["n_missing"] or er["n_unparseable"] or er["n_missing"]:
    st.sidebar.warning(f"Dates: {dr['n_unparseable']:,} unparseable + {dr['n_missing']:,} missing trade dates; "
                       f"{er['n_unparseable']:,} unparseable + {er['n_missing']:,} missing expiry dates. "
                       "Rows are kept but cannot enter date-based analysis.")
if report["iv_looks_like_percent"]:
    st.sidebar.error("Median successful IV > 3: 'IV' may already be in percent, not decimals. Check units.")

# =============================================================================
# Sidebar controls
# =============================================================================
sb = st.sidebar
sb.header("2. Dates")
c1, c2 = sb.columns(2)
start = c1.date_input("Start", value=td.min().date(), min_value=td.min().date(), max_value=td.max().date())
end = c2.date_input("End", value=td.max().date(), min_value=td.min().date(), max_value=td.max().date())
if start > end:
    sb.error("Start date is after end date.")
    st.stop()
sb.caption(f"{iva.trading_days_in_range(report, start, end):,} trading days in selected period")

sb.header("3. Maturity")
mat_mode = sb.radio("Maturity selection", [
    "Overall - all maturities pooled", "Selected bucket(s) - pooled",
    "Selected buckets - compare separately", "Custom day range - pooled"])
buckets, custom_days, compare_buckets = None, None, False
if mat_mode.startswith("Selected"):
    buckets = tuple(sb.multiselect("Maturity buckets", iva.MATURITY_LABELS, default=[iva.MATURITY_LABELS[2]]))
    compare_buckets = "compare" in mat_mode
    if not buckets:
        sb.warning("Select at least one bucket.")
        st.stop()
elif mat_mode.startswith("Custom"):
    a, b = sb.columns(2)
    d_lo = a.number_input("Min days", 0, 5000, 0)
    d_hi = b.number_input("Max days", 0, 5000, 30)
    custom_days = (float(d_lo), float(d_hi))
sb.caption("Days to expiry = T x 365 (fallback: expiry_dt - Date).")

sb.header("4. Option type")
opt_mode = sb.radio("Option type", ["Calls and puts pooled", "Calls only (CE)", "Puts only (PE)",
                                   "Calls vs puts (separate curves)"])
option_types = {"Calls only (CE)": ("CE",), "Puts only (PE)": ("PE",)}.get(opt_mode, ("CE", "PE"))
split_options = opt_mode.startswith("Calls vs")

sb.header("5. Moneyness")
mny = sb.radio("Moneyness", ["All", "Near ATM", "OTM only", "ITM only", "Custom log-moneyness range"])
near_w, lm_custom = 0.02, None
if mny == "Near ATM":
    near_w = sb.number_input("ATM half-width |ln(K/S)|", 0.001, 0.5, 0.02, 0.005, format="%.3f")
elif mny.startswith("Custom"):
    lm_custom = sb.slider("ln(K/S) range", -1.0, 1.0, (-0.2, 0.2), 0.01)
sb.caption("ITM/OTM is option-type aware: CE is OTM if K>S, PE is OTM if K<S.")
mny_mode = {"All": "all", "Near ATM": "near_atm", "OTM only": "otm", "ITM only": "itm"}.get(mny, "custom")

sb.header("6. Liquidity and IV bounds")
min_contracts = sb.number_input("Minimum contracts traded", 0, 10**9, 0)
min_oi = sb.number_input("Minimum open interest", 0, 10**9, 0)
drop_zero = sb.checkbox("Remove zero-volume observations", False)
exp_type = sb.selectbox("Expiry type (data-driven weekly/monthly)", ["all", "weekly", "monthly"])
use_bounds = sb.checkbox("Apply IV bounds (sensitivity analysis)", False)
iv_lo = iv_hi = None
if use_bounds:
    lo_, hi_ = sb.slider("IV range (%)", 0.0, 500.0, (1.0, 200.0), 1.0)
    iv_lo, iv_hi = lo_ / 100, hi_ / 100
sb.caption("Extreme IVs are never removed unless you set bounds here.")

sb.header("7. Aggregation")
n_bins = sb.slider("Number of log-moneyness bins", 5, 120, 40)
lm_lo, lm_hi = sb.slider("Smile range ln(K/S)", -1.0, 1.0, (-0.20, 0.20), 0.01)
daily_agg = sb.selectbox("Within-date aggregation", ["median", "mean"])
across_agg = sb.selectbox("Across-date aggregation", ["median", "mean"])
min_cell = sb.number_input("Min observations per date-bin cell", 1, 1000, 1)
show_iqr = sb.checkbox("Show 25th-75th percentile band", True)
boot = sb.checkbox("Date-level bootstrap CI", False)
n_boot = sb.number_input("Bootstrap replications", 50, 5000, 300, 50) if boot else 300
atm_w = sb.number_input("ATM half-width for Figs 4-5 (|ln(K/S)|)", 0.001, 0.5, 0.02, 0.005, format="%.3f")
roll_w = sb.number_input("Rolling window (trading dates)", 2, 250, 20)
dte_w = sb.number_input("Term-structure DTE bin width (days)", 1, 90, 7)
wing_lo, wing_hi = sb.slider("Skew wing band |ln(K/S)|", 0.0, 0.5, (0.02, 0.10), 0.01)

cfg = iva.FilterConfig(start=start.isoformat(), end=end.isoformat(), maturity_buckets=buckets, custom_days=custom_days,
                       option_types=option_types, moneyness_mode=mny_mode, near_atm_width=float(near_w),
                       lm_range=lm_custom, expiry_type=exp_type, min_contracts=float(min_contracts),
                       min_open_int=float(min_oi), drop_zero_volume=drop_zero, iv_min=iv_lo, iv_max=iv_hi,
                       success_only=True)
params = {"compare_buckets": compare_buckets}


def filtered(c):
    """Cached filtering; returns (frame, log)."""
    return get_filtered(df, data_key, c.key(), c)


fdf, flog = filtered(cfg)                                                  # main (successful IV only)
fdf_status, _ = filtered(replace(cfg, success_only=False, iv_min=None, iv_max=None))   # incl. failures

# groups used by Figures 1-2
g1 = (["maturity_bucket"] if compare_buckets else []) + (["option_typ"] if split_options else [])
g2 = (["maturity_bucket"] if compare_buckets else []) + ["option_typ"]

tabs = st.tabs(["Summary", "1 Historical smile", "2 Call vs put", "3 Daily smile", "4 Term structure",
                "5 ATM IV history", "6 Skew history", "Data quality", "Methodology"])

# =============================================================================
# Summary
# =============================================================================
with tabs[0]:
    banner(cfg, params, fdf)
    st.subheader("Observation counts after each filter")
    st.dataframe(flog, width="stretch", hide_index=True)
    q = iva.quality_counts(fdf_status)
    cols = st.columns(6)
    cols[0].metric("Observations (any status)", f"{q['n_obs']:,}")
    cols[1].metric("Successful IV", f"{q['success']:,}")
    cols[2].metric("Invalid price", f"{q['invalid_price']:,}")
    cols[3].metric("Solver failure", f"{q['solver_failure']:,}")
    cols[4].metric("Trading dates", f"{q['n_trading_dates']:,}")
    cols[5].metric("Expiries", f"{q['n_expiries']:,}")
    if q["missing_input"] or q["other_status"]:
        st.caption(f"Also: {q['missing_input']:,} missing_input, {q['other_status']:,} other status.")
    if need_data(fdf, "summary statistics"):
        st.subheader("IV statistics (% , successful observations)")
        s_all = c_summary(fdf, data_key, cfg.key())
        st.dataframe(s_all.round(3), width="stretch")
        st.subheader("By option type")
        st.dataframe(c_summary(fdf, data_key, cfg.key(), by="option_typ").round(3), width="stretch")
        st.subheader("By maturity bucket")
        s_mat = c_summary(fdf, data_key, cfg.key(), by="maturity_bucket")
        st.dataframe(s_mat.round(3), width="stretch")
        st.download_button("Download maturity summary (CSV)", s_mat.to_csv().encode(), "iv_summary_by_maturity.csv",
                           "text/csv")

# =============================================================================
# Figure 1
# =============================================================================
with tabs[1]:
    banner(cfg, params, fdf)
    if need_data(fdf, "smile"):
        prof = c_profile(fdf, data_key, cfg.key(), lm_range=(lm_lo, lm_hi), n_bins=n_bins, group_cols=g1,
                         daily_agg=daily_agg, across_agg=across_agg, min_obs_per_cell=int(min_cell),
                         bootstrap=boot, n_boot=int(n_boot))
        title = (f"Date-balanced historical IV smile ({across_agg} of daily {daily_agg}s; "
                 f"{start:%b %Y} - {end:%b %Y})")
        fig = plot_profile(prof, g1, title, show_iqr, boot)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key="fig1")
        st.caption("Lower panel: number of trading dates contributing to each bin. Bands: IQR across dates "
                   "(light) and date-level bootstrap CI of the central value (darker, if enabled). "
                   "Descriptive only - no statistical-significance claim.")
        export_block(fig, prof, "fig1_historical_smile")

# =============================================================================
# Figure 2
# =============================================================================
with tabs[2]:
    banner(cfg, params, fdf)
    if need_data(fdf, "call/put skew"):
        prof2 = c_profile(fdf, data_key, cfg.key() + "|f2", lm_range=(lm_lo, lm_hi), n_bins=n_bins, group_cols=g2,
                          daily_agg=daily_agg, across_agg=across_agg, min_obs_per_cell=int(min_cell),
                          bootstrap=boot, n_boot=int(n_boot))
        fig2 = plot_profile(prof2, g2, "Call vs put historical IV curves (date-balanced)", show_iqr, boot)
        st.plotly_chart(fig2, width="stretch", config=PLOT_CONFIG, key="fig2")
        st.caption("Colour = option type; line style = maturity bucket (when buckets are compared). "
                   "Calls and puts differ in which side is OTM, so compare them with that in mind.")
        export_block(fig2, prof2, "fig2_call_vs_put")

# =============================================================================
# Figure 3
# =============================================================================
with tabs[3]:
    cfg3 = replace(cfg, start=None, end=None, maturity_buckets=None, custom_days=None, moneyness_mode="all",
                   lm_range=None, expiry_type="all")
    base3, _ = filtered(cfg3)
    st.caption("Daily smile ignores the sidebar date range, maturity and moneyness filters (you pick the date and expiry); "
               "option-type, liquidity and IV filters still apply.")
    if need_data(base3, "daily smile"):
        dts = iva.available_dates(base3)
        pick = st.date_input("Trading date", value=dts.max().date(), min_value=dts.min().date(),
                             max_value=dts.max().date(), key="d3")
        if pd.Timestamp(pick) not in set(dts):
            near = ", ".join(f"{d:%d-%b-%Y}" for d in iva.nearest_dates(base3, pick))
            st.warning(f"No observations for {pick:%d-%b-%Y} under the current filters (holiday, weekend, or filtered out). "
                       f"Nearest available dates: {near}. Nothing was substituted.")
        else:
            ex = iva.available_expiries(base3, pick)
            opts = {f"{r.expiry_dt:%d-%b-%Y} ({r.days_to_expiry:g}d, {r.n_obs} obs, {r.n_CE} CE / {r.n_PE} PE)": r.expiry_dt
                    for r in ex.itertuples()}
            label = st.selectbox("Expiry (available on this date)", list(opts))
            sm = iva.daily_smile(base3, pick, opts[label])
            fig3 = go.Figure()
            for typ in ("CE", "PE"):
                s = sm[sm["option_typ"] == typ]
                if len(s):
                    fig3.add_trace(go.Scatter(x=s["log_moneyness"], y=s["IV_pct"], mode="markers+lines", name=typ,
                                              marker=dict(color=OPT_COLORS[typ], size=7),
                                              line=dict(color=OPT_COLORS[typ], width=1, dash="dot"),
                                              customdata=s[["strike_pr", "contracts", "open_int"]].to_numpy(),
                                              hovertemplate="K=%{customdata[0]:.0f}<br>IV %{y:.2f}%<br>"
                                                            "contracts %{customdata[1]:.0f}, OI %{customdata[2]:.0f}"))
            fig3.add_vline(x=0, line_dash="dash", line_color="grey", annotation_text="ATM (K = S)")
            style_figure(fig3, f"IV smile - {pick:%d-%b-%Y}, expiry {opts[label]:%d-%b-%Y}",
                         "Log-moneyness ln(K/S)", "Implied volatility (%)")
            fig3.update_xaxes(title_text="Log-moneyness ln(K/S)")
            fig3.update_yaxes(title_text="Implied volatility (%)")
            st.markdown(f"**Observations:** {len(sm):,} (successful IV only)")
            st.plotly_chart(fig3, width="stretch", config=PLOT_CONFIG, key="fig3")
            export_block(fig3, sm, "fig3_daily_smile")

# =============================================================================
# Figure 4
# =============================================================================
with tabs[4]:
    cfg4 = replace(cfg, maturity_buckets=None, custom_days=None, moneyness_mode="all", lm_range=None)
    st.caption("Term structure spans maturities by construction, so the sidebar maturity and moneyness filters are "
               "ignored here. ATM = |ln(K/S)| <= the half-width set in the sidebar.")
    mode4 = st.radio("View", ["Daily term structure", "Historical median term structure"], horizontal=True)
    split4 = st.selectbox("Separate curves by", ["None", "Option type", "Expiry type (weekly/monthly)"])
    g4 = {"None": [], "Option type": ["option_typ"]}.get(split4, ["expiry_type"])
    if mode4.startswith("Daily"):
        base4, _ = filtered(replace(cfg4, start=None, end=None))
        if need_data(base4, "term structure"):
            dts = iva.available_dates(base4)
            pick = st.date_input("Trading date", value=dts.max().date(), min_value=dts.min().date(),
                                 max_value=dts.max().date(), key="d4")
            if pd.Timestamp(pick) not in set(dts):
                near = ", ".join(f"{d:%d-%b-%Y}" for d in iva.nearest_dates(base4, pick))
                st.warning(f"No observations on {pick:%d-%b-%Y}. Nearest available dates: {near}.")
            else:
                ts = c_term(base4, data_key, cfg4.key() + f"|{pick}", atm_width=float(atm_w), mode="daily",
                            date=pd.Timestamp(pick), group_cols=g4)
                fig4 = go.Figure()
                for i, (key, sub) in enumerate(group_iter(ts, g4)):
                    nm = " | ".join(map(str, key)) or "ATM IV"
                    col, _ = curve_style(key, g4, i, {})
                    fig4.add_trace(go.Scatter(x=sub["days_to_expiry"], y=sub["iv_pct"], mode="lines+markers", name=nm,
                                              line=dict(color=col, width=2.5),
                                              customdata=sub[["n_obs"]].to_numpy(),
                                              hovertemplate="DTE %{x:.0f}<br>IV %{y:.2f}%<br>obs %{customdata[0]}"))
                style_figure(fig4, f"DAILY ATM term structure - {pick:%d-%b-%Y}", "", "")
                fig4.update_xaxes(title_text="Days to expiry"); fig4.update_yaxes(title_text="ATM implied volatility (%)")
                st.markdown(f"**Observations:** {int(ts['n_obs'].sum()) if len(ts) else 0:,} ATM contracts, {len(ts)} points")
                if len(ts) == 0:
                    st.warning("No ATM observations on this date. Increase the ATM half-width.")
                st.plotly_chart(fig4, width="stretch", config=PLOT_CONFIG, key="fig4d")
                export_block(fig4, ts, "fig4_daily_term_structure")
    else:
        base4, _ = filtered(cfg4)
        if need_data(base4, "term structure"):
            ts = c_term(base4, data_key, cfg4.key(), atm_width=float(atm_w), mode="historical", group_cols=g4,
                        dte_bin_width=int(dte_w), daily_agg=daily_agg, across_agg=across_agg)
            fig4 = go.Figure()
            for i, (key, sub) in enumerate(group_iter(ts, g4)):
                nm = " | ".join(map(str, key)) or "Historical median ATM IV"
                col, _ = curve_style(key, g4, i, {})
                sub = sub.sort_values("dte_center")
                if show_iqr:
                    fig4.add_trace(go.Scatter(x=sub["dte_center"], y=sub["q75_pct"], mode="lines", line=dict(width=0),
                                              showlegend=False, hoverinfo="skip"))
                    fig4.add_trace(go.Scatter(x=sub["dte_center"], y=sub["q25_pct"], mode="lines", line=dict(width=0),
                                              fill="tonexty", fillcolor=hex_to_rgba(col, 0.15), name=f"{nm}: IQR",
                                              hoverinfo="skip"))
                fig4.add_trace(go.Scatter(x=sub["dte_center"], y=sub["iv_central_pct"], mode="lines+markers", name=nm,
                                          line=dict(color=col, width=2.5), customdata=sub[["n_dates"]].to_numpy(),
                                          hovertemplate="DTE %{x:.0f}<br>IV %{y:.2f}%<br>dates %{customdata[0]}"))
            style_figure(fig4, f"HISTORICAL date-balanced ATM term structure ({start:%b %Y} - {end:%b %Y})", "", "")
            fig4.update_xaxes(title_text="Days to expiry (bin centre)"); fig4.update_yaxes(title_text="ATM implied volatility (%)")
            st.markdown(f"**Observations:** {len(base4):,} contracts in range; ATM cells aggregated per date. "
                        "Hover shows contributing dates per bin.")
            st.plotly_chart(fig4, width="stretch", config=PLOT_CONFIG, key="fig4h")
            export_block(fig4, ts, "fig4_historical_term_structure")

# =============================================================================
# Figure 5
# =============================================================================
with tabs[5]:
    mode5 = st.radio("Series", ["As per sidebar option-type selection", "CE vs PE", "Weekly vs monthly expiries"], horizontal=True)
    cfg5 = replace(cfg, moneyness_mode="all", lm_range=None)
    if mode5 == "CE vs PE":
        cfg5 = replace(cfg5, option_types=("CE", "PE"))
    base5, _ = filtered(cfg5)
    banner(cfg5, params, base5)
    if need_data(base5, "ATM IV history"):
        g5 = {"CE vs PE": ["option_typ"], "Weekly vs monthly expiries": ["expiry_type"]}.get(mode5, [])
        at = c_atm(base5, data_key, cfg5.key() + mode5, atm_width=float(atm_w), group_cols=g5, agg=daily_agg, window=int(roll_w))
        rc = f"iv_pct_roll{int(roll_w)}"
        fig5 = go.Figure()
        for i, (key, sub) in enumerate(group_iter(at, g5)):
            nm = " | ".join(map(str, key)) or "Daily median ATM IV"
            col, _ = curve_style(key, g5, i, {})
            fig5.add_trace(go.Scatter(x=sub["Date"], y=sub["iv_pct"], mode="lines", name=nm, opacity=0.55,
                                      line=dict(color=col, width=1.2), customdata=sub[["n_obs"]].to_numpy(),
                                      hovertemplate="%{y:.2f}% (obs %{customdata[0]})"))
            fig5.add_trace(go.Scatter(x=sub["Date"], y=sub[rc], mode="lines", name=f"{nm}: rolling {int(roll_w)}d median",
                                      line=dict(color=col, width=3, dash="dash")))
        style_figure(fig5, "Daily ATM implied volatility", "", "")
        fig5.update_xaxes(title_text="Trading date"); fig5.update_yaxes(title_text="ATM implied volatility (%)")
        st.plotly_chart(fig5, width="stretch", config=PLOT_CONFIG, key="fig5")
        st.caption("Rolling median runs over the last N observed trading dates of each series. If maturities are "
                   "pooled, the constant-maturity composition of ATM IV changes through the expiry cycle.")
        export_block(fig5, at, "fig5_atm_iv_history")

# =============================================================================
# Figure 6
# =============================================================================
with tabs[6]:
    cfg6 = replace(cfg, option_types=("CE", "PE"), moneyness_mode="all", lm_range=None)
    base6, _ = filtered(cfg6)
    banner(cfg6, params, base6)
    st.caption("Skew = median IV of OTM puts in the put wing minus median IV of OTM calls in the mirrored call wing "
               "(moneyness-band measure; NOT a 25-delta risk reversal). Both option types are always used here.")
    if need_data(base6, "skew history"):
        g6 = ["maturity_bucket"] if compare_buckets else []
        sk = c_skew(base6, data_key, cfg6.key(), wing_lo=float(wing_lo), wing_hi=float(wing_hi), group_cols=g6,
                    agg=daily_agg, window=int(roll_w))
        if len(sk) == 0:
            st.warning("No dates have observations in both wings. Widen the wing band.")
        else:
            rc = f"skew_pct_roll{int(roll_w)}"
            fig6 = go.Figure()
            for i, (key, sub) in enumerate(group_iter(sk, g6)):
                nm = " | ".join(map(str, key)) or "Daily skew"
                col, _ = curve_style(key, g6, i, {})
                fig6.add_trace(go.Scatter(x=sub["Date"], y=sub["skew_pct"], mode="lines", name=nm, opacity=0.55,
                                          line=dict(color=col, width=1.2),
                                          customdata=sub[["n_put", "n_call"]].to_numpy(),
                                          hovertemplate="%{y:.2f} pts (put obs %{customdata[0]}, call obs %{customdata[1]})"))
                fig6.add_trace(go.Scatter(x=sub["Date"], y=sub[rc], mode="lines", name=f"{nm}: rolling {int(roll_w)}d median",
                                          line=dict(color=col, width=3, dash="dash")))
            fig6.add_hline(y=0, line_color="grey", line_dash="dot")
            style_figure(fig6, f"Wing skew: IV(OTM put wing) - IV(OTM call wing), |ln(K/S)| in [{wing_lo:g}, {wing_hi:g}]", "", "")
            fig6.update_xaxes(title_text="Trading date"); fig6.update_yaxes(title_text="Skew (IV percentage points)")
            st.markdown(f"**Observations:** {len(sk):,} date-level skew values.")
            st.plotly_chart(fig6, width="stretch", config=PLOT_CONFIG, key="fig6")
            export_block(fig6, sk, "fig6_skew_history")

# =============================================================================
# Data quality
# =============================================================================
with tabs[7]:
    scope = st.radio("Scope", ["Sidebar-filtered data", "Entire dataset"], horizontal=True)
    dq = fdf_status if scope.startswith("Sidebar") else df
    st.markdown(f"**{len(dq):,} observations in scope**")
    st.subheader("Date parsing")
    date_tbl = pd.DataFrame([{"column": k, "n_total": v["n_total"], "n_valid": v["n_valid"],
                              "n_missing": v["n_missing"], "n_unparseable": v["n_unparseable"],
                              "min": v["min"], "max": v["max"]} for k, v in (("Date", dr), ("expiry_dt", er))])
    st.dataframe(date_tbl, width="stretch", hide_index=True)
    if dr["unparseable_examples"] or er["unparseable_examples"]:
        st.caption(f"Unparseable examples - Date: {dr['unparseable_examples']}; expiry_dt: {er['unparseable_examples']}")
    st.caption(f"Rows where T disagrees with expiry-date minus date by >1.5 days: {report['n_T_inconsistent']:,}; "
               f"days fallback used: {report['n_days_fallback']:,}; expiry not after trade date: "
               f"{report['n_expiry_not_after_date']:,}.")
    chk = report["iv_pct_check"]
    if chk is None:
        st.caption("IV_pct column absent; not used.")
    elif chk["consistent"] is None:
        st.caption("IV_pct could not be verified (no rows with both IV and IV_pct). Dashboard uses IV x 100.")
    else:
        st.caption(f"IV_pct vs 100 x IV on {chk['n_compared']:,} rows: max abs difference {chk['max_abs_diff']:.2e} -> "
                   f"{'units verified' if chk['consistent'] else 'MISMATCH - do not use IV_pct'}. The dashboard always plots IV x 100.")
    st.subheader("IV status by year")
    st.dataframe(iva.status_by_year(dq), width="stretch")
    ok_dq = dq[dq["IV_status"] == iva.STATUS_SUCCESS]
    a, b = st.columns(2)
    a.subheader("IV quantiles (successful)")
    a.dataframe(iva.iv_quantiles(ok_dq), hide_index=True, width="stretch")
    b.subheader("IV by maturity bucket (successful)")
    b.dataframe(iva.summarize_iv(ok_dq, "maturity_bucket").round(2), width="stretch")
    st.subheader("Extreme IV observations (not removed)")
    n_ext = st.slider("Rows to show", 5, 200, 25)
    e1, e2 = st.columns(2)
    e1.caption("Highest IV"); e1.dataframe(iva.extreme_observations(ok_dq, n_ext, "high"), hide_index=True)
    e2.caption("Lowest IV"); e2.dataframe(iva.extreme_observations(ok_dq, n_ext, "low"), hide_index=True)
    st.subheader("Invalid-price and solver-failure observations")
    stat = st.selectbox("Status", ["invalid_price", "solver_failure", "missing_input", "all non-success"])
    bad = iva.failed_observations(dq, None if stat.startswith("all") else stat)
    st.markdown(f"**{len(bad):,} rows**")
    st.dataframe(bad.head(1000), hide_index=True, width="stretch")
    st.download_button("Download these rows (CSV, first 200,000)", bad.head(200_000).to_csv(index=False).encode(),
                       "failed_observations.csv", "text/csv")

# =============================================================================
# Methodology
# =============================================================================
with tabs[8]:
    st.markdown("""
**Date-balanced smile (Figs 1-2).** For every trading date and fixed log-moneyness bin the within-date median IV is
computed; the plotted value is the median across dates of those daily medians, with the 25th/75th percentiles across
dates as the band. Each date has equal weight irrespective of the number of listed contracts. Bin counts show how many
dates contribute.

**Bootstrap CI.** Trading dates are resampled with replacement (the resampling unit is the date, not the contract);
the band is the 2.5-97.5 percentile of the resampled central value. It ignores serial dependence between adjacent days
and the strong overlap of expiry cycles, so treat it as descriptive.

**Maturity.** Buckets use calendar days to expiry (T x 365). Pooled maturities are always labelled; they are not a
fixed-maturity smile.

**Term structure (Fig 4).** ATM = |ln(K/S)| within the chosen half-width. Daily: one median IV per expiry.
Historical: (date, expiry) -> (date, DTE bin) -> median across dates.

**Skew (Fig 6).** IV(OTM puts, -b_hi <= ln(K/S) <= -b_lo) minus IV(OTM calls, b_lo <= ln(K/S) <= b_hi). Not delta-based.

**Limitations.** See README: spot-based BS with a rough dividend yield and 91-day T-bill rate; end-of-day closes of
illiquid options; ATM defined against spot rather than the forward; data-driven weekly/monthly classification;
survivorship of successful IV solves (invalid/failed contracts are excluded from the figures and are often deep ITM or
very far OTM).
""")
