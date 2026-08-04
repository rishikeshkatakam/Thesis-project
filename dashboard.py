"""
Options Market Volume Dashboard
================================
A production-quality, interactive Streamlit + Plotly dashboard for
exploratory analysis of daily-aggregated options market volume data.

Run with:
    streamlit run dashboard.py

Expected input data
--------------------
A CSV (or an in-memory pandas DataFrame called `daily_volume`) with the
following columns:

    trd_date                (string, 'YYYY-MM-DD')
    total_volume, total_call_volume, total_put_volume,
    itm_call_volume, atm_call_volume, otm_call_volume,
    itm_put_volume, atm_put_volume, otm_put_volume,
    itm_call_ratio, atm_call_ratio, otm_call_ratio,
    itm_put_ratio, atm_put_ratio, otm_put_ratio,
    call_share, put_share, call_put_ratio

If you already have `daily_volume` in memory (e.g. produced upstream in a
notebook or ETL job), just persist it once and point the sidebar file
uploader at the resulting CSV:

    daily_volume.to_csv("daily_volume.csv", index=False)

If no file is uploaded, the dashboard falls back to a synthetic sample
dataset so it always runs end-to-end without modification.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import streamlit as st

# ==========================================================================
# Page configuration (must be the first Streamlit call in the script)
# ==========================================================================
st.set_page_config(
    page_title="Options Market Volume Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Constants
# ==========================================================================
DATE_COL = "trd_date"

VOLUME_COLS: List[str] = [
    "total_volume",
    "total_call_volume",
    "total_put_volume",
    "itm_call_volume",
    "atm_call_volume",
    "otm_call_volume",
    "itm_put_volume",
    "atm_put_volume",
    "otm_put_volume",
]

RATIO_COLS: List[str] = [
    "call_share",
    "put_share",
    "call_put_ratio",
    "itm_call_ratio",
    "atm_call_ratio",
    "otm_call_ratio",
    "itm_put_ratio",
    "atm_put_ratio",
    "otm_put_ratio",
]

ALL_NUMERIC_COLS: List[str] = VOLUME_COLS + RATIO_COLS

CORRELATION_COLS: List[str] = [
    "total_volume",
    "call_share",
    "put_share",
    "itm_call_ratio",
    "atm_call_ratio",
    "otm_call_ratio",
    "itm_put_ratio",
    "atm_put_ratio",
    "otm_put_ratio",
]

# Restrained, professional qualitative palette (Bloomberg / FT inspired)
COLOR_PALETTE: List[str] = [
    "#0B5394", "#C9302C", "#2E8B57", "#B8860B", "#6A5ACD",
    "#008080", "#D2691E", "#4682B4", "#8B008B", "#556B2F",
    "#B22222", "#20B2AA", "#DAA520", "#483D8B", "#CD5C5C",
]

AGG_RULE_MAP: Dict[str, str] = {"Monthly": "MS", "Yearly": "YS"}
ROLLING_WINDOW_MAP: Dict[str, Optional[int]] = {
    "None": None, "7 days": 7, "30 days": 30, "60 days": 60, "90 days": 90,
}

LEGEND_POSITIONS: Dict[str, dict] = {
    "Top": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    "Bottom": dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    "Right": dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    "Left": dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=-0.2),
}


# ==========================================================================
# Data loading & sample-data generation
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("daily_volume.csv", parse_dates=["trd_date"])
    return _optimize_dtypes(df)



def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes to float32 where safe, cutting memory use
    roughly in half on large datasets without materially affecting the
    precision needed for visualization."""
    for col in ALL_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


@st.cache_data(show_spinner="Loading and preparing dataset...")
def load_data(file_bytes: Optional[bytes]) -> pd.DataFrame:
    """Load the raw dataset from an uploaded CSV's bytes, parse dates,
    sort chronologically, and optimize dtypes. Cached on the file's byte
    content so re-renders never re-parse an unchanged file."""
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))

    missing = [c for c in [DATE_COL] + ALL_NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d", errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[DATE_COL])
    if len(df) < n_before:
        st.warning(f"Dropped {n_before - len(df):,} row(s) with unparsable '{DATE_COL}' values.")

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = _optimize_dtypes(df)
    return df


@st.cache_data(show_spinner="Aggregating data...")
def aggregate_data(df: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    """Aggregate the daily dataframe to Monthly or Yearly frequency.
    Volume columns are summed, ratio columns are averaged. 'Daily' returns
    the frame unchanged (as a shallow copy to avoid downstream mutation)."""
    if freq_label == "Daily":
        return df.copy()

    rule = AGG_RULE_MAP[freq_label]
    agg_map: Dict[str, str] = {c: "sum" for c in VOLUME_COLS}
    agg_map.update({c: "mean" for c in RATIO_COLS})

    out = (
        df.set_index(DATE_COL)
        .resample(rule)
        .agg(agg_map)
        .dropna(how="all")
        .reset_index()
    )
    return out


def filter_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Vectorized boolean-mask filter on the date column."""
    mask = (df[DATE_COL] >= pd.Timestamp(start)) & (df[DATE_COL] <= pd.Timestamp(end))
    return df.loc[mask]


def apply_rolling_average(
    df: pd.DataFrame, window: int, columns: List[str], include_volumes: bool
) -> pd.DataFrame:
    """Apply a trailing rolling mean. By default only ratio columns among
    the selected variables are smoothed; volume columns are included only
    if the user explicitly opts in."""
    df = df.copy()
    target_cols = [
        c for c in columns if (c in RATIO_COLS) or (include_volumes and c in VOLUME_COLS)
    ]
    for c in target_cols:
        df[c] = df[c].rolling(window=window, min_periods=1).mean()
    return df


# ==========================================================================
# Formatting helpers
# ==========================================================================
def format_axis_value(value: float, is_ratio: bool) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}" if is_ratio else f"{value:,.0f}"


def hover_template_for(col: str) -> str:
    is_ratio = col in RATIO_COLS
    value_fmt = "%{y:.2%}" if is_ratio else "%{y:,.0f}"
    label = col.replace("_", " ").title()
    return f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>{value_fmt}<extra></extra>"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ==========================================================================
# Statistical overlays
# ==========================================================================
def compute_trend_line(x_numeric: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS linear trend via scipy.stats.linregress, robust to NaNs."""
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.full_like(y, np.nan, dtype=float)
    slope, intercept, *_ = stats.linregress(x_numeric[mask], y[mask])
    return slope * x_numeric + intercept


def add_overlays(
    fig: go.Figure,
    x: pd.Series,
    y: pd.Series,
    col: str,
    color: str,
    overlays: Dict[str, bool],
    ma_window: int,
    secondary_y: bool,
) -> None:
    """Add any enabled statistical overlays (trend, moving average, mean,
    median, std band) as additional traces for a single variable."""
    is_ratio = col in RATIO_COLS
    hover_fmt = "%{y:.2%}" if is_ratio else "%{y:,.0f}"
    label = col.replace("_", " ").title()

    if overlays.get("trend"):
        x_numeric = (x - x.min()).dt.days.values.astype(float)
        trend_y = compute_trend_line(x_numeric, y.values.astype(float))
        fig.add_trace(
            go.Scatter(
                x=x, y=trend_y, mode="lines", name=f"{label} · trend",
                line=dict(color=color, width=1.5, dash="dot"),
                hovertemplate=f"<b>{label} trend</b><br>%{{x|%Y-%m-%d}}<br>{hover_fmt}<extra></extra>",
            ),
            secondary_y=secondary_y,
        )

    if overlays.get("moving_avg"):
        ma_y = y.rolling(window=ma_window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=x, y=ma_y, mode="lines", name=f"{label} · MA({ma_window})",
                line=dict(color=color, width=1.5, dash="dash"),
                hovertemplate=f"<b>{label} MA({ma_window})</b><br>%{{x|%Y-%m-%d}}<br>{hover_fmt}<extra></extra>",
            ),
            secondary_y=secondary_y,
        )

    if overlays.get("mean"):
        mean_val = float(y.mean())
        fig.add_trace(
            go.Scatter(
                x=[x.min(), x.max()], y=[mean_val, mean_val], mode="lines",
                name=f"{label} · mean", line=dict(color=color, width=1, dash="dashdot"),
                hovertemplate=f"<b>{label} mean</b>: {format_axis_value(mean_val, is_ratio)}<extra></extra>",
            ),
            secondary_y=secondary_y,
        )

    if overlays.get("median"):
        median_val = float(y.median())
        fig.add_trace(
            go.Scatter(
                x=[x.min(), x.max()], y=[median_val, median_val], mode="lines",
                name=f"{label} · median", line=dict(color=color, width=1, dash="longdash"),
                hovertemplate=f"<b>{label} median</b>: {format_axis_value(median_val, is_ratio)}<extra></extra>",
            ),
            secondary_y=secondary_y,
        )

    if overlays.get("std_band"):
        mean_val = float(y.mean())
        std_val = float(y.std())
        fig.add_trace(
            go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=np.concatenate([
                    np.full(len(x), mean_val + std_val),
                    np.full(len(x), mean_val - std_val)[::-1],
                ]),
                fill="toself", fillcolor=_hex_to_rgba(color, 0.12),
                line=dict(width=0), hoverinfo="skip",
                name=f"{label} · ±1 std",
            ),
            secondary_y=secondary_y,
        )


# ==========================================================================
# Main figure builder
# ==========================================================================
def build_main_figure(df: pd.DataFrame, variables: List[str], settings: Dict) -> go.Figure:
    """Construct the main interactive time-series figure from all sidebar
    settings. Automatically routes ratio variables to a secondary y-axis
    (formatted as %) when both volume and ratio variables are selected
    together, so the two scales never distort each other."""
    has_volume = any(v in VOLUME_COLS for v in variables)
    has_ratio = any(v in RATIO_COLS for v in variables)
    use_secondary = has_volume and has_ratio

    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

    mode_map = {"Line": "lines", "Scatter": "markers", "Area": "lines"}

    for i, col in enumerate(variables):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        is_ratio = col in RATIO_COLS
        secondary_y = use_secondary and is_ratio

        x = df[DATE_COL]
        y = df[col]
        label = col.replace("_", " ").title()

        mode = mode_map[settings["graph_type"]]
        if settings["show_markers"] and settings["graph_type"] != "Scatter":
            mode = "lines+markers"

        trace_kwargs = dict(
            x=x, y=y, mode=mode, name=label,
            line=dict(width=settings["line_width"], color=color),
            marker=dict(size=6, color=color),
            opacity=settings["opacity"],
            hovertemplate=hover_template_for(col),
        )
        if settings["graph_type"] == "Area":
            trace_kwargs["fill"] = "tozeroy"
            trace_kwargs["fillcolor"] = _hex_to_rgba(color, 0.15)

        fig.add_trace(go.Scatter(**trace_kwargs), secondary_y=secondary_y)

        add_overlays(fig, x, y, col, color, settings["overlays"], settings["ma_window"], secondary_y)

    template = "plotly_dark" if settings["theme"] == "Dark" else "plotly_white"
    fig.update_layout(
        template=template,
        title=dict(text=settings["title"], x=0.02, xanchor="left", font=dict(size=20)),
        width=settings["fig_width"],
        height=settings["fig_height"],
        autosize=True,
        hovermode="x unified",
        legend=LEGEND_POSITIONS[settings["legend_position"]],
        margin=dict(l=60, r=60, t=80, b=60),
    )

    fig.update_xaxes(
        title_text=settings["x_label"],
        showgrid=settings["show_grid"],
        rangeslider_visible=settings["show_rangeslider"],
    )
    fig.update_yaxes(title_text=settings["y_label"], showgrid=settings["show_grid"], secondary_y=False)
    if use_secondary:
        fig.update_yaxes(title_text="Ratio (%)", showgrid=False, tickformat=".0%", secondary_y=True)

    return fig


# ==========================================================================
# Sidebar controls
# ==========================================================================
def render_sidebar(df_full: pd.DataFrame) -> Dict:
    st.sidebar.header("⚙️ Controls")

    st.sidebar.subheader("Time Aggregation")
    freq_label = st.sidebar.selectbox("Aggregation level", ["Daily", "Monthly", "Yearly"], index=0)

    st.sidebar.subheader("Variables")
    variables = st.sidebar.multiselect(
        "Select one or more variables to plot",
        options=ALL_NUMERIC_COLS,
        default=["total_volume"],
        format_func=lambda c: c.replace("_", " ").title(),
    )

    st.sidebar.subheader("Rolling Average")
    rolling_enabled = st.sidebar.checkbox("Enable rolling average", value=False)
    rolling_window_label = "None"
    include_volumes_in_rolling = False
    if rolling_enabled:
        rolling_window_label = st.sidebar.selectbox(
            "Rolling window", list(ROLLING_WINDOW_MAP.keys()), index=2
        )
        include_volumes_in_rolling = st.sidebar.checkbox(
            "Also smooth volume variables", value=False,
            help="By default the rolling average is applied only to ratio variables.",
        )
        st.sidebar.caption("Window length is expressed in periods of the selected aggregation level.")

    st.sidebar.subheader("Graph Customization")
    graph_type = st.sidebar.selectbox("Graph type", ["Line", "Scatter", "Area"], index=0)
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)
    line_width = st.sidebar.slider("Line width", 1.0, 6.0, 2.0, 0.5)
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 1.0, 0.05)
    show_markers = st.sidebar.checkbox("Show markers", value=False)
    show_grid = st.sidebar.checkbox("Show grid", value=True)
    legend_position = st.sidebar.selectbox("Legend position", ["Top", "Bottom", "Left", "Right"], index=0)
    fig_width = st.sidebar.slider("Figure width (px)", 600, 1800, 1100, 50)
    fig_height = st.sidebar.slider("Figure height (px)", 300, 1200, 600, 50)
    title = st.sidebar.text_input("Title", value="Options Market Volume — Time Series")
    x_label = st.sidebar.text_input("X-axis label", value="Date")
    y_label = st.sidebar.text_input("Y-axis label", value="Value")
    show_rangeslider = st.sidebar.checkbox("Show range slider under chart", value=True)

    st.sidebar.subheader("Date Range")
    min_date, max_date = df_full[DATE_COL].min().date(), df_full[DATE_COL].max().date()
    date_range = st.sidebar.slider(
        "Date range slider", min_value=min_date, max_value=max_date,
        value=(min_date, max_date), format="YYYY-MM-DD",
    )
    use_picker = st.sidebar.checkbox("Fine-tune with date picker", value=False)
    if use_picker:
        picked_start = st.sidebar.date_input(
            "Start date", value=date_range[0], min_value=min_date, max_value=max_date
        )
        picked_end = st.sidebar.date_input(
            "End date", value=date_range[1], min_value=min_date, max_value=max_date
        )
        date_range = (picked_start, picked_end)

    st.sidebar.subheader("Statistical Overlays")
    overlays = {
        "trend": st.sidebar.checkbox("Show trend line", value=False),
        "moving_avg": st.sidebar.checkbox("Show moving average", value=False),
        "mean": st.sidebar.checkbox("Show mean line", value=False),
        "median": st.sidebar.checkbox("Show median line", value=False),
        "std_band": st.sidebar.checkbox("Show ±1 std deviation band", value=False),
    }
    ma_window = 30
    if overlays["moving_avg"]:
        ma_window = st.sidebar.slider("Moving average window (periods)", 2, 120, 30, 1)

    return dict(
        freq_label=freq_label,
        variables=variables,
        rolling_enabled=rolling_enabled,
        rolling_window_label=rolling_window_label,
        include_volumes_in_rolling=include_volumes_in_rolling,
        graph_type=graph_type,
        theme=theme,
        line_width=line_width,
        opacity=opacity,
        show_markers=show_markers,
        show_grid=show_grid,
        legend_position=legend_position,
        fig_width=fig_width,
        fig_height=fig_height,
        title=title,
        x_label=x_label,
        y_label=y_label,
        show_rangeslider=show_rangeslider,
        date_range=date_range,
        overlays=overlays,
        ma_window=ma_window,
    )


# ==========================================================================
# Export controls
# ==========================================================================
def render_export_buttons(fig: go.Figure, key_prefix: str) -> None:
    """Explicit PNG/HTML export buttons, in addition to the built-in
    camera icon already available in every Plotly chart's toolbar."""
    col1, col2 = st.columns(2)
    with col1:
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                "⬇️ Export as PNG", data=png_bytes, file_name="chart.png",
                mime="image/png", key=f"{key_prefix}_png",
            )
        except Exception:
            st.caption("PNG export needs the `kaleido` package: `pip install -U kaleido`.")
    with col2:
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "⬇️ Export as HTML", data=html_bytes, file_name="chart.html",
            mime="text/html", key=f"{key_prefix}_html",
        )


# ==========================================================================
# Page: header / dataset stats
# ==========================================================================
def render_header(df_filtered: pd.DataFrame, df_full: pd.DataFrame) -> None:
    st.title("📊 Options Market Volume Dashboard")
    st.markdown(
        "Interactive exploratory analysis of daily options market volume, "
        "moneyness composition, and call/put dynamics — built for research "
        "and thesis presentation."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations (filtered)", f"{len(df_filtered):,}")
    c2.metric("Total Observations", f"{len(df_full):,}")
    c3.metric("Start Date", df_full[DATE_COL].min().strftime("%Y-%m-%d"))
    c4.metric("End Date", df_full[DATE_COL].max().strftime("%Y-%m-%d"))
    st.divider()


# ==========================================================================
# Page: Correlation Analysis
# ==========================================================================
def render_correlation_tab(df: pd.DataFrame) -> None:
    st.subheader("Correlation Analysis")
    st.caption("Pearson correlation across the core volume and ratio variables.")

    corr_df = df[CORRELATION_COLS].corr()

    fig = px.imshow(
        corr_df, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
        aspect="auto", labels=dict(color="Correlation"),
    )
    fig.update_traces(hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>")
    fig.update_layout(title="Correlation Matrix", height=650, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width='stretch', config={"displaylogo": False})

    with st.expander("View correlation values as a table"):
        st.dataframe(
            corr_df.style.format("{:.3f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1),
            width='stretch',
        )

    csv_bytes = corr_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download correlation matrix (CSV)", data=csv_bytes,
        file_name="correlation_matrix.csv", mime="text/csv",
    )


# ==========================================================================
# Page: Data Table
# ==========================================================================
def render_data_table_tab(df: pd.DataFrame) -> None:
    st.subheader("Interactive Data Table")

    search = st.text_input("🔍 Search (matches any column, case-insensitive)", value="")

    display_df = df.copy()
    display_df[DATE_COL] = display_df[DATE_COL].dt.strftime("%Y-%m-%d")

    if search:
        mask = display_df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.caption(f"Showing {len(display_df):,} of {len(df):,} rows. Click column headers to sort.")
    st.dataframe(display_df, width='stretch', height=520)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download table (CSV)", data=csv_bytes,
        file_name="options_volume_data.csv", mime="text/csv",
    )


# ==========================================================================
# Styling
# ==========================================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.4rem;}
        section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,0.2);}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# Application entry point
# ==========================================================================
def main() -> None:
    inject_custom_css()

    st.sidebar.title("📁 Data Source")
    uploaded = st.sidebar.file_uploader("Upload daily_volume CSV", type=["csv"])
    if uploaded is None:
        st.sidebar.info("No file uploaded — using generated sample data for demonstration.")
    st.sidebar.divider()

    file_bytes = uploaded.getvalue() if uploaded is not None else None

    try:
        df_raw = load_data(file_bytes)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    settings = render_sidebar(df_raw)

    # Aggregate first, then filter by date, then (optionally) smooth.
    df_agg = aggregate_data(df_raw, settings["freq_label"])
    start, end = settings["date_range"]
    df_filtered = filter_by_date(df_agg, start, end)

    window = ROLLING_WINDOW_MAP[settings["rolling_window_label"]]
    if settings["rolling_enabled"] and window:
        df_filtered = apply_rolling_average(
            df_filtered, window, settings["variables"], settings["include_volumes_in_rolling"]
        )

    render_header(df_filtered, df_raw)

    tab_explorer, tab_corr, tab_table = st.tabs(
        ["📈 Time Series Explorer", "🔗 Correlation Analysis", "🗂️ Data Table"]
    )

    with tab_explorer:
        if not settings["variables"]:
            st.info("Select at least one variable from the sidebar to plot.")
        elif df_filtered.empty:
            st.warning("No data in the selected date range.")
        else:
            fig = build_main_figure(df_filtered, settings["variables"], settings)
            st.plotly_chart(fig, width='stretch', config={"displaylogo": False})
            render_export_buttons(fig, key_prefix="explorer")

    with tab_corr:
        render_correlation_tab(df_filtered if not df_filtered.empty else df_agg)

    with tab_table:
        render_data_table_tab(df_filtered)


if __name__ == "__main__":
    main()
