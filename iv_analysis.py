"""
iv_analysis.py
==============
Analytical engine for the NIFTY implied-volatility research dashboard.

This module has NO dependency on Streamlit or Plotly, so it can be imported
directly in a thesis notebook or script:

    import iv_analysis as iva
    raw = iva.load_csv("nifty_IV_calculated.csv")
    df, report = iva.prepare_data(raw)
    cfg = iva.FilterConfig(maturity_buckets=("16-30 days",))
    sub, log = iva.apply_filters(df, cfg)
    smile = iva.historical_profile(sub, lm_range=(-0.15, 0.15), n_bins=30)

Conventions
-----------
* IV is stored as a decimal (0.18 = 18%). Columns ending in ``_pct`` are x100.
* ``log_moneyness = ln(strike_pr / spot_close)``.
* ``days_to_expiry = round(T * 365)`` (fallback: expiry_dt - Date in calendar days).
* The original CSV / raw DataFrame is never modified.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. Schema
# ----------------------------------------------------------------------------
SCHEMA_COLUMNS = [
    "instrument", "symbol", "expiry_dt", "strike_pr", "option_typ",
    "open", "high", "low", "close", "settle_pr", "contracts", "val_inlakh",
    "open_int", "chg_in_oi", "timestamp", "bhav_format", "trd_date",
    "Date", "spot_close", "price_ratio", "Moneyness", "Div Yield%",
    "rfr_91d_pct", "T", "r_decimal", "q_decimal", "market_price",
    "theoretical_price", "IV", "IV_pct", "IV_status",
]
# Columns the analysis cannot work without
REQUIRED_COLUMNS = ["Date", "expiry_dt", "strike_pr", "spot_close", "option_typ",
                    "contracts", "open_int", "T", "IV", "IV_status"]
# Columns used if present
OPTIONAL_COLUMNS = ["IV_pct", "Moneyness"]

STATUS_SUCCESS = "success"
STATUS_INVALID_PRICE = "invalid_price"
STATUS_SOLVER_FAILURE = "solver_failure"
STATUS_MISSING_INPUT = "missing_input"

# (label, lower day, upper day) in calendar days to expiry, inclusive
MATURITY_BUCKETS = [
    ("0-7 days", 0, 7), ("8-15 days", 8, 15), ("16-30 days", 16, 30),
    ("31-60 days", 31, 60), ("61-90 days", 61, 90), ("91-180 days", 91, 180),
    ("181-365 days", 181, 365), ("Above 365 days", 366, np.inf),
]
MATURITY_LABELS = [b[0] for b in MATURITY_BUCKETS]
_BUCKET_UPPER_EDGES = np.array([b[2] for b in MATURITY_BUCKETS[:-1]], dtype=float)


class SchemaError(ValueError):
    """Raised when the uploaded file lacks columns required by the analysis."""


# ----------------------------------------------------------------------------
# 1. Robust date parsing
# ----------------------------------------------------------------------------
_FAST_FORMATS = ["%Y-%m-%d", "%d-%b-%Y"]   # ISO and 28-Apr-2016 (case-insensitive month)


def _elementwise_parse(x):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def _parse_unique_strings(u: pd.Series) -> pd.Series:
    """Parse a Series of UNIQUE date strings (index must be unique)."""
    out = pd.Series(np.full(len(u), np.datetime64("NaT"), dtype="datetime64[ns]"), index=u.index)
    remaining = (u != "").to_numpy().copy()
    for fmt in _FAST_FORMATS:
        if not remaining.any():
            break
        p = pd.to_datetime(u[remaining], format=fmt, errors="coerce")
        got = p.notna()
        out.loc[p.index[got]] = p[got].to_numpy("datetime64[ns]")
        remaining[u.index.get_indexer(p.index[got])] = False
    if remaining.any():
        rest = u[remaining]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = pd.to_datetime(rest, format="mixed", errors="coerce")   # pandas >= 2.0
        except (TypeError, ValueError):                                      # older pandas
            p = pd.to_datetime(rest.map(_elementwise_parse), errors="coerce")
        out.loc[p.index] = p.to_numpy("datetime64[ns]")
    return out


def parse_mixed_dates(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Parse a column containing mixed date formats (e.g. '28-Apr-2016',
    '01-APR-2016', '2025-05-29'). Never raises on bad values and never drops
    rows: unparseable entries become NaT and are counted in the report.

    Only the UNIQUE strings are parsed, so a 1.3M-row column costs a few
    thousand parse operations.
    """
    s = pd.Series(series)
    txt = s.astype("string").str.strip()
    codes, uniques = pd.factorize(txt)                         # NA -> code -1
    u = pd.Series(np.asarray(uniques, dtype=object).astype(str))
    parsed_u = _parse_unique_strings(u).to_numpy("datetime64[ns]")
    vals = np.full(len(s), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = codes >= 0
    vals[ok] = parsed_u[codes[ok]]
    out = pd.Series(vals, index=s.index, name=s.name)

    missing = ((codes < 0) | (txt.fillna("").to_numpy() == ""))
    unparseable = out.isna().to_numpy() & ~missing
    report = {
        "n_total": int(len(s)),
        "n_missing": int(missing.sum()),
        "n_unparseable": int(unparseable.sum()),
        "n_valid": int(out.notna().sum()),
        "min": out.min() if out.notna().any() else pd.NaT,
        "max": out.max() if out.notna().any() else pd.NaT,
        "unparseable_examples": list(pd.unique(txt[unparseable].dropna()))[:10],
    }
    return out, report


# ----------------------------------------------------------------------------
# 2. Loading and preparation
# ----------------------------------------------------------------------------
def load_csv(source) -> pd.DataFrame:
    """
    Read only the columns the analysis needs from a path or file-like object.
    The source file is opened read-only and never modified.
    """
    header = list(pd.read_csv(source, nrows=0).columns)
    if hasattr(source, "seek"):
        source.seek(0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise SchemaError(
            f"Missing required column(s): {missing}. "
            f"Expected columns follow the schema: {SCHEMA_COLUMNS}")
    usecols = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ["close"] if c in header]
    str_cols = {c: str for c in ["Date", "expiry_dt", "option_typ", "IV_status", "Moneyness"] if c in usecols}
    return pd.read_csv(source, usecols=usecols, dtype=str_cols, low_memory=False)


def classify_expiry_types(expiry: pd.Series) -> pd.Series:
    """
    Data-driven weekly/monthly classification: within each calendar month, the
    LATEST expiry present in the dataset is 'monthly'; any earlier expiry in
    that month is 'weekly'. Before weekly contracts existed (only one expiry
    per month) every expiry is therefore 'monthly'. See README for limitations.
    """
    uniq = pd.Series(pd.DatetimeIndex(expiry.dropna().unique()).sort_values())
    last_in_month = uniq.groupby([uniq.dt.year, uniq.dt.month]).transform("max")
    labels = pd.Series(np.where(uniq == last_in_month, "monthly", "weekly"), index=uniq)
    return expiry.map(labels)


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Convert a raw DataFrame into the analysis frame and a data-quality report.
    ``raw`` is not modified. No rows are dropped.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {missing}")

    report: dict = {"n_rows": int(len(raw))}
    date, report["trade_date"] = parse_mixed_dates(raw["Date"])
    expiry, report["expiry_date"] = parse_mixed_dates(raw["expiry_dt"])
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")

    typ = raw["option_typ"].astype("string").str.strip().str.upper()
    status = raw["IV_status"].astype("string").str.strip().str.lower().fillna("unknown")

    df = pd.DataFrame({
        "Date": date, "expiry_dt": expiry,
        "strike_pr": num("strike_pr"), "spot_close": num("spot_close"),
        "option_typ": typ.astype("category"),
        "contracts": num("contracts"), "open_int": num("open_int"),
        "T": num("T"), "IV": num("IV"),
        "IV_status": status.astype("category"),
    }, index=raw.index)
    if "close" in raw.columns:
        df["close"] = num("close")
    if "Moneyness" in raw.columns:
        df["moneyness_csv"] = raw["Moneyness"].astype("string").str.strip().str.upper().astype("category")

    # --- days to expiry: T*365, fallback to calendar-day difference -----------
    T = df["T"].to_numpy(float)
    diff = (expiry - date).dt.days.astype("float64").to_numpy()
    days = np.rint(np.where(np.isfinite(T), T * 365.0, diff))
    df["days_to_expiry"] = days
    report["n_days_fallback"] = int((~np.isfinite(T) & np.isfinite(diff)).sum())
    report["n_T_inconsistent"] = int((np.isfinite(T) & np.isfinite(diff) & (np.abs(T * 365 - diff) > 1.5)).sum())
    report["n_expiry_not_after_date"] = int((diff <= 0).sum())
    report["n_days_missing"] = int((~np.isfinite(days)).sum())

    # --- maturity bucket ----------------------------------------------------------
    codes = np.searchsorted(_BUCKET_UPPER_EDGES, days, side="left")
    codes = np.where(np.isfinite(days) & (days >= 0), codes, -1)
    df["maturity_bucket"] = pd.Categorical.from_codes(codes, categories=MATURITY_LABELS, ordered=True)

    # --- log-moneyness and ITM/OTM (depends on option type) ----------------------------
    K, S = df["strike_pr"].to_numpy(float), df["spot_close"].to_numpy(float)
    valid = (K > 0) & (S > 0)
    lm = np.where(valid, np.log(np.where(valid, K, 1.0) / np.where(valid, S, 1.0)), np.nan)
    df["log_moneyness"] = lm
    is_ce, is_pe = (typ == "CE").to_numpy(), (typ == "PE").to_numpy()
    df["is_otm"] = (is_ce & (lm > 0)) | (is_pe & (lm < 0))
    df["is_itm"] = (is_ce & (lm < 0)) | (is_pe & (lm > 0))

    df["expiry_type"] = pd.Categorical(classify_expiry_types(expiry))

    # --- reporting ------------------------------------------------------------------
    report["n_other_option_type"] = int((~(is_ce | is_pe)).sum())
    report["status_counts"] = status.value_counts().to_dict()
    ok = (status == STATUS_SUCCESS).to_numpy() & np.isfinite(df["IV"].to_numpy()) & (df["IV"].to_numpy() > 0)
    report["n_success"] = int(ok.sum())
    report["trading_dates"] = pd.DatetimeIndex(date.dropna().unique()).sort_values()
    report["n_trading_dates"] = int(len(report["trading_dates"]))
    iv_ok = df["IV"].to_numpy()[ok]
    report["iv_median_success"] = float(np.median(iv_ok)) if len(iv_ok) else np.nan
    report["iv_looks_like_percent"] = bool(len(iv_ok) and np.median(iv_ok) > 3)

    # IV_pct unit verification (IV_pct should equal 100*IV)
    if "IV_pct" in raw.columns:
        pct = num("IV_pct").to_numpy(float)
        both = np.isfinite(pct) & np.isfinite(df["IV"].to_numpy())
        if both.any():
            d = np.abs(pct[both] - 100 * df["IV"].to_numpy()[both])
            report["iv_pct_check"] = {"n_compared": int(both.sum()), "max_abs_diff": float(d.max()),
                                      "consistent": bool(d.max() < 1e-3)}
        else:
            report["iv_pct_check"] = {"n_compared": 0, "max_abs_diff": np.nan, "consistent": None}
    else:
        report["iv_pct_check"] = None
    return df, report


# ----------------------------------------------------------------------------
# 3. Filtering
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterConfig:
    """All user-selectable filters. Immutable and hashable-by-key for caching."""
    start: Optional[str] = None                 # 'YYYY-MM-DD'; None = no lower bound
    end: Optional[str] = None
    maturity_buckets: Optional[tuple] = None    # None = all maturities pooled
    custom_days: Optional[tuple] = None         # (min_days, max_days); overrides buckets
    option_types: tuple = ("CE", "PE")
    moneyness_mode: str = "all"                 # all | near_atm | otm | itm | custom
    near_atm_width: float = 0.02                # |ln(K/S)| <= width
    lm_range: Optional[tuple] = None            # used when moneyness_mode == 'custom'
    expiry_type: str = "all"                    # all | weekly | monthly
    min_contracts: float = 0.0
    min_open_int: float = 0.0
    drop_zero_volume: bool = False
    iv_min: Optional[float] = None              # decimals (0.05 = 5%)
    iv_max: Optional[float] = None
    success_only: bool = True

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def apply_filters(df: pd.DataFrame, cfg: FilterConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply filters sequentially with boolean masks (one copy at the end).
    Returns (filtered_frame, log) where log lists the rows remaining after every step.
    """
    mask = np.ones(len(df), dtype=bool)
    steps = [("Input observations", len(df))]

    def step(label, cond):
        nonlocal mask
        mask &= np.asarray(cond, dtype=bool)
        steps.append((label, int(mask.sum())))

    d = df["Date"]
    if cfg.start is not None or cfg.end is not None:
        cond = d.notna().to_numpy().copy()                                   # NaT dates cannot be placed in a window
        if cfg.start is not None:
            cond &= (d >= pd.Timestamp(cfg.start)).to_numpy()
        if cfg.end is not None:
            cond &= (d <= pd.Timestamp(cfg.end)).to_numpy()
        step("Date range", cond)

    if cfg.expiry_type != "all":
        step(f"Expiry type = {cfg.expiry_type}", (df["expiry_type"] == cfg.expiry_type).to_numpy())

    if cfg.custom_days is not None:
        lo, hi = cfg.custom_days
        dte = df["days_to_expiry"].to_numpy()
        step(f"Custom maturity {lo:g}-{hi:g} days", (dte >= lo) & (dte <= hi))
    elif cfg.maturity_buckets is not None:
        step("Maturity bucket(s): " + ", ".join(cfg.maturity_buckets),
             df["maturity_bucket"].isin(list(cfg.maturity_buckets)).to_numpy())

    if set(cfg.option_types) != {"CE", "PE"}:
        step("Option type: " + "/".join(cfg.option_types), df["option_typ"].isin(list(cfg.option_types)).to_numpy())
    else:
        step("Option type: CE and PE", df["option_typ"].isin(["CE", "PE"]).to_numpy())

    lm = df["log_moneyness"].to_numpy()
    if cfg.moneyness_mode == "near_atm":
        step(f"Near ATM |ln(K/S)| <= {cfg.near_atm_width:g}", np.abs(lm) <= cfg.near_atm_width)
    elif cfg.moneyness_mode == "otm":
        step("OTM only", df["is_otm"].to_numpy())
    elif cfg.moneyness_mode == "itm":
        step("ITM only", df["is_itm"].to_numpy())
    elif cfg.moneyness_mode == "custom" and cfg.lm_range is not None:
        step(f"Log-moneyness in [{cfg.lm_range[0]:g}, {cfg.lm_range[1]:g}]",
             (lm >= cfg.lm_range[0]) & (lm <= cfg.lm_range[1]))

    if cfg.min_contracts > 0:
        step(f"Contracts >= {cfg.min_contracts:g}", (df["contracts"] >= cfg.min_contracts).to_numpy())
    if cfg.min_open_int > 0:
        step(f"Open interest >= {cfg.min_open_int:g}", (df["open_int"] >= cfg.min_open_int).to_numpy())
    if cfg.drop_zero_volume:
        step("Non-zero volume", (df["contracts"] > 0).to_numpy())

    iv = df["IV"].to_numpy()
    if cfg.success_only:
        step("IV_status == success and IV finite", ((df["IV_status"] == STATUS_SUCCESS).to_numpy()
                                                    & np.isfinite(iv) & (iv > 0)))
    if cfg.iv_min is not None or cfg.iv_max is not None:
        lo = -np.inf if cfg.iv_min is None else cfg.iv_min
        hi = np.inf if cfg.iv_max is None else cfg.iv_max
        step(f"IV bounds [{lo:g}, {hi:g}] (rows without IV unaffected)", np.isnan(iv) | ((iv >= lo) & (iv <= hi)))

    log = pd.DataFrame(steps, columns=["step", "remaining"])
    log["removed"] = -log["remaining"].diff().fillna(0).astype(int)
    return df.loc[mask], log


def trading_days_in_range(report: dict, start, end) -> int:
    """Number of distinct trading dates (any status) in [start, end]."""
    td = report["trading_dates"]
    lo = td.searchsorted(pd.Timestamp(start), side="left")
    hi = td.searchsorted(pd.Timestamp(end), side="right")
    return int(hi - lo)


# ----------------------------------------------------------------------------
# 4. Date-balanced historical profile (Figures 1 and 2)
# ----------------------------------------------------------------------------
_AGGS = ("median", "mean")


def _check_agg(a):
    if a not in _AGGS:
        raise ValueError(f"Aggregation must be one of {_AGGS}, got {a!r}")


def date_balanced_profile(df, lo, hi, n_bins, group_cols=(), daily_agg="median",
                          across_agg="median", min_obs_per_cell=1):
    """
    Step 1: per (trading date, log-moneyness bin[, group]) aggregate IV (daily_agg).
    Step 2: across trading dates, take across_agg and the 25th/75th percentiles
            of those daily values. Every date therefore has equal weight
            regardless of how many contracts traded on it.
    Returns (profile, daily_table).
    """
    _check_agg(daily_agg); _check_agg(across_agg)
    group_cols = list(group_cols)
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    lm = df["log_moneyness"].to_numpy()
    ok = np.isfinite(lm) & (lm >= lo) & (lm <= hi) & df["Date"].notna().to_numpy() & np.isfinite(df["IV"].to_numpy())
    sub = df.loc[ok, ["Date", "IV", *group_cols]]
    cols = [*group_cols, "bin", "log_moneyness", "bin_left", "bin_right", "iv_central", "q25", "q75",
            "n_dates", "n_obs", "iv_central_pct", "q25_pct", "q75_pct"]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["Date", "bin", *group_cols, "iv", "n_obs"])

    b = np.minimum(np.searchsorted(edges, lm[ok], side="right") - 1, n_bins - 1)
    sub = sub.assign(bin=b)
    daily = (sub.groupby(["Date", "bin", *group_cols], observed=True)["IV"]
                .agg(iv=daily_agg, n_obs="size").reset_index())
    daily = daily[daily["n_obs"] >= min_obs_per_cell]

    keys = [*group_cols, "bin"]
    gi = daily.groupby(keys, observed=True)["iv"]
    prof = pd.concat({"iv_central": gi.agg(across_agg), "q25": gi.quantile(0.25),
                      "q75": gi.quantile(0.75), "n_dates": gi.size()}, axis=1)
    prof["n_obs"] = daily.groupby(keys, observed=True)["n_obs"].sum()
    prof = prof.reset_index()
    prof["log_moneyness"] = centres[prof["bin"].to_numpy()]
    prof["bin_left"] = edges[prof["bin"].to_numpy()]
    prof["bin_right"] = edges[prof["bin"].to_numpy() + 1]
    for c in ("iv_central", "q25", "q75"):
        prof[c + "_pct"] = prof[c] * 100
    return prof[cols], daily


def bootstrap_profile(daily, group_cols, n_bins, across_agg="median", n_boot=300, ci=95.0, seed=12345):
    """
    Date-level bootstrap CI for the across-date central value. The resampling
    unit is the TRADING DATE (rows of the date x bin matrix), with replacement.
    This is a descriptive uncertainty band; it ignores serial correlation
    between neighbouring dates (see README).
    """
    group_cols = list(group_cols)
    rng = np.random.default_rng(seed)
    func = np.nanmedian if across_agg == "median" else np.nanmean
    alpha = (100.0 - ci) / 2.0
    rows = []
    iterator = daily.groupby(group_cols, observed=True) if group_cols else [((), daily)]
    for key, sub in iterator:
        key = key if isinstance(key, tuple) else (key,)
        mat = (sub.pivot(index="Date", columns="bin", values="iv")
                  .reindex(columns=range(n_bins)).to_numpy(float))
        n = mat.shape[0]
        if n < 2:
            continue
        boots = np.full((n_boot, n_bins), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(n_boot):
                boots[i] = func(mat[rng.integers(0, n, n)], axis=0)
            lo, hi = np.nanpercentile(boots, [alpha, 100 - alpha], axis=0)
        for j in range(n_bins):
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                rows.append((*[str(k) for k in key], j, lo[j], hi[j]))
    return pd.DataFrame(rows, columns=[*group_cols, "bin", "ci_low", "ci_high"])


def historical_profile(df, lm_range=(-0.20, 0.20), n_bins=40, group_cols=(), daily_agg="median",
                       across_agg="median", min_obs_per_cell=1, bootstrap=False, n_boot=300,
                       ci=95.0, seed=12345):
    """Date-balanced smile/skew profile, optionally with bootstrap CI columns."""
    group_cols = list(group_cols)
    prof, daily = date_balanced_profile(df, lm_range[0], lm_range[1], n_bins, group_cols,
                                        daily_agg, across_agg, min_obs_per_cell)
    for c in group_cols:                                   # plain strings; row order keeps category order
        prof[c] = prof[c].astype(str)
    if bootstrap and len(prof):
        ci_df = bootstrap_profile(daily, group_cols, n_bins, across_agg, n_boot, ci, seed)
        prof = prof.merge(ci_df, on=[*group_cols, "bin"], how="left")
        prof["ci_low_pct"], prof["ci_high_pct"] = prof["ci_low"] * 100, prof["ci_high"] * 100
    return prof


# ----------------------------------------------------------------------------
# 5. Single-date smile (Figure 3)
# ----------------------------------------------------------------------------
def available_dates(df) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(df["Date"].dropna().unique()).sort_values()


def nearest_dates(df, date, k=5) -> list:
    d = available_dates(df)
    if len(d) == 0:
        return []
    order = np.argsort(np.abs((d - pd.Timestamp(date)).to_numpy().astype("int64")))
    return list(d[order[:k]].sort_values())


def available_expiries(df, date) -> pd.DataFrame:
    """Expiries with observations on a given trading date."""
    sub = df[df["Date"] == pd.Timestamp(date)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["expiry_dt", "days_to_expiry", "n_obs", "n_CE", "n_PE"])
    g = sub.groupby("expiry_dt", observed=True)
    out = pd.DataFrame({"days_to_expiry": g["days_to_expiry"].first(), "n_obs": g.size(),
                        "n_CE": g["option_typ"].apply(lambda s: int((s == "CE").sum())),
                        "n_PE": g["option_typ"].apply(lambda s: int((s == "PE").sum()))})
    return out.reset_index().sort_values("expiry_dt")


def daily_smile(df, date, expiry) -> pd.DataFrame:
    """All strikes for exactly this trading date and expiry (no substitution)."""
    m = (df["Date"] == pd.Timestamp(date)) & (df["expiry_dt"] == pd.Timestamp(expiry))
    cols = [c for c in ["Date", "expiry_dt", "strike_pr", "spot_close", "option_typ", "log_moneyness",
                        "IV", "close", "contracts", "open_int", "days_to_expiry", "is_otm"] if c in df.columns]
    out = df.loc[m, cols].sort_values(["option_typ", "strike_pr"]).copy()
    out["IV_pct"] = out["IV"] * 100
    return out


# ----------------------------------------------------------------------------
# 6. ATM term structure, ATM IV history and skew (Figures 4-6)
# ----------------------------------------------------------------------------
def _atm(df, atm_width):
    return df[np.abs(df["log_moneyness"].to_numpy()) <= atm_width]


def term_structure(df, atm_width=0.02, mode="daily", date=None, group_cols=(), dte_bin_width=7,
                   daily_agg="median", across_agg="median") -> pd.DataFrame:
    """
    ATM IV versus days to expiry.
    mode='daily'      : one trading date; one point per expiry.
    mode='historical' : date-balanced. (date, expiry) cells -> (date, DTE-bin) -> median across dates.
    """
    _check_agg(daily_agg); _check_agg(across_agg)
    group_cols = list(group_cols)
    atm = _atm(df, atm_width)
    if mode == "daily":
        if date is None:
            raise ValueError("date is required for mode='daily'")
        atm = atm[atm["Date"] == pd.Timestamp(date)]
        if len(atm) == 0:
            return pd.DataFrame(columns=[*group_cols, "expiry_dt", "days_to_expiry", "iv", "n_obs", "iv_pct"])
        out = (atm.groupby([*group_cols, "expiry_dt", "days_to_expiry"], observed=True)["IV"]
                  .agg(iv=daily_agg, n_obs="size").reset_index())
        out["iv_pct"] = out["iv"] * 100
        for c in group_cols:
            out[c] = out[c].astype(str)
        return out.sort_values([*group_cols, "days_to_expiry"]).reset_index(drop=True)

    if mode != "historical":
        raise ValueError("mode must be 'daily' or 'historical'")
    cell = (atm.groupby(["Date", "expiry_dt", "days_to_expiry", *group_cols], observed=True)["IV"]
               .agg(iv=daily_agg).reset_index())
    w = float(dte_bin_width)
    k = np.ceil(cell["days_to_expiry"].to_numpy() / w)
    cell["dte_center"] = k * w - (w - 1) / 2.0
    db = (cell.groupby(["Date", "dte_center", *group_cols], observed=True)["iv"]
              .agg(daily_agg).rename("iv").reset_index())
    keys = [*group_cols, "dte_center"]
    gi = db.groupby(keys, observed=True)["iv"]
    out = pd.concat({"iv_central": gi.agg(across_agg), "q25": gi.quantile(0.25),
                     "q75": gi.quantile(0.75), "n_dates": gi.size()}, axis=1).reset_index()
    for c in ("iv_central", "q25", "q75"):
        out[c + "_pct"] = out[c] * 100
    for c in group_cols:
        out[c] = out[c].astype(str)
    return out


def add_rolling(df, col, window=20, group_cols=(), min_periods=None) -> pd.DataFrame:
    """Rolling median over the last `window` observed trading dates (per group)."""
    group_cols = list(group_cols)
    mp = max(1, window // 2) if min_periods is None else min_periods
    df = df.sort_values([*group_cols, "Date"]).copy()
    name = f"{col}_roll{window}"
    if group_cols:
        df[name] = df.groupby(group_cols, observed=True)[col].transform(lambda s: s.rolling(window, min_periods=mp).median())
    else:
        df[name] = df[col].rolling(window, min_periods=mp).median()
    return df


def daily_atm_series(df, atm_width=0.02, group_cols=(), agg="median", window=20) -> pd.DataFrame:
    """Daily ATM IV (one value per date[, group]) plus a rolling median."""
    _check_agg(agg)
    group_cols = list(group_cols)
    atm = _atm(df, atm_width)
    out = (atm.groupby(["Date", *group_cols], observed=True)["IV"].agg(iv=agg, n_obs="size").reset_index())
    for c in group_cols:
        out[c] = out[c].astype(str)
    out["iv_pct"] = out["iv"] * 100
    out = add_rolling(out, "iv_pct", window, group_cols)
    return out


def daily_skew(df, wing_lo=0.02, wing_hi=0.10, group_cols=(), min_obs=1, agg="median", window=20) -> pd.DataFrame:
    """
    Daily wing skew = IV(OTM put wing) - IV(OTM call wing), where
        put wing  : PE with  -wing_hi <= ln(K/S) <= -wing_lo
        call wing : CE with   wing_lo <= ln(K/S) <=  wing_hi
    This is a moneyness-band skew, NOT a delta-based risk reversal.
    """
    _check_agg(agg)
    group_cols = list(group_cols)
    lm, typ = df["log_moneyness"].to_numpy(), df["option_typ"]
    put = (typ == "PE").to_numpy() & (lm <= -wing_lo) & (lm >= -wing_hi)
    call = (typ == "CE").to_numpy() & (lm >= wing_lo) & (lm <= wing_hi)
    sub = df.loc[put | call, ["Date", "option_typ", "IV", *group_cols]]
    cols = [*group_cols, "Date", "put_wing_iv", "call_wing_iv", "skew", "n_put", "n_call", "skew_pct"]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    g = sub.groupby(["Date", *group_cols, "option_typ"], observed=True)["IV"].agg(iv=agg, n="size")
    iv = g["iv"].unstack("option_typ").reindex(columns=["CE", "PE"])
    n = g["n"].unstack("option_typ").reindex(columns=["CE", "PE"]).fillna(0)
    out = pd.DataFrame({"put_wing_iv": iv["PE"], "call_wing_iv": iv["CE"],
                        "n_put": n["PE"].astype(int), "n_call": n["CE"].astype(int)})
    out["skew"] = out["put_wing_iv"] - out["call_wing_iv"]
    out.loc[(out["n_put"] < min_obs) | (out["n_call"] < min_obs), "skew"] = np.nan
    out = out.reset_index()
    out = out.dropna(subset=["skew"])
    for c in group_cols:
        out[c] = out[c].astype(str)
    out["skew_pct"] = out["skew"] * 100
    out = add_rolling(out, "skew_pct", window, group_cols)
    return out[[*cols, f"skew_pct_roll{window}"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 7. Summary statistics and data-quality helpers
# ----------------------------------------------------------------------------
def summarize_iv(df, by: Optional[str] = None) -> pd.DataFrame:
    """Descriptive IV statistics (in %) overall or by a column such as option_typ / maturity_bucket."""
    def one(sub):
        iv = sub["IV"].dropna() * 100
        q = iv.quantile([0.05, 0.25, 0.75, 0.95]) if len(iv) else pd.Series([np.nan] * 4, index=[.05, .25, .75, .95])
        return {"n_obs": int(len(iv)), "mean_pct": iv.mean(), "median_pct": iv.median(), "std_pct": iv.std(),
                "p05_pct": q[0.05], "p25_pct": q[0.25], "p75_pct": q[0.75], "p95_pct": q[0.95],
                "min_pct": iv.min(), "max_pct": iv.max(),
                "n_trading_dates": int(sub.loc[iv.index, "Date"].nunique()),
                "n_expiries": int(sub.loc[iv.index, "expiry_dt"].nunique())}
    if by is None:
        return pd.DataFrame([one(df)], index=["All"])
    return pd.DataFrame.from_dict({str(k): one(s) for k, s in df.groupby(by, observed=True)}, orient="index")


def quality_counts(df) -> dict:
    """Counts by IV_status plus distinct dates/expiries (use a frame filtered with success_only=False)."""
    vc = df["IV_status"].value_counts()
    known = [STATUS_SUCCESS, STATUS_INVALID_PRICE, STATUS_SOLVER_FAILURE, STATUS_MISSING_INPUT]
    out = {"n_obs": int(len(df))}
    for s in known:
        out[s] = int(vc.get(s, 0))
    out["other_status"] = int(len(df) - sum(out[s] for s in known))
    out["n_trading_dates"] = int(df["Date"].nunique())
    out["n_expiries"] = int(df["expiry_dt"].nunique())
    return out


def status_by_year(df) -> pd.DataFrame:
    year = df["Date"].dt.year.astype("Int64").astype(str).replace("<NA>", "invalid date")
    return pd.crosstab(year, df["IV_status"].astype(str), margins=True, margins_name="Total")


def iv_quantiles(df, qs=(0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)) -> pd.DataFrame:
    iv = df["IV"].dropna() * 100
    return pd.DataFrame({"quantile": qs, "iv_pct": iv.quantile(list(qs)).to_numpy()}) if len(iv) else pd.DataFrame(columns=["quantile", "iv_pct"])


def extreme_observations(df, n=25, which="high") -> pd.DataFrame:
    cols = [c for c in ["Date", "expiry_dt", "strike_pr", "spot_close", "option_typ", "log_moneyness", "days_to_expiry",
                        "close", "contracts", "open_int", "IV", "IV_status"] if c in df.columns]
    d = df.dropna(subset=["IV"])
    out = d.nlargest(n, "IV") if which == "high" else d.nsmallest(n, "IV")
    return out[cols].assign(IV_pct=lambda x: x["IV"] * 100)


def failed_observations(df, status: Optional[str] = None) -> pd.DataFrame:
    """Rows whose IV_status is not 'success' (optionally one specific status)."""
    m = df["IV_status"] != STATUS_SUCCESS
    if status:
        m = df["IV_status"] == status
    cols = [c for c in ["Date", "expiry_dt", "strike_pr", "spot_close", "option_typ", "log_moneyness", "days_to_expiry",
                        "maturity_bucket", "close", "contracts", "open_int", "IV_status"] if c in df.columns]
    return df.loc[m, cols]
