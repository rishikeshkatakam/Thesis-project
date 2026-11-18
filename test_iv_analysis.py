"""
Tests for iv_analysis.py. Run with:  python test_iv_analysis.py   (or: pytest)
Uses small synthetic data only; your real CSV is never touched.
"""
import numpy as np
import pandas as pd

import iv_analysis as iva


def make_raw(rows):
    """rows: list of dicts; fills the full 31-column schema."""
    base = {c: np.nan for c in iva.SCHEMA_COLUMNS}
    base.update(instrument="OPTIDX", symbol="NIFTY", contracts=10.0, open_int=100.0)
    return pd.DataFrame([{**base, **r} for r in rows])[iva.SCHEMA_COLUMNS]


def row(date, expiry, K, typ, S=10000.0, iv=0.2, status="success", contracts=10.0, oi=100.0):
    d, e = pd.to_datetime(date, errors="coerce"), pd.to_datetime(expiry, errors="coerce")
    T = (e - d).days / 365 if pd.notna(d) else 30 / 365
    return dict(Date=date, expiry_dt=expiry, strike_pr=K, option_typ=typ, spot_close=S, T=T,
                IV=iv, IV_pct=iv * 100 if np.isfinite(iv) else np.nan, IV_status=status,
                contracts=contracts, open_int=oi)


# 1 ------------------------------------------------------------------------------------
def test_mixed_dates():
    s = pd.Series(["28-Apr-2016", "2025-05-29", "01-APR-2016", "15-apr-2016", "", None, "garbage", "2016-13-45"])
    out, rep = iva.parse_mixed_dates(s)
    assert out.iloc[0] == pd.Timestamp("2016-04-28")
    assert out.iloc[1] == pd.Timestamp("2025-05-29")
    assert out.iloc[2] == pd.Timestamp("2016-04-01")
    assert out.iloc[3] == pd.Timestamp("2016-04-15")
    assert out.iloc[4:].isna().all()
    assert len(out) == len(s)                                  # nothing dropped
    assert rep["n_valid"] == 4 and rep["n_missing"] == 2 and rep["n_unparseable"] == 2
    assert rep["min"] == pd.Timestamp("2016-04-01") and rep["max"] == pd.Timestamp("2025-05-29")


# 2 ------------------------------------------------------------------------------------
def test_schema_and_no_mutation():
    raw = make_raw([row("2016-04-01", "28-Apr-2016", 10000, "CE")])
    try:
        iva.prepare_data(raw.drop(columns=["IV"]))
        assert False, "should raise"
    except iva.SchemaError:
        pass
    before = raw.copy(deep=True)
    df, rep = iva.prepare_data(raw)
    assert raw.equals(before)                                   # input untouched
    assert len(df) == len(raw)


def test_invalid_dates_and_missing_iv_kept():
    raw = make_raw([row("2016-04-01", "28-Apr-2016", 10000, "CE"),
                    row("bad-date", "28-Apr-2016", 10000, "CE"),
                    row("2016-04-02", "28-Apr-2016", 10000, "PE", iv=np.nan, status="invalid_price")])
    df, rep = iva.prepare_data(raw)
    assert len(df) == 3 and rep["trade_date"]["n_unparseable"] == 1
    sub, log = iva.apply_filters(df, iva.FilterConfig())          # success only
    assert len(sub) == 0 or sub["IV"].notna().all()
    assert len(sub) == 2                                          # NaT-date row is retained when no date window is set
    # date window excludes NaT row and the log says so
    sub2, log2 = iva.apply_filters(df, iva.FilterConfig(start="2016-01-01", end="2016-12-31"))
    assert len(sub2) == 1 and "Date range" in log2["step"].tolist()


# 3 ------------------------------------------------------------------------------------
def test_maturity_buckets_each():
    days = [1, 7, 8, 15, 16, 30, 31, 60, 61, 90, 91, 180, 181, 365, 366, 1000]
    expected = ["0-7 days", "0-7 days", "8-15 days", "8-15 days", "16-30 days", "16-30 days",
                "31-60 days", "31-60 days", "61-90 days", "61-90 days", "91-180 days", "91-180 days",
                "181-365 days", "181-365 days", "Above 365 days", "Above 365 days"]
    start = pd.Timestamp("2016-01-04")
    rows = [row(start.strftime("%Y-%m-%d"), (start + pd.Timedelta(days=d)).strftime("%d-%b-%Y"), 10000, "CE") for d in days]
    df, _ = iva.prepare_data(make_raw(rows))
    assert df["maturity_bucket"].astype(str).tolist() == expected
    for label in iva.MATURITY_LABELS:                           # filtering by each bucket
        sub, _ = iva.apply_filters(df, iva.FilterConfig(maturity_buckets=(label,)))
        assert (sub["maturity_bucket"] == label).all() and len(sub) == 2
    sub, _ = iva.apply_filters(df, iva.FilterConfig(custom_days=(10, 40)))
    assert sorted(sub["days_to_expiry"]) == [15, 16, 30, 31]


# 4 ------------------------------------------------------------------------------------
def test_ce_pe_moneyness():
    rows = [row("2016-04-01", "28-Apr-2016", 11000, "CE"), row("2016-04-01", "28-Apr-2016", 9000, "CE"),
            row("2016-04-01", "28-Apr-2016", 11000, "PE"), row("2016-04-01", "28-Apr-2016", 9000, "PE")]
    df, _ = iva.prepare_data(make_raw(rows))
    assert df["is_otm"].tolist() == [True, False, False, True]    # CE OTM: K>S ; PE OTM: K<S
    assert df["is_itm"].tolist() == [False, True, True, False]
    otm, _ = iva.apply_filters(df, iva.FilterConfig(moneyness_mode="otm"))
    assert len(otm) == 2 and otm["is_otm"].all()
    ce, _ = iva.apply_filters(df, iva.FilterConfig(option_types=("CE",)))
    assert (ce["option_typ"] == "CE").all() and len(ce) == 2
    assert np.isclose(df["log_moneyness"].iloc[0], np.log(1.1))


# 5 ------------------------------------------------------------------------------------
def test_date_balanced_aggregation():
    rows = [row("2016-04-01", "28-Apr-2016", 10000, "CE", iv=0.20) for _ in range(100)]   # busy date
    rows += [row("2016-04-04", "28-Apr-2016", 10000, "CE", iv=0.40)]                      # quiet date
    df, _ = iva.prepare_data(make_raw(rows))
    prof = iva.historical_profile(df, lm_range=(-0.1, 0.1), n_bins=5)
    assert len(prof) == 1 and prof["n_dates"].iloc[0] == 2 and prof["n_obs"].iloc[0] == 101
    assert np.isclose(prof["iv_central"].iloc[0], 0.30)          # date-balanced
    assert np.isclose(df["IV"].median(), 0.20)                    # naive pooled median differs
    prof_mean = iva.historical_profile(df, lm_range=(-0.1, 0.1), n_bins=5, daily_agg="mean", across_agg="mean")
    assert np.isclose(prof_mean["iv_central"].iloc[0], 0.30)


def test_date_without_success_and_groups_and_bootstrap():
    rng = np.random.default_rng(0)
    rows = []
    for i, d in enumerate(pd.bdate_range("2016-04-01", periods=40)):
        ds = d.strftime("%Y-%m-%d")
        for K in range(9000, 11001, 250):
            for typ in ("CE", "PE"):
                rows.append(row(ds, "30-Jun-2016", K, typ, iv=0.2 + rng.normal(0, 0.01)))
    rows += [row("2016-06-01", "30-Jun-2016", 10000, "CE", iv=np.nan, status="solver_failure")]   # date with no success
    df, rep = iva.prepare_data(make_raw(rows))
    sub, log = iva.apply_filters(df, iva.FilterConfig())
    assert pd.Timestamp("2016-06-01") not in set(sub["Date"])
    assert rep["status_counts"]["solver_failure"] == 1
    prof = iva.historical_profile(sub, lm_range=(-0.15, 0.15), n_bins=10, group_cols=["option_typ"],
                                  bootstrap=True, n_boot=100)
    assert set(prof["option_typ"]) == {"CE", "PE"}
    assert (prof["n_dates"] <= 40).all()
    ok = prof.dropna(subset=["ci_low"])
    assert len(ok) > 0 and (ok["ci_low"] <= ok["iv_central"] + 1e-9).all() and (ok["ci_high"] >= ok["iv_central"] - 1e-9).all()
    again = iva.historical_profile(sub, lm_range=(-0.15, 0.15), n_bins=10, group_cols=["option_typ"],
                                   bootstrap=True, n_boot=100)
    assert np.allclose(prof["ci_low"].fillna(0), again["ci_low"].fillna(0))      # seeded -> reproducible
    # empty frame must not crash
    empty = iva.historical_profile(sub.iloc[0:0], bootstrap=True)
    assert len(empty) == 0


# 6 ------------------------------------------------------------------------------------
def test_multiple_expiries_daily_smile_term_structure():
    rows = []
    for exp in ("28-Apr-2016", "26-May-2016"):
        for K in (9800, 10000, 10200):
            for typ in ("CE", "PE"):
                rows.append(row("2016-04-01", exp, K, typ, iv=0.15 if exp == "28-Apr-2016" else 0.18))
    df, _ = iva.prepare_data(make_raw(rows))
    ex = iva.available_expiries(df, "2016-04-01")
    assert len(ex) == 2 and ex["n_obs"].tolist() == [6, 6]
    sm = iva.daily_smile(df, "2016-04-01", "2016-05-26")
    assert len(sm) == 6 and (sm["expiry_dt"] == pd.Timestamp("2016-05-26")).all()
    assert len(iva.available_expiries(df, "2020-01-01")) == 0                  # unavailable -> empty, no substitution
    assert len(iva.nearest_dates(df, "2020-01-01")) == 1
    ts = iva.term_structure(df, 0.02, "daily", "2016-04-01")
    assert len(ts) == 2 and np.allclose(sorted(ts["iv"]), [0.15, 0.18])
    th = iva.term_structure(df, 0.02, "historical", dte_bin_width=7)
    assert th["n_dates"].eq(1).all() and len(th) == 2


# 7 ------------------------------------------------------------------------------------
def test_skew_and_atm_series_and_summary():
    rows = []
    for d in pd.bdate_range("2016-04-01", periods=30):
        ds = d.strftime("%Y-%m-%d")
        rows += [row(ds, "28-Apr-2016" if d < pd.Timestamp("2016-04-28") else "26-May-2016", 9500, "PE", iv=0.25),
                 row(ds, "28-Apr-2016" if d < pd.Timestamp("2016-04-28") else "26-May-2016", 10500, "CE", iv=0.18),
                 row(ds, "28-Apr-2016" if d < pd.Timestamp("2016-04-28") else "26-May-2016", 10000, "CE", iv=0.20)]
    df, _ = iva.prepare_data(make_raw(rows))
    sk = iva.daily_skew(df, 0.02, 0.10, window=5)
    assert len(sk) == 30 and np.allclose(sk["skew"], 0.07)
    assert "skew_pct_roll5" in sk.columns
    at = iva.daily_atm_series(df, 0.02, window=5)
    assert len(at) == 30 and np.allclose(at["iv"], 0.20)
    s = iva.summarize_iv(df, "option_typ")
    assert set(s.index) == {"CE", "PE"} and s.loc["PE", "median_pct"] == 25.0
    assert iva.quality_counts(df)["n_obs"] == 90
    # CE-only frame -> skew cannot be formed (no put wing), must return empty, not crash
    ce_only, _ = iva.apply_filters(df, iva.FilterConfig(option_types=("CE",)))
    assert len(iva.daily_skew(ce_only)) == 0


def test_expiry_type_and_iv_pct_check():
    rows = [row("2019-03-01", e, 10000, "CE") for e in ("07-Mar-2019", "14-Mar-2019", "28-Mar-2019")]
    rows += [row("2016-04-01", "28-Apr-2016", 10000, "CE")]
    df, rep = iva.prepare_data(make_raw(rows))
    assert df["expiry_type"].astype(str).tolist() == ["weekly", "weekly", "monthly", "monthly"]
    assert rep["iv_pct_check"]["consistent"] is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
