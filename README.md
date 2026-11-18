# NIFTY Implied-Volatility Research Dashboard

| File | Role |
|---|---|
| `iv_analysis.py` | All data processing and calculations. No Streamlit/Plotly dependency, so it can be reused in a thesis notebook. |
| `iv_dashboard.py` | Streamlit UI and Plotly figures only. Calls `iv_analysis`. |
| `test_iv_analysis.py` | Tests on synthetic data (mixed dates, missing IV, empty dates, buckets, CE/PE, date-balancing, bootstrap). |
| `requirements.txt` | Dependencies (`kaleido` is optional, for PNG export). |

## Install and run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run iv_dashboard.py --server.maxUploadSize 2048
```
Upload the CSV in the sidebar, or paste its path into "local file path" (more convenient for a ~1.3M-row file).
The file is only read, never modified. Run the tests with `python test_iv_analysis.py`.

## Using `iv_analysis.py` outside the dashboard
```python
import iv_analysis as iva
df, report = iva.prepare_data(iva.load_csv("nifty_IV_calculated.csv"))
sub, log = iva.apply_filters(df, iva.FilterConfig(maturity_buckets=("16-30 days",), min_contracts=10))
smile = iva.historical_profile(sub, lm_range=(-0.15, 0.15), n_bins=30, group_cols=["option_typ"], bootstrap=True)
```

## Methodology per figure
* **Common.** Only `IV_status == "success"` with finite positive `IV` enters Figs 1-6. `log_moneyness = ln(strike_pr / spot_close)`.
  Days to expiry = `round(T*365)`, falling back to `expiry_dt - Date`. IV is plotted as `IV x 100`; `IV_pct` is only checked
  against it (Data-quality tab) and never plotted. ITM/OTM is option-type aware (CE OTM if K>S, PE OTM if K<S).
* **Dates.** `parse_mixed_dates` parses unique strings with explicit `%Y-%m-%d` and `%d-%b-%Y` (case-insensitive) first, then
  `format="mixed"` (with a per-element fallback for old pandas). Failures become `NaT`, are counted and shown; rows are
  never silently dropped (a date-range filter necessarily excludes rows whose date is `NaT`, and the filter log shows it).
* **Fig 1 / 2 (date-balanced smile).** Per trading date and fixed log-moneyness bin: median IV (or mean). Across dates:
  median (or mean) and 25th/75th percentiles of those daily values. Dates, not contracts, carry equal weight; the lower panel
  shows contributing dates per bin. Optional bootstrap resamples trading dates with replacement (seeded) and reports the
  central 95% of the resampled central value.
* **Fig 3.** Exactly the date and expiry you pick; an unavailable date triggers a warning with nearest dates, no substitution.
* **Fig 4.** ATM = `|ln(K/S)| <= width` (default 0.02). Daily: median ATM IV per expiry. Historical:
  (date, expiry) -> (date, DTE bin) -> median across dates. Sidebar maturity/moneyness filters are ignored here.
* **Fig 5.** Daily median ATM IV with a rolling median over the last N observed trading dates.
* **Fig 6.** `median IV(OTM puts, -b_hi <= ln(K/S) <= -b_lo) - median IV(OTM calls, b_lo <= ln(K/S) <= b_hi)`.
  A moneyness-band measure, not a delta-based risk reversal.

## Limitations to state in academic work
1. **Model dependence.** IVs come from spot-based Black-Scholes with a dividend yield and the 91-day T-bill rate. Index options
   are priced off the forward (futures); errors in `r` and `q` shift IV differently across strikes and maturities and can
   distort skew, especially for long maturities and for puts versus calls (put-call parity violations).
2. **Closing prices.** `close` is the last traded price, not a mid-quote, and can be stale or non-synchronous with `spot_close`
   (especially far OTM / illiquid contracts and near expiry, where IV is hypersensitive to price).
3. **Selection (survivorship) of solved IVs.** Invalid-price and solver-failure rows are excluded from the figures; they are
   concentrated in deep ITM, very short-dated and illiquid contracts, so the smile wings are conditional on successful inversion.
   Use the Data-quality tab to see how this varies by year/maturity.
4. **ATM definition.** ATM is defined against spot (K = S), not the forward `S e^{(r-q)T}`; the difference grows with maturity.
5. **Weekly/monthly classification** is inferred from the data: the last expiry in each calendar month present in the file is
   "monthly", earlier ones "weekly". It can misclassify at the edges of the sample or if expiries are missing from the file.
   Verify before relying on weekly-vs-monthly comparisons.
6. **Pooling.** Pooling maturities or moneyness regions mixes different objects; the dashboard labels pooled views, but
   interpretation is yours. Regime changes (e.g. contract specifications, lot sizes, weekly-expiry introduction in 2019,
   shifts in the expiry weekday) are not modelled.
7. **Inference.** Bands (IQR, bootstrap CI) are descriptive. Dates are serially correlated and expiry cycles overlap, so the
   date-level bootstrap understates uncertainty; no significance claims are made.
8. **Extreme IVs** are kept by default. Use the IV-bounds control for sensitivity checks.
