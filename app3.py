# app.py
# Streamlit Tourism Forecasting Project (NO SIDEBAR, NO QQ-PLOT, NO NAIVE BASELINE)
# Requirements implemented:
# - Dataset: Italy monthly arrivals, use data up to 2019-12-31 (pre-COVID)
# - Stationarity: run ADF+KPSS, then FORCE seasonal differencing D=1 (s=12) and choose d (0..2),
#   then re-run ADF+KPSS + ACF/PACF on differenced series
# - SARIMA first: choose candidate p,q,P,Q from ACF/PACF (on differenced series), fit those SARIMA models,
#   then choose best using Ljung-Box + Durbin-Watson + AIC/BIC (prefer BIC by default)
# - Baseline: ONLY Seasonal Naive (no naive)
# - ETS after SARIMA
# - Diagnostics: residual time plot, residual ACF, Ljung-Box table, Durbin-Watson (NO QQ plot)
#
# Run:
#   streamlit run app.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH_DEFAULT = "arrivals_1989_2024.csv"
COUNTRY_DEFAULT = "Italy"
END_DATE_DEFAULT = "2019-12-31"

FREQ = "MS"              # monthly start
S = 12                   # seasonal period for monthly data
FORCED_D = 1             # force seasonal differencing as requested
MAX_D = 2                # search d in {0,1,2}
TEST_MONTHS_DEFAULT = 24

ACF_PACF_PQ_MAX_LAG_DEFAULT = 5   # look for non-seasonal spikes in lags 1..5
SEASONAL_MULTS_DEFAULT = (1, 2)   # check spikes at s and 2s (12 and 24)

ALPHA_DEFAULT = 0.05
LJUNG_LAGS_DEFAULT = (12, 24)     # residual autocorrelation check
SELECTION_PRIMARY_DEFAULT = "BIC" # "BIC" (recommended) or "AIC"

ETS_SEASONAL_DEFAULT = "mul"      # "mul" or "add"


# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="Tourism Forecasting – Italy (pre-COVID)", layout="wide")
st.title("Tourism Forecasting – Italy (Monthly Arrivals, pre-COVID)")
st.caption("Workflow: EDA → ADF/KPSS → FORCE D=1 choose d → ACF/PACF → SARIMA selection → Seasonal naive → ETS → Compare → Diagnostics")


# -----------------------------
# Utils: metrics
# -----------------------------
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


# -----------------------------
# Utils: plots
# -----------------------------
def plot_series(y: pd.Series, title: str, ylabel="Arrivals"):
    fig, ax = plt.subplots()
    ax.plot(y.index, y.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_train_test(train: pd.Series, test: pd.Series, title: str):
    fig, ax = plt.subplots()
    ax.plot(train.index, train.values, label="Train")
    ax.plot(test.index, test.values, label="Test")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Arrivals")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_forecast(train: pd.Series, test: pd.Series, forecast: pd.Series, title: str, label: str):
    fig, ax = plt.subplots()
    ax.plot(train.index, train.values, label="Train")
    ax.plot(test.index, test.values, label="Test")
    ax.plot(test.index, forecast.values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Arrivals")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_acf_pacf(series: pd.Series, lags: int = 48):
    s = series.dropna()
    if len(s) < 10:
        st.warning("Series too short to plot ACF/PACF.")
        return
    lags = int(min(lags, len(s) - 1))

    acf_vals = acf(s, nlags=lags, fft=True)
    fig, ax = plt.subplots()
    ax.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
    ax.set_title("ACF")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    st.pyplot(fig)

    pacf_vals = pacf(s, nlags=lags, method="ywm")
    fig, ax = plt.subplots()
    ax.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
    ax.set_title("PACF")
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    st.pyplot(fig)


# -----------------------------
# Stationarity tests
# -----------------------------
def adf_test(series: pd.Series, regression: str = "c"):
    res = adfuller(series.dropna().values, regression=regression, autolag="AIC")
    return {"stat": float(res[0]), "pvalue": float(res[1]), "lags": int(res[2]), "nobs": int(res[3])}

def kpss_test(series: pd.Series, regression: str = "c"):
    stat, pvalue, lags, crit = kpss(series.dropna().values, regression=regression, nlags="auto")
    return {"stat": float(stat), "pvalue": float(pvalue), "lags": int(lags), "crit": {k: float(v) for k, v in crit.items()}}

def is_stationary_by_tests(series: pd.Series, alpha: float = 0.05):
    # Stationary if: ADF p < alpha AND KPSS p > alpha (regression='c')
    a = adf_test(series, "c")["pvalue"]
    k = kpss_test(series, "c")["pvalue"]
    return (a < alpha) and (k > alpha), a, k

def apply_differencing(series: pd.Series, d: int, D: int, s: int) -> pd.Series:
    out = series.copy()
    for _ in range(d):
        out = out.diff()
    for _ in range(D):
        out = out.diff(s)
    return out.dropna()

def choose_d_with_forced_D(series: pd.Series, s: int, alpha: float, max_d: int, forced_D: int = 1):
    candidates = []
    for d in range(0, max_d + 1):
        t = apply_differencing(series, d=d, D=forced_D, s=s)
        ok, adf_p, kpss_p = is_stationary_by_tests(t, alpha=alpha)
        candidates.append({"d": d, "D": forced_D, "stationary": ok, "adf_p": adf_p, "kpss_p": kpss_p, "n": len(t)})

    stationary = [c for c in candidates if c["stationary"]]
    if stationary:
        stationary.sort(key=lambda x: x["d"])  # minimal d
        return stationary[0], candidates

    # closest heuristic
    candidates.sort(key=lambda x: (x["adf_p"], -x["kpss_p"], x["d"]))
    return candidates[0], candidates


# -----------------------------
# ACF/PACF -> candidate order proposal
# -----------------------------
def sig_threshold(n: int) -> float:
    return 1.96 / np.sqrt(max(n, 1))

def propose_sarima_grid_from_acf_pacf(series_diff: pd.Series, s: int, max_nonseasonal: int, seasonal_mults):
    sd = series_diff.dropna()
    if len(sd) < 30:
        raise ValueError("Differenced series too short for reliable ACF/PACF-based order proposal.")

    nlags_needed = max(max_nonseasonal, max(seasonal_mults) * s)
    nlags = min(nlags_needed, len(sd) - 1)

    acf_vals = acf(sd, nlags=nlags, fft=True)
    pacf_vals = pacf(sd, nlags=nlags, method="ywm")
    thr = sig_threshold(len(sd))

    # non-seasonal candidates
    p_list = sorted({0} | {lag for lag in range(1, min(max_nonseasonal, nlags) + 1) if abs(pacf_vals[lag]) > thr})
    q_list = sorted({0} | {lag for lag in range(1, min(max_nonseasonal, nlags) + 1) if abs(acf_vals[lag]) > thr})

    # seasonal candidates: spikes at s,2s,... => propose {0,1} (typical coursework)
    P_set, Q_set = {0}, {0}
    for m in seasonal_mults:
        L = m * s
        if L <= nlags:
            if abs(pacf_vals[L]) > thr:
                P_set.add(1)
            if abs(acf_vals[L]) > thr:
                Q_set.add(1)
    P_list = sorted(P_set)
    Q_list = sorted(Q_set)

    return {
        "p_list": p_list, "q_list": q_list,
        "P_list": P_list, "Q_list": Q_list,
        "thr": thr, "nlags": nlags,
        "acf_vals": acf_vals, "pacf_vals": pacf_vals
    }


# -----------------------------
# Models
# -----------------------------
def seasonal_naive_forecast(train: pd.Series, steps: int, s: int = 12, freq: str = "MS") -> pd.Series:
    last_season = train.iloc[-s:].values
    vals = np.resize(last_season, steps)
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq=freq)
    return pd.Series(vals, index=idx)

def fit_ets(train: pd.Series, seasonal_period: int, seasonal: str):
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_period,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)

def fit_sarima(train: pd.Series, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

def sarima_diag(res, lb_lags, alpha):
    resid = pd.Series(res.resid).dropna()
    dw = float(durbin_watson(resid.values)) if len(resid) > 5 else np.nan

    lb_min_p = np.nan
    lb_pass = False
    if len(resid) > max(lb_lags) + 5:
        lb_df = acorr_ljungbox(resid, lags=list(lb_lags), return_df=True)
        lb_min_p = float(lb_df["lb_pvalue"].min())
        lb_pass = bool(lb_min_p > alpha)

    return {
        "resid": resid,
        "dw": dw,
        "dw_dist": float(abs(dw - 2.0)) if np.isfinite(dw) else np.inf,
        "lb_min_p": lb_min_p,
        "lb_pass": lb_pass,
    }


# -----------------------------
# Inputs in main page (no sidebar)
# -----------------------------
with st.expander("Settings", expanded=False):
    colA, colB, colC = st.columns(3)
    csv_path = colA.text_input("CSV path", value=CSV_PATH_DEFAULT)
    country = colB.text_input("Country", value=COUNTRY_DEFAULT)
    end_date = colC.text_input("Use data up to (YYYY-MM-DD)", value=END_DATE_DEFAULT)

    col1, col2, col3 = st.columns(3)
    test_months = col1.number_input("Test months", min_value=6, max_value=60, value=TEST_MONTHS_DEFAULT, step=1)
    alpha = col2.selectbox("Alpha (ADF/KPSS & Ljung-Box pass)", [0.01, 0.05, 0.10], index=[0.01,0.05,0.10].index(ALPHA_DEFAULT))
    selection_primary = col3.selectbox("Primary SARIMA selection criterion", ["BIC", "AIC"], index=["BIC","AIC"].index(SELECTION_PRIMARY_DEFAULT))

    col4, col5, col6 = st.columns(3)
    pq_max_lag = col4.number_input("Max non-seasonal lag for p/q (ACF/PACF)", min_value=1, max_value=10, value=ACF_PACF_PQ_MAX_LAG_DEFAULT, step=1)
    ljung_lag1 = col5.number_input("Ljung-Box lag 1", min_value=6, max_value=48, value=LJUNG_LAGS_DEFAULT[0], step=1)
    ljung_lag2 = col6.number_input("Ljung-Box lag 2", min_value=6, max_value=72, value=LJUNG_LAGS_DEFAULT[1], step=1)

    col7, col8 = st.columns(2)
    ets_seasonal = col7.selectbox("ETS seasonality", ["mul", "add"], index=["mul","add"].index(ETS_SEASONAL_DEFAULT))
    acf_pacf_plot_lags = col8.number_input("ACF/PACF plot lags", min_value=12, max_value=120, value=48, step=1)

lb_lags = (int(ljung_lag1), int(ljung_lag2))


# -----------------------------
# Load and prepare data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare(csv_path: str, country: str, end_date: str) -> pd.Series:
    df = pd.read_csv(csv_path)

    mask = df["country"].astype(str).str.strip().str.lower() == country.strip().lower()
    d = df.loc[mask].copy()
    if d.empty:
        raise ValueError(f"No rows found for country='{country}'.")

    d["date"] = pd.to_datetime(
        d["year"].astype(str) + "-" + d["month"].astype(str),
        format="%Y-%B",
        errors="coerce",
    )
    d["arrivals"] = pd.to_numeric(d["arrivals"], errors="coerce")
    d = d.dropna(subset=["date", "arrivals"])

    s = d.groupby("date")["arrivals"].sum().sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq=FREQ)
    s = s.reindex(idx).interpolate("time")
    s = s.loc[:pd.to_datetime(end_date)]
    s.name = "arrivals"
    return s

try:
    with st.spinner("Loading data..."):
        y = load_and_prepare(csv_path, country, end_date)
except Exception as e:
    st.error(f"Failed to load/prepare data: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Country", country)
c2.metric("Date range", f"{y.index.min().date()} → {y.index.max().date()}")
c3.metric("Observations", str(len(y)))

with st.expander("Data preview", expanded=False):
    st.dataframe(y.reset_index().rename(columns={"index": "date"}).head(24))

if int(test_months) >= len(y):
    st.error("Test months too large for series length.")
    st.stop()

train = y.iloc[:-int(test_months)].copy()
test = y.iloc[-int(test_months):].copy()


# =============================
# 1) EDA
# =============================
st.header("1) Exploratory Data Analysis")
plot_series(y, f"Monthly Tourist Arrivals – {country} (up to {end_date})")
plot_train_test(train, test, "Train/Test Split")

with st.expander("Descriptive statistics", expanded=False):
    st.write(y.describe())
    month_means = y.groupby(y.index.month).mean()
    fig, ax = plt.subplots()
    ax.bar(range(1, 13), month_means.values)
    ax.set_title("Average Arrivals by Month-of-Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average arrivals")
    st.pyplot(fig)

st.subheader("Seasonal decomposition")
try:
    decomp = seasonal_decompose(y, model="multiplicative", period=S)
    fig = decomp.plot()
    fig.suptitle("Seasonal Decomposition (multiplicative)", y=1.02)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Decomposition failed: {e}")


# =============================
# 2) Stationarity + FORCE D=1 + choose d
# =============================
st.header("2) Stationarity Tests and Differencing (D forced to 1)")

left, right = st.columns(2)
with left:
    st.subheader("ADF (train)")
    st.json({"ADF (c)": adf_test(train, "c"), "ADF (ct)": adf_test(train, "ct")})
with right:
    st.subheader("KPSS (train)")
    st.json({"KPSS (c)": kpss_test(train, "c"), "KPSS (ct)": kpss_test(train, "ct")})

best_dd, d_candidates = choose_d_with_forced_D(train, s=S, alpha=float(alpha), max_d=MAX_D, forced_D=FORCED_D)
d_sel, D_sel = int(best_dd["d"]), FORCED_D

st.subheader("Chosen differencing")
st.write({"d": d_sel, "D": D_sel, "s": S, "rule": "D forced = 1; choose minimal d that satisfies ADF p<alpha and KPSS p>alpha"})

with st.expander("All d candidates (D fixed)", expanded=False):
    st.dataframe(pd.DataFrame(d_candidates).sort_values(["stationary", "d"], ascending=[False, True]), hide_index=True)

train_diff = apply_differencing(train, d=d_sel, D=D_sel, s=S)

st.subheader(f"Re-test AFTER differencing (d={d_sel}, D={D_sel})")
l, r = st.columns(2)
with l:
    st.json({"ADF (c)": adf_test(train_diff, "c"), "ADF (ct)": adf_test(train_diff, "ct")})
with r:
    st.json({"KPSS (c)": kpss_test(train_diff, "c"), "KPSS (ct)": kpss_test(train_diff, "ct")})

st.subheader("ACF / PACF (original train)")
plot_acf_pacf(train, lags=int(acf_pacf_plot_lags))
st.subheader(f"ACF / PACF (differenced train: d={d_sel}, D={D_sel})")
plot_acf_pacf(train_diff, lags=int(acf_pacf_plot_lags))


# =============================
# 3) SARIMA: choose p,q,P,Q from ACF/PACF then select best via diagnostics
# =============================
st.header("3) SARIMA Order Proposal (ACF/PACF) + Best Model Selection")

try:
    grid = propose_sarima_grid_from_acf_pacf(
        train_diff,
        s=S,
        max_nonseasonal=int(pq_max_lag),
        seasonal_mults=SEASONAL_MULTS_DEFAULT,
    )
except Exception as e:
    st.error(f"Could not propose SARIMA orders from ACF/PACF: {e}")
    st.stop()

st.write(f"Significance threshold used for spikes: **±{grid['thr']:.4f}** (≈ 1.96/sqrt(N))")
st.write("Candidate orders suggested from ACF/PACF on differenced series:")
st.write({
    "p candidates": grid["p_list"],
    "q candidates": grid["q_list"],
    "P candidates": grid["P_list"],
    "Q candidates": grid["Q_list"],
    "d": d_sel,
    "D": D_sel,
    "s": S,
})

total_models = len(grid["p_list"]) * len(grid["q_list"]) * len(grid["P_list"]) * len(grid["Q_list"])
st.caption(f"Total SARIMA candidates to fit: {total_models}")

rows = []
progress = st.progress(0)
tried = 0

with st.spinner("Fitting SARIMA candidates (from ACF/PACF) ..."):
    for p_ in grid["p_list"]:
        for q_ in grid["q_list"]:
            for P_ in grid["P_list"]:
                for Q_ in grid["Q_list"]:
                    tried += 1
                    if total_models > 0:
                        progress.progress(min(int(tried / total_models * 100), 100))

                    order = (p_, d_sel, q_)
                    seasonal_order = (P_, D_sel, Q_, S)

                    try:
                        res = fit_sarima(train, order, seasonal_order)
                        diag = sarima_diag(res, lb_lags=lb_lags, alpha=float(alpha))
                        rows.append({
                            "order": str(order),
                            "seasonal_order": str(seasonal_order),
                            "AIC": float(res.aic),
                            "BIC": float(res.bic),
                            "LB_min_p": diag["lb_min_p"],
                            "LB_pass": diag["lb_pass"],
                            "DW": diag["dw"],
                            "DW_dist_to_2": diag["dw_dist"],
                            "model_obj": res,
                        })
                    except Exception:
                        continue

progress.progress(100)

if not rows:
    st.error("No SARIMA models successfully fitted. Reduce candidate sets or check data.")
    st.stop()

dfm = pd.DataFrame(rows)

primary = selection_primary  # "BIC" or "AIC"
secondary = "AIC" if primary == "BIC" else "BIC"

dfm_sorted = dfm.sort_values(
    by=["LB_pass", primary, secondary, "DW_dist_to_2"],
    ascending=[False, True, True, True]
).reset_index(drop=True)

st.subheader("Top SARIMA candidates")
st.dataframe(dfm_sorted[["order","seasonal_order","AIC","BIC","LB_min_p","LB_pass","DW","DW_dist_to_2"]].head(25), hide_index=True)

best_row = dfm_sorted.iloc[0]
sarima_best = best_row["model_obj"]

st.subheader("Selected BEST SARIMA")
st.write({
    "order": best_row["order"],
    "seasonal_order": best_row["seasonal_order"],
    "AIC": float(best_row["AIC"]),
    "BIC": float(best_row["BIC"]),
    "Ljung-Box min p-value": best_row["LB_min_p"],
    "Ljung-Box pass (min p > alpha)": bool(best_row["LB_pass"]),
    "Durbin–Watson": best_row["DW"],
})

# Forecast best SARIMA
fc_obj = sarima_best.get_forecast(steps=len(test))
sarima_fc = pd.Series(fc_obj.predicted_mean, index=test.index)
sarima_ci = fc_obj.conf_int()
sarima_ci.index = test.index

fig, ax = plt.subplots()
ax.plot(train.index, train.values, label="Train")
ax.plot(test.index, test.values, label="Test")
ax.plot(test.index, sarima_fc.values, label="Best SARIMA forecast")
ax.fill_between(test.index, sarima_ci.iloc[:, 0].values, sarima_ci.iloc[:, 1].values, alpha=0.2, label="95% CI")
ax.set_title("Best SARIMA Forecast with 95% CI")
ax.set_xlabel("Date")
ax.set_ylabel("Arrivals")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

with st.expander("Best SARIMA summary", expanded=False):
    st.text(str(sarima_best.summary()))


# =============================
# 4) Seasonal Naive baseline (ONLY baseline)
# =============================
st.header("4) Baseline: Seasonal Naive")

snaive_fc = seasonal_naive_forecast(train, steps=len(test), s=S, freq=FREQ).reindex(test.index)

baseline_tbl = pd.DataFrame([{
    "model": f"Seasonal Naive (s={S})",
    "MAE": mae(test, snaive_fc),
    "RMSE": rmse(test, snaive_fc),
    "MAPE_%": safe_mape(test, snaive_fc),
}])

st.dataframe(baseline_tbl, hide_index=True)
plot_forecast(train, test, snaive_fc, "Seasonal Naive forecast vs Test", label=f"Seasonal Naive (s={S})")


# =============================
# 5) ETS (after SARIMA)
# =============================
st.header("5) ETS (Holt-Winters)")

ets_fc = None
try:
    with st.spinner("Fitting ETS..."):
        ets_fit = fit_ets(train, seasonal_period=S, seasonal=str(ets_seasonal))
        ets_fc = pd.Series(ets_fit.forecast(len(test)), index=test.index)

    ets_tbl = pd.DataFrame([{
        "model": f"ETS(trend=add, seasonal={ets_seasonal})",
        "MAE": mae(test, ets_fc),
        "RMSE": rmse(test, ets_fc),
        "MAPE_%": safe_mape(test, ets_fc),
    }])
    st.dataframe(ets_tbl, hide_index=True)
    plot_forecast(train, test, ets_fc, "ETS forecast vs Test", label=f"ETS ({ets_seasonal})")

    with st.expander("ETS summary", expanded=False):
        st.text(str(ets_fit.summary()))
except Exception as e:
    st.warning(f"ETS failed: {e}")


# =============================
# 6) Compare (Seasonal Naive vs ETS vs Best SARIMA)
# =============================
st.header("6) Model Comparison (Test Metrics)")

cmp_rows = [
    {"model": f"Seasonal Naive (s={S})", "MAE": mae(test, snaive_fc), "RMSE": rmse(test, snaive_fc), "MAPE_%": safe_mape(test, snaive_fc)},
    {"model": "Best SARIMA (selected)", "MAE": mae(test, sarima_fc), "RMSE": rmse(test, sarima_fc), "MAPE_%": safe_mape(test, sarima_fc)},
]
if ets_fc is not None:
    cmp_rows.append({"model": f"ETS ({ets_seasonal})", "MAE": mae(test, ets_fc), "RMSE": rmse(test, ets_fc), "MAPE_%": safe_mape(test, ets_fc)})

cmp = pd.DataFrame(cmp_rows).sort_values("RMSE")
st.dataframe(cmp, hide_index=True)

# Plot best two models by RMSE
top2 = cmp["model"].tolist()[:2]
forecast_map = {
    f"Seasonal Naive (s={S})": snaive_fc,
    "Best SARIMA (selected)": sarima_fc,
    f"ETS ({ets_seasonal})": ets_fc,
}
sel = {name: forecast_map[name] for name in top2 if forecast_map.get(name) is not None}

fig, ax = plt.subplots()
ax.plot(train.index, train.values, label="Train")
ax.plot(test.index, test.values, label="Test")
for name, fc in sel.items():
    ax.plot(test.index, fc.values, label=name)
ax.set_title("Top models (by RMSE) vs Test")
ax.set_xlabel("Date")
ax.set_ylabel("Arrivals")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)


# =============================
# 7) Diagnostics (Best SARIMA) - NO QQ plot
# =============================
st.header("7) Diagnostics (Best SARIMA Residuals)")

resid = pd.Series(sarima_best.resid, index=train.index).dropna()

fig, ax = plt.subplots()
ax.plot(resid.index, resid.values)
ax.set_title("Residuals over Time (Best SARIMA)")
ax.set_xlabel("Date")
ax.set_ylabel("Residual")
fig.autofmt_xdate()
st.pyplot(fig)

st.subheader("Residual ACF")
acf_vals = acf(resid.dropna(), nlags=min(48, len(resid) - 1), fft=True)
fig, ax = plt.subplots()
ax.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
ax.set_title("Residual ACF")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
st.pyplot(fig)

st.subheader("Ljung–Box (H0: no residual autocorrelation)")
lb_lags_list = [l for l in list(lb_lags) if l < len(resid)]
if lb_lags_list:
    lb_df = acorr_ljungbox(resid, lags=lb_lags_list, return_df=True)
    st.dataframe(lb_df)
else:
    st.warning("Not enough residual points for Ljung–Box at chosen lags.")

dw_val = float(durbin_watson(resid.values))
st.metric("Durbin–Watson", f"{dw_val:.3f}")

st.success("Done. SARIMA was proposed by ACF/PACF and selected using Ljung–Box + Durbin–Watson + AIC/BIC (no sidebar, no QQ plot, seasonal naive only).")