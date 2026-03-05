# app.py
# FINAL VERSION (Rich story-telling EDA + choose d and D automatically (D ∈ {0,1}) + SARIMA + ARIMA+Fourier + ETS + Seasonal Naive)
#
# ✅ Rich EDA (story-telling):
#    - Summary stats + skew/kurtosis + coefficient of variation
#    - Distribution + log1p distribution
#    - Month-of-year boxplot + monthly means
#    - Yearly totals + YoY annual growth
#    - Seasonal subseries plot + Year×Month heatmap
#    - Rolling mean & rolling std (trend + heteroskedasticity)
#    - Decomposition + STL robust
#    - Anomaly proxy via deviation from rolling mean (z-score)
#    - STORY 1: Peak & trough month each year + peak-month frequency
#    - STORY 2: Seasonal amplitude per year (peak-trough)
#    - STORY 3: Volatility per year (CV = std/mean)
#    - STORY 4: Volatility by month (month CV)
#    - STORY 5: YoY monthly change heatmap (t vs t-12)
#
# ✅ Stationarity:
#    - ADF & KPSS on train
#    - Automatically choose (d,D) by searching d=0..2 and D=0..1 (s=12),
#      preferring simplest (smaller D then smaller d) that passes ADF p<alpha and KPSS p>alpha
#    - Re-test after differencing + ACF/PACF on original and differenced
#
# ✅ SARIMA:
#    - Propose p,q,P,Q from ACF/PACF of differenced series
#    - Fit candidate SARIMA models, pick best by Ljung-Box pass + BIC + AIC + DW closeness
#
# ✅ Fourier model:
#    - ARIMA(p,d,q) + Fourier(K) as exogenous seasonality, with seasonal differencing D_sel allowed (0 or 1)
#
# ✅ Baseline:
#    - Seasonal Naive ONLY
#
# ✅ ETS:
#    - Holt-Winters (add trend, add/mul seasonality)
#
# ✅ Diagnostics:
#    - Residuals over time + residual ACF + Ljung-Box + Durbin–Watson (NO QQ plot)
#
# Run:
#   streamlit run app.py

import warnings
warnings.filterwarnings("ignore")

from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


# -----------------------------
# Defaults / constants
# -----------------------------
CSV_PATH_DEFAULT = "arrivals_1989_2024.csv"
COUNTRY_DEFAULT = "Italy"
END_DATE_DEFAULT = "2019-12-31"

FREQ = "MS"
S = 12

MAX_D = 2
MAX_D_SEASONAL = 1  # D ∈ {0,1}

TEST_MONTHS_DEFAULT = 24
ALPHA_DEFAULT = 0.05

ACF_PACF_PLOT_LAGS_DEFAULT = 48
PQ_MAX_LAG_DEFAULT = 6
SEASONAL_MULTS_DEFAULT = (1, 2)  # seasonal lags: 12, 24

LJUNG_LAGS_DEFAULT = (12, 24)

ETS_SEASONAL_DEFAULT = "mul"
FOURIER_K_DEFAULT = 3


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Tourism Forecasting – Italy (pre-COVID)", layout="wide")
st.title("Tourism Forecasting – Italy (Monthly Arrivals, pre-COVID)")
st.caption("Rich EDA → stationarity → choose (d,D) → SARIMA from ACF/PACF → ARIMA+Fourier → seasonal naive → ETS → compare → diagnostics")


# -----------------------------
# Metrics
# -----------------------------
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


# -----------------------------
# Plot helpers
# -----------------------------
def plot_line(y: pd.Series, title: str, ylabel: str = "Arrivals"):
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

def plot_multi_forecast(train: pd.Series, test: pd.Series, forecasts: Dict[str, Optional[pd.Series]], title: str):
    fig, ax = plt.subplots()
    ax.plot(train.index, train.values, label="Train")
    ax.plot(test.index, test.values, label="Test")
    for name, fc in forecasts.items():
        if fc is None:
            continue
        ax.plot(test.index, fc.values, label=name)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Arrivals")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_hist(series: pd.Series, title: str, xlabel: str):
    vals = pd.Series(series).dropna().values
    fig, ax = plt.subplots()
    ax.hist(vals, bins=35)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_acf_pacf(series: pd.Series, lags: int, title_prefix: str = ""):
    s = series.dropna()
    if len(s) < 10:
        st.warning("Series too short for ACF/PACF.")
        return
    lags = int(min(lags, len(s) - 1))

    acf_vals = acf(s, nlags=lags, fft=True)
    fig, ax = plt.subplots()
    ax.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
    ax.set_title(f"{title_prefix}ACF")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    st.pyplot(fig)

    pacf_vals = pacf(s, nlags=lags, method="ywm")
    fig, ax = plt.subplots()
    ax.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
    ax.set_title(f"{title_prefix}PACF")
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    st.pyplot(fig)

def plot_monthly_box(y: pd.Series, title: str):
    df = pd.DataFrame({"date": y.index, "arrivals": y.values})
    df["month"] = df["date"].dt.month
    data = [df.loc[df["month"] == m, "arrivals"].values for m in range(1, 13)]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=list(range(1, 13)))
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Arrivals")
    st.pyplot(fig)

def plot_yearly_total(y: pd.Series, title: str):
    yearly = y.resample("Y").sum()
    fig, ax = plt.subplots()
    ax.plot(yearly.index.year, yearly.values)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total arrivals (year)")
    st.pyplot(fig)

def plot_heatmap_year_month(y: pd.Series, title: str):
    df = pd.DataFrame({"arrivals": y.values}, index=y.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="arrivals", aggfunc="mean")

    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(range(12))
    ax.set_xticklabels(range(1, 13))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

def plot_seasonal_subseries(y: pd.Series, title: str):
    df = pd.DataFrame({"arrivals": y.values}, index=y.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="month", columns="year", values="arrivals", aggfunc="mean").sort_index()

    fig, ax = plt.subplots()
    for yr in pivot.columns:
        ax.plot(pivot.index, pivot[yr].values, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Arrivals")
    ax.set_xticks(range(1, 13))
    st.pyplot(fig)

def plot_rolling(y: pd.Series, window: int, title: str):
    df = pd.DataFrame({"y": y})
    df["roll_mean"] = df["y"].rolling(window).mean()
    df["roll_std"] = df["y"].rolling(window).std()
    fig, ax = plt.subplots()
    ax.plot(df.index, df["y"], label="Series", alpha=0.6)
    ax.plot(df.index, df["roll_mean"], label=f"{window}-month rolling mean")
    ax.plot(df.index, df["roll_std"], label=f"{window}-month rolling std")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Arrivals")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

# NEW helper plots for story blocks
def plot_bar(x, y, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)


# -----------------------------
# Stationarity tests
# -----------------------------
def adf_test(series: pd.Series, regression: str = "c") -> Dict[str, Any]:
    res = adfuller(series.dropna().values, regression=regression, autolag="AIC")
    return {"stat": float(res[0]), "pvalue": float(res[1]), "lags": int(res[2]), "nobs": int(res[3])}

def kpss_test(series: pd.Series, regression: str = "c") -> Dict[str, Any]:
    stat, pvalue, lags, crit = kpss(series.dropna().values, regression=regression, nlags="auto")
    return {"stat": float(stat), "pvalue": float(pvalue), "lags": int(lags), "crit": {k: float(v) for k, v in crit.items()}}

def is_stationary_by_tests(series: pd.Series, alpha: float) -> Tuple[bool, float, float]:
    a = adf_test(series, "c")["pvalue"]
    k = kpss_test(series, "c")["pvalue"]
    return (a < alpha) and (k > alpha), float(a), float(k)

def apply_differencing(series: pd.Series, d: int, D: int, s: int) -> pd.Series:
    out = series.copy()
    for _ in range(int(d)):
        out = out.diff()
    for _ in range(int(D)):
        out = out.diff(int(s))
    return out.dropna()

def choose_d_D(series: pd.Series, s: int, alpha: float, max_d: int, max_D: int) -> Tuple[Dict[str, Any], list]:
    candidates = []
    for D in range(0, max_D + 1):
        for d in range(0, max_d + 1):
            t = apply_differencing(series, d=d, D=D, s=s)
            ok, adf_p, kpss_p = is_stationary_by_tests(t, alpha=alpha)
            candidates.append({
                "d": d, "D": D, "stationary": ok,
                "adf_p": adf_p, "kpss_p": kpss_p, "n": int(len(t))
            })

    passers = [c for c in candidates if c["stationary"]]
    if passers:
        passers.sort(key=lambda x: (x["D"], x["d"], x["adf_p"], -x["kpss_p"]))
        return passers[0], candidates

    candidates.sort(key=lambda x: (x["adf_p"], -x["kpss_p"], x["D"], x["d"]))
    return candidates[0], candidates


# -----------------------------
# ACF/PACF -> SARIMA candidate proposal
# -----------------------------
def sig_threshold(n: int) -> float:
    return 1.96 / np.sqrt(max(n, 1))

def propose_sarima_grid_from_acf_pacf(series_diff: pd.Series, s: int, max_nonseasonal: int, seasonal_mults) -> Dict[str, Any]:
    sd = series_diff.dropna()
    if len(sd) < 30:
        raise ValueError("Differenced series too short for reliable ACF/PACF-based order proposal.")

    nlags_needed = max(max_nonseasonal, max(seasonal_mults) * s)
    nlags = min(nlags_needed, len(sd) - 1)

    acf_vals = acf(sd, nlags=nlags, fft=True)
    pacf_vals = pacf(sd, nlags=nlags, method="ywm")
    thr = sig_threshold(len(sd))

    p_list = sorted({0} | {lag for lag in range(1, min(max_nonseasonal, nlags) + 1) if abs(pacf_vals[lag]) > thr})
    q_list = sorted({0} | {lag for lag in range(1, min(max_nonseasonal, nlags) + 1) if abs(acf_vals[lag]) > thr})

    P_set, Q_set = {0}, {0}
    for m in seasonal_mults:
        L = m * s
        if L <= nlags:
            if abs(pacf_vals[L]) > thr:
                P_set.add(1)
            if abs(acf_vals[L]) > thr:
                Q_set.add(1)

    return {
        "p_list": p_list, "q_list": q_list,
        "P_list": sorted(P_set), "Q_list": sorted(Q_set),
        "thr": float(thr), "nlags": int(nlags)
    }


# -----------------------------
# Fourier exogenous terms
# -----------------------------
def make_fourier_terms(index: pd.DatetimeIndex, period: int, K: int) -> pd.DataFrame:
    n = len(index)
    t = np.arange(n)
    cols = {}
    for k in range(1, int(K) + 1):
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


# -----------------------------
# Models
# -----------------------------
def seasonal_naive_forecast(train: pd.Series, steps: int, s: int, freq: str) -> pd.Series:
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

def fit_sarimax(train: pd.Series, order, seasonal_order, exog: Optional[pd.DataFrame] = None):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

def sarimax_diag(res, lb_lags: Tuple[int, int], alpha: float) -> Dict[str, Any]:
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
# Inputs (NO SIDEBAR)
# -----------------------------
with st.expander("Settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    csv_path = c1.text_input("CSV path", value=CSV_PATH_DEFAULT)
    country = c2.text_input("Country", value=COUNTRY_DEFAULT)
    end_date = c3.text_input("Use data up to (YYYY-MM-DD)", value=END_DATE_DEFAULT)

    c4, c5, c6 = st.columns(3)
    test_months = c4.number_input("Test months", min_value=6, max_value=60, value=TEST_MONTHS_DEFAULT, step=1)
    alpha = c5.selectbox("Alpha (ADF/KPSS & Ljung-Box pass)", [0.01, 0.05, 0.10], index=[0.01, 0.05, 0.10].index(ALPHA_DEFAULT))
    pq_max_lag = c6.number_input("Max non-seasonal lag for p/q proposal", min_value=1, max_value=12, value=PQ_MAX_LAG_DEFAULT, step=1)

    c7, c8, c9 = st.columns(3)
    acf_pacf_plot_lags = c7.number_input("ACF/PACF plot lags", min_value=12, max_value=120, value=ACF_PACF_PLOT_LAGS_DEFAULT, step=1)
    lj1 = c8.number_input("Ljung-Box lag 1", min_value=6, max_value=48, value=LJUNG_LAGS_DEFAULT[0], step=1)
    lj2 = c9.number_input("Ljung-Box lag 2", min_value=6, max_value=72, value=LJUNG_LAGS_DEFAULT[1], step=1)

    c10, c11 = st.columns(2)
    ets_seasonal = c10.selectbox("ETS seasonality", ["mul", "add"], index=["mul", "add"].index(ETS_SEASONAL_DEFAULT))
    fourier_K = c11.number_input("Fourier K (pairs)", min_value=1, max_value=12, value=FOURIER_K_DEFAULT, step=1)

lb_lags = (int(lj1), int(lj2))


# -----------------------------
# Load & prepare series
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare(csv_path: str, country: str, end_date: str) -> pd.Series:
    df = pd.read_csv(csv_path)

    required = {"country", "year", "month", "arrivals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

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

if int(test_months) >= len(y):
    st.error("Test months too large for series length.")
    st.stop()

train = y.iloc[:-int(test_months)].copy()
test = y.iloc[-int(test_months):].copy()

m1, m2, m3 = st.columns(3)
m1.metric("Country", country)
m2.metric("Date range", f"{y.index.min().date()} → {y.index.max().date()}")
m3.metric("Observations", str(len(y)))

with st.expander("Data preview", expanded=False):
    st.dataframe(y.reset_index().rename(columns={"index": "date"}).head(24))


# =============================
# 1) RICH EDA + 5 STORY BLOCKS
# =============================
st.header("1) Exploratory Data Analysis (Rich + Story-telling)")

plot_line(y, f"Monthly tourist arrivals – {country} (up to {end_date})")

# Summary stats (skew/kurtosis + coefficient of variation)
with st.expander("EDA: Summary statistics", expanded=False):
    desc = y.describe()
    extra = pd.Series({
        "skew": float(y.skew()),
        "kurtosis": float(y.kurtosis()),
        "coef_var (std/mean)": float(y.std() / (y.mean() + 1e-9)),
        "min_date": str(y.index.min().date()),
        "max_date": str(y.index.max().date()),
    })
    st.write(pd.concat([desc, extra], axis=0))

# Distribution vs log distribution
cA, cB = st.columns(2)
with cA:
    plot_hist(y, "Distribution of monthly arrivals", "Arrivals")
with cB:
    plot_hist(np.log1p(y), "Distribution of log1p(arrivals)", "log1p(arrivals)")

# Seasonality: monthly boxplot + monthly means
st.subheader("Seasonality (month-of-year behavior)")
plot_monthly_box(y, "Monthly arrivals by month (boxplot)")

month_means = y.groupby(y.index.month).mean()
fig, ax = plt.subplots()
ax.bar(range(1, 13), month_means.values)
ax.set_title("Average arrivals by month-of-year")
ax.set_xlabel("Month")
ax.set_ylabel("Average arrivals")
st.pyplot(fig)

# Yearly totals + YoY annual growth
st.subheader("Long-term evolution (annual totals + YoY growth)")
plot_yearly_total(y, "Yearly total arrivals (sum over months)")

yearly = y.resample("Y").sum()
yoy_annual = yearly.pct_change() * 100.0
fig, ax = plt.subplots()
ax.plot(yoy_annual.index.year, yoy_annual.values)
ax.axhline(0, linewidth=1)
ax.set_title("Year-over-year growth in total arrivals (%)")
ax.set_xlabel("Year")
ax.set_ylabel("YoY %")
st.pyplot(fig)

# Seasonal stability visuals
st.subheader("Seasonality stability across years")
plot_seasonal_subseries(y, "Seasonal subseries plot (each line = one year)")
plot_heatmap_year_month(y, "Heatmap (Year × Month): arrivals level and seasonal intensity")

# Rolling mean & std
st.subheader("Rolling behavior (trend and changing variability)")
plot_rolling(y, window=12, title="12-month rolling mean & rolling std")

# Decomposition + STL
st.subheader("Decomposition and STL")
try:
    decomp = seasonal_decompose(y, model="multiplicative", period=S)
    fig = decomp.plot()
    fig.suptitle("Seasonal decomposition (multiplicative)", y=1.02)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"seasonal_decompose failed: {e}")

try:
    stl = STL(y, period=S, robust=True).fit()
    fig, ax = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(y.index, y.values); ax[0].set_title("STL: Observed")
    ax[1].plot(y.index, stl.trend); ax[1].set_title("STL: Trend")
    ax[2].plot(y.index, stl.seasonal); ax[2].set_title("STL: Seasonal")
    ax[3].plot(y.index, stl.resid); ax[3].set_title("STL: Residual")
    fig.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"STL failed: {e}")

# Anomaly proxy (deviation from rolling mean)
st.subheader("Anomaly proxy (deviation from 12-month rolling mean)")
roll = y.rolling(12).mean()
dev = (y - roll).dropna()
z = (dev - dev.mean()) / (dev.std() + 1e-9)
anoms = z[np.abs(z) > 3]

fig, ax = plt.subplots()
ax.plot(dev.index, dev.values, label="y - rolling_mean(12)")
if len(anoms) > 0:
    ax.scatter(anoms.index, dev.loc[anoms.index].values, label="|z|>3 anomalies", s=20)
ax.set_title("Deviation from rolling mean (with anomalies)")
ax.set_xlabel("Date")
ax.set_ylabel("Deviation")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

# ----- ADDITIONAL 5 STORY-TELLING EDA BLOCKS -----
df_story = pd.DataFrame({"arrivals": y.values}, index=y.index)
df_story["year"] = df_story.index.year
df_story["month"] = df_story.index.month

# Story 1: peak and trough month per year + peak month frequency
st.subheader("EDA Story 1: Peak and trough month each year (seasonality stability)")

yearly_peak = df_story.loc[df_story.groupby("year")["arrivals"].idxmax()][["year", "month", "arrivals"]].set_index("year")
yearly_trough = df_story.loc[df_story.groupby("year")["arrivals"].idxmin()][["year", "month", "arrivals"]].set_index("year")

c1, c2 = st.columns(2)
with c1:
    st.write("Peak month per year:")
    st.dataframe(yearly_peak.rename(columns={"month": "peak_month", "arrivals": "peak_arrivals"}))
with c2:
    st.write("Trough month per year:")
    st.dataframe(yearly_trough.rename(columns={"month": "trough_month", "arrivals": "trough_arrivals"}))

peak_counts = yearly_peak["month"].value_counts().sort_index()
plot_bar(
    peak_counts.index.tolist(),
    peak_counts.values.tolist(),
    "How often each month is the PEAK across years",
    "Month",
    "Count of years"
)

# Story 2: seasonal amplitude per year (peak - trough)
st.subheader("EDA Story 2: Seasonal amplitude per year (peak − trough)")

amp = (yearly_peak["arrivals"] - yearly_trough["arrivals"]).rename("amplitude")
fig, ax = plt.subplots()
ax.plot(amp.index, amp.values)
ax.set_title("Seasonal amplitude per year (peak − trough)")
ax.set_xlabel("Year")
ax.set_ylabel("Amplitude (arrivals)")
st.pyplot(fig)

# Story 3: volatility per year (CV = std/mean)
st.subheader("EDA Story 3: Volatility per year (CV = std/mean)")

year_stats = df_story.groupby("year")["arrivals"].agg(["mean", "std"])
year_stats["cv"] = year_stats["std"] / (year_stats["mean"] + 1e-9)

fig, ax = plt.subplots()
ax.plot(year_stats.index, year_stats["cv"].values)
ax.set_title("Volatility per year (Coefficient of Variation)")
ax.set_xlabel("Year")
ax.set_ylabel("CV")
st.pyplot(fig)

# Story 4: month-by-month volatility (month CV)
st.subheader("EDA Story 4: Which months are most variable across years? (month CV)")

month_stats = df_story.groupby("month")["arrivals"].agg(["mean", "std"])
month_stats["cv"] = month_stats["std"] / (month_stats["mean"] + 1e-9)

plot_bar(
    month_stats.index.tolist(),
    month_stats["cv"].values.tolist(),
    "Month-of-year volatility (CV = std/mean)",
    "Month",
    "CV"
)

# Story 5: YoY monthly change heatmap (t vs t-12)
st.subheader("EDA Story 5: YoY monthly change heatmap (t vs t−12)")

df_yoy = df_story.copy()
df_yoy["yoy_monthly_%"] = df_yoy["arrivals"].pct_change(S) * 100.0
pivot_yoy = df_yoy.pivot_table(index="year", columns="month", values="yoy_monthly_%", aggfunc="mean")

fig, ax = plt.subplots()
im = ax.imshow(pivot_yoy.values, aspect="auto")
ax.set_title("YoY monthly change (%) heatmap")
ax.set_xlabel("Month")
ax.set_ylabel("Year")
ax.set_xticks(range(12))
ax.set_xticklabels(range(1, 13))
ax.set_yticks(range(len(pivot_yoy.index)))
ax.set_yticklabels(pivot_yoy.index.tolist())
fig.colorbar(im, ax=ax)
st.pyplot(fig)


# =============================
# 2) Stationarity + choose d and D (D can be 0 or 1)
# =============================
st.header("2) Stationarity Tests and Differencing (choose d and D)")

left, right = st.columns(2)
with left:
    st.subheader("ADF (train)")
    st.json({"ADF (c)": adf_test(train, "c"), "ADF (ct)": adf_test(train, "ct")})
with right:
    st.subheader("KPSS (train)")
    st.json({"KPSS (c)": kpss_test(train, "c"), "KPSS (ct)": kpss_test(train, "ct")})

best_dd, cand = choose_d_D(train, s=S, alpha=float(alpha), max_d=MAX_D, max_D=MAX_D_SEASONAL)
d_sel, D_sel = int(best_dd["d"]), int(best_dd["D"])

st.subheader("Selected differencing")
st.write({
    "chosen d": d_sel,
    "chosen D": D_sel,
    "s": S,
    "decision rule": "Pick simplest (D,d) that satisfies ADF p<alpha and KPSS p>alpha; D allowed 0 or 1."
})

with st.expander("All (d,D) candidates", expanded=False):
    st.dataframe(pd.DataFrame(cand).sort_values(["stationary", "D", "d"], ascending=[False, True, True]), hide_index=True)

train_diff = apply_differencing(train, d=d_sel, D=D_sel, s=S)

st.subheader(f"Re-test AFTER differencing (d={d_sel}, D={D_sel}, s={S})")
l, r = st.columns(2)
with l:
    st.json({"ADF (c)": adf_test(train_diff, "c"), "ADF (ct)": adf_test(train_diff, "ct")})
with r:
    st.json({"KPSS (c)": kpss_test(train_diff, "c"), "KPSS (ct)": kpss_test(train_diff, "ct")})

st.subheader("ACF/PACF on original train")
plot_acf_pacf(train, lags=int(acf_pacf_plot_lags), title_prefix="")
st.subheader(f"ACF/PACF on differenced train (d={d_sel}, D={D_sel})")
plot_acf_pacf(train_diff, lags=int(acf_pacf_plot_lags), title_prefix="Differenced: ")


# =============================
# 3) SARIMA: propose p,q,P,Q from ACF/PACF and select best
# =============================
st.header("3) SARIMA: Orders from ACF/PACF + Best Model Selection")

grid = propose_sarima_grid_from_acf_pacf(train_diff, s=S, max_nonseasonal=int(pq_max_lag), seasonal_mults=SEASONAL_MULTS_DEFAULT)

st.write(f"Spike threshold: **±{grid['thr']:.4f}** (≈ 1.96/sqrt(N))")
st.write("Candidate sets (from ACF/PACF on differenced train):")
st.write({
    "p": grid["p_list"], "q": grid["q_list"],
    "P": grid["P_list"], "Q": grid["Q_list"],
    "d": d_sel, "D": D_sel, "s": S
})

total_models = len(grid["p_list"]) * len(grid["q_list"]) * len(grid["P_list"]) * len(grid["Q_list"])
st.caption(f"Total SARIMA candidates to fit: {total_models}")

rows = []
prog = st.progress(0)
tried = 0

with st.spinner("Fitting SARIMA candidates..."):
    for p_ in grid["p_list"]:
        for q_ in grid["q_list"]:
            for P_ in grid["P_list"]:
                for Q_ in grid["Q_list"]:
                    tried += 1
                    if total_models > 0:
                        prog.progress(min(int(tried / total_models * 100), 100))

                    order = (p_, d_sel, q_)
                    seasonal_order = (P_, D_sel, Q_, S)

                    try:
                        res = fit_sarimax(train, order=order, seasonal_order=seasonal_order, exog=None)
                        diag = sarimax_diag(res, lb_lags=lb_lags, alpha=float(alpha))
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

prog.progress(100)

if not rows:
    st.error("No SARIMA models could be fitted from the proposed grid.")
    st.stop()

dfm = pd.DataFrame(rows).sort_values(
    by=["LB_pass", "BIC", "AIC", "DW_dist_to_2"],
    ascending=[False, True, True, True]
).reset_index(drop=True)

st.subheader("Top SARIMA candidates")
st.dataframe(dfm[["order", "seasonal_order", "AIC", "BIC", "LB_min_p", "LB_pass", "DW"]].head(25), hide_index=True)

best_row = dfm.iloc[0]
best_sarima = best_row["model_obj"]

st.subheader("Selected BEST SARIMA")
st.write({
    "order": best_row["order"],
    "seasonal_order": best_row["seasonal_order"],
    "AIC": float(best_row["AIC"]),
    "BIC": float(best_row["BIC"]),
    "Ljung-Box min p": best_row["LB_min_p"],
    "Ljung-Box pass": bool(best_row["LB_pass"]),
    "Durbin–Watson": best_row["DW"],
})

sarima_fc_obj = best_sarima.get_forecast(steps=len(test))
sarima_fc = pd.Series(sarima_fc_obj.predicted_mean, index=test.index)
sarima_ci = sarima_fc_obj.conf_int()
sarima_ci.index = test.index

fig, ax = plt.subplots()
ax.plot(train.index, train.values, label="Train")
ax.plot(test.index, test.values, label="Test")
ax.plot(test.index, sarima_fc.values, label="Best SARIMA forecast")
ax.fill_between(test.index, sarima_ci.iloc[:, 0].values, sarima_ci.iloc[:, 1].values, alpha=0.2, label="95% CI")
ax.set_title("Best SARIMA forecast with 95% CI")
ax.set_xlabel("Date")
ax.set_ylabel("Arrivals")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)


# =============================
# 4) ARIMA + Fourier
# =============================
st.header("4) ARIMA + Fourier (seasonality via Fourier terms)")

order_tuple = eval(best_row["order"])  # generated internally
p_best, d_best, q_best = order_tuple

fourier_train = make_fourier_terms(train.index, period=S, K=int(fourier_K))
fourier_test = make_fourier_terms(test.index, period=S, K=int(fourier_K))

fourier_seasonal_order = (0, D_sel, 0, S)

try:
    fourier_model = fit_sarimax(
        train,
        order=(p_best, d_best, q_best),
        seasonal_order=fourier_seasonal_order,
        exog=fourier_train
    )
    diag_f = sarimax_diag(fourier_model, lb_lags=lb_lags, alpha=float(alpha))

    fourier_fc_obj = fourier_model.get_forecast(steps=len(test), exog=fourier_test)
    fourier_fc = pd.Series(fourier_fc_obj.predicted_mean, index=test.index)

    st.subheader("ARIMA+Fourier fit metrics (train)")
    st.write({
        "order": (p_best, d_best, q_best),
        "seasonal_order": fourier_seasonal_order,
        "Fourier K": int(fourier_K),
        "AIC": float(fourier_model.aic),
        "BIC": float(fourier_model.bic),
        "Ljung-Box min p": diag_f["lb_min_p"],
        "Ljung-Box pass": bool(diag_f["lb_pass"]),
        "Durbin–Watson": diag_f["dw"],
    })

    plot_multi_forecast(train, test, {"ARIMA+Fourier": fourier_fc}, "ARIMA+Fourier forecast vs Test")
except Exception as e:
    st.error(f"ARIMA+Fourier failed: {e}")
    fourier_model = None
    fourier_fc = None


# =============================
# 5) Baseline: Seasonal Naive only
# =============================
st.header("5) Baseline: Seasonal Naive")
snaive_fc = seasonal_naive_forecast(train, steps=len(test), s=S, freq=FREQ).reindex(test.index)
plot_multi_forecast(train, test, {"Seasonal Naive (s=12)": snaive_fc}, "Seasonal Naive forecast vs Test")


# =============================
# 6) ETS
# =============================
st.header("6) ETS (Holt-Winters)")
try:
    ets_fit = fit_ets(train, seasonal_period=S, seasonal=str(ets_seasonal))
    ets_fc = pd.Series(ets_fit.forecast(len(test)), index=test.index)
    plot_multi_forecast(train, test, {f"ETS ({ets_seasonal})": ets_fc}, "ETS forecast vs Test")
except Exception as e:
    st.warning(f"ETS failed: {e}")
    ets_fc = None


# =============================
# 7) Compare models
# =============================
st.header("7) Model Comparison (Test set metrics)")

rows_cmp = [
    {"model": "Best SARIMA", "RMSE": rmse(test, sarima_fc), "MAE": mae(test, sarima_fc), "MAPE_%": safe_mape(test, sarima_fc)},
    {"model": "Seasonal Naive (s=12)", "RMSE": rmse(test, snaive_fc), "MAE": mae(test, snaive_fc), "MAPE_%": safe_mape(test, snaive_fc)},
]
if ets_fc is not None:
    rows_cmp.append({"model": f"ETS ({ets_seasonal})", "RMSE": rmse(test, ets_fc), "MAE": mae(test, ets_fc), "MAPE_%": safe_mape(test, ets_fc)})
if fourier_fc is not None:
    rows_cmp.append({"model": f"ARIMA+Fourier (K={int(fourier_K)})", "RMSE": rmse(test, fourier_fc), "MAE": mae(test, fourier_fc), "MAPE_%": safe_mape(test, fourier_fc)})

cmp = pd.DataFrame(rows_cmp).sort_values("RMSE")
st.dataframe(cmp, hide_index=True)

top2 = cmp["model"].tolist()[:2]
forecast_map = {
    "Best SARIMA": sarima_fc,
    "Seasonal Naive (s=12)": snaive_fc,
    f"ETS ({ets_seasonal})": ets_fc,
    f"ARIMA+Fourier (K={int(fourier_K)})": fourier_fc,
}
sel = {name: forecast_map[name] for name in top2 if forecast_map.get(name) is not None}
plot_multi_forecast(train, test, sel, "Top-2 models (by RMSE) vs Test")


# =============================
# 8) Residual diagnostics
# =============================
st.header("8) Residual Diagnostics")

st.subheader("Best SARIMA residuals")
resid_s = pd.Series(best_sarima.resid, index=train.index).dropna()

fig, ax = plt.subplots()
ax.plot(resid_s.index, resid_s.values)
ax.set_title("Residuals over time (Best SARIMA)")
ax.set_xlabel("Date")
ax.set_ylabel("Residual")
fig.autofmt_xdate()
st.pyplot(fig)

acf_vals = acf(resid_s, nlags=min(48, len(resid_s) - 1), fft=True)
fig, ax = plt.subplots()
ax.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
ax.set_title("Residual ACF (Best SARIMA)")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
st.pyplot(fig)

lb_list = [l for l in list(lb_lags) if l < len(resid_s)]
if lb_list:
    lb_df = acorr_ljungbox(resid_s, lags=lb_list, return_df=True)
    st.dataframe(lb_df)
st.metric("Durbin–Watson (Best SARIMA)", f"{durbin_watson(resid_s.values):.3f}")

if fourier_model is not None:
    st.subheader("ARIMA+Fourier residuals")
    resid_f = pd.Series(fourier_model.resid, index=train.index).dropna()

    fig, ax = plt.subplots()
    ax.plot(resid_f.index, resid_f.values)
    ax.set_title("Residuals over time (ARIMA+Fourier)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    fig.autofmt_xdate()
    st.pyplot(fig)

    acf_vals = acf(resid_f, nlags=min(48, len(resid_f) - 1), fft=True)
    fig, ax = plt.subplots()
    ax.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
    ax.set_title("Residual ACF (ARIMA+Fourier)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    st.pyplot(fig)

    lb_list = [l for l in list(lb_lags) if l < len(resid_f)]
    if lb_list:
        lb_df = acorr_ljungbox(resid_f, lags=lb_list, return_df=True)
        st.dataframe(lb_df)
    st.metric("Durbin–Watson (ARIMA+Fourier)", f"{durbin_watson(resid_f.values):.3f}")

st.success("Final project ready: richer story-telling EDA + (d,D) chosen automatically + SARIMA + ARIMA+Fourier + ETS + Seasonal Naive + diagnostics.")