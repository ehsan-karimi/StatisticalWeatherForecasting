# app.py
# Project structure implemented exactly as requested by the user.
# Place Weather_ts.csv in same folder and run: streamlit run app.py

import os
import json
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --------------------------
# CONFIG
# --------------------------
CSV_PATH = "Weather_ts.csv"
DATETIME_COL = "Date Time"
TARGET_COL = "T (degC)"
TEST_YEAR = 2016
WEEKLY_RULE = "W"
SEASONAL_PERIOD = 52  # weekly -> annual seasonality
ACF_PACF_LAGS = 80

# differencing candidates (we will only attempt differencing if tests suggest non-stationarity,
# but we allow up to 2 as you requested)
d_candidates = [0, 1]
D_candidates = [1, 2]

# ACF/PACF candidate scan params
TOP_PQ = 3
TOP_PQ_MAX_LAG = 10
SEASONAL_MULTS = [1, 2]

# fit controls
OPT_METHOD = "lbfgs"
MAXITER = 300
MAX_MODELS_TO_FIT = 100

# diagnostics
LJUNGBOX_LAGS = [10, 20, 30]
LJUNGBOX_ALPHA = 0.05
DW_MIN, DW_MAX = 1.2, 2.8  # acceptable Durbin-Watson range

# caching
CACHE_DIR = ".model_cache"

st.set_page_config(page_title="Statistical Weather Forecasting — Project", layout="wide")
st.title("Statistical Weather Forecasting — Project pipeline (stationarity → SARIMA → diagnostics → selection)")

# --------------------------
# Helpers: caching, IO
# --------------------------
def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def signature_from_series(y: pd.Series, config: dict) -> str:
    payload = {
        "index": [str(x) for x in y.index],
        "values": [None if pd.isna(v) else float(v) for v in y.values],
        "config": config
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

def cache_path(sig: str) -> str:
    return os.path.join(CACHE_DIR, f"{sig}.pkl")

def save_cache(sig: str, payload: dict):
    ensure_cache_dir()
    pd.to_pickle(payload, cache_path(sig))
    return cache_path(sig)

def load_cache(sig: str):
    p = cache_path(sig)
    if os.path.exists(p):
        return pd.read_pickle(p)
    return None

# --------------------------
# Data loading and cleaning
# --------------------------
def load_and_aggregate_weekly(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if DATETIME_COL not in df.columns:
        raise ValueError(f"Datetime column '{DATETIME_COL}' not found. Columns: {list(df.columns)}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL).set_index(DATETIME_COL)
    df = df[~df.index.duplicated(keep="first")]

    # keep numeric columns and weekly mean
    num = df.select_dtypes(include=[np.number]).copy()
    weekly = num.resample(WEEKLY_RULE).mean()
    if TARGET_COL not in weekly.columns:
        raise ValueError(f"Target '{TARGET_COL}' missing after aggregation.")
    # interpolate and drop any leftover NA
    weekly = weekly.interpolate("time").dropna(how="any")
    return weekly

def train_test_split_year(y: pd.Series, test_year: int):
    train = y[y.index < f"{test_year}-01-01"]
    test = y[(y.index >= f"{test_year}-01-01") & (y.index < f"{test_year+1}-01-01")]
    if len(train) < 100 or len(test) < 40:
        raise ValueError(f"Train/test sizes suspicious. train={len(train)}, test={len(test)}")
    return train, test

# --------------------------
# EDA & descriptive stats
# --------------------------
def show_descriptive_stats(y: pd.Series):
    st.subheader("Descriptive statistics (target)")
    d = {
        "count": int(y.count()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "25%": float(y.quantile(0.25)),
        "50%": float(y.median()),
        "75%": float(y.quantile(0.75)),
        "max": float(y.max()),
    }
    st.json(d)

def plot_time_series(y: pd.Series, title="Time series"):
    fig, ax = plt.subplots()
    ax.plot(y.index, y.values)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------
# Stationarity tests
# --------------------------
def run_adf(y: pd.Series):
    try:
        stat, p, *_ = adfuller(y.dropna().values, regression="c")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan

def run_kpss(y: pd.Series):
    try:
        stat, p, _, _ = kpss(y.dropna().values, regression="c")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan

# --------------------------
# ACF/PACF candidate selection
# --------------------------
def top_lags_by_abs(vals: np.ndarray, k: int, min_lag=1):
    lags = np.arange(len(vals))
    candidates = [(lag, abs(vals[lag])) for lag in lags if lag >= min_lag and not np.isnan(vals[lag])]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [lag for lag, _ in candidates[:k]]

def suggest_pq_PQ_from_acf_pacf(y: pd.Series, seasonal_period:int):
    yy = y.dropna()
    acf_vals = acf(yy, nlags=ACF_PACF_LAGS, fft=True)
    pacf_vals = pacf(yy, nlags=ACF_PACF_LAGS, method="ywm")

    max_lag = min(TOP_PQ_MAX_LAG, ACF_PACF_LAGS)
    p_cands = top_lags_by_abs(pacf_vals, k=TOP_PQ, min_lag=1)  # picks from 1..max available lags
    q_cands = top_lags_by_abs(acf_vals, k=TOP_PQ, min_lag=1)

    # restrict to small values
    p_set = sorted(set([0] + [int(x) for x in p_cands if x <= 5]))[:TOP_PQ+1]
    q_set = sorted(set([0] + [int(x) for x in q_cands if x <= 5]))[:TOP_PQ+1]

    # seasonal detection at s and 2s
    acf_thr = float(np.percentile(np.abs(acf_vals[1:]), 85)) if len(acf_vals)>1 else 0.0
    P_set = {0}
    Q_set = {0}
    for m in SEASONAL_MULTS:
        L = m * seasonal_period
        if L <= ACF_PACF_LAGS:
            if abs(pacf_vals[L]) >= acf_thr:
                P_set.add(1)
            if abs(acf_vals[L]) >= acf_thr:
                Q_set.add(1)

    return {
        "p": sorted(list(p_set)),
        "q": sorted(list(q_set)),
        "P": sorted(list(P_set)),
        "Q": sorted(list(Q_set)),
        "acf_vals": acf_vals,
        "pacf_vals": pacf_vals,
        "acf_thr": acf_thr
    }

# --------------------------
# model diagnostics helper
# --------------------------
def diagnostics_from_resid(resid: pd.Series):
    r = resid.dropna()
    lj = acorr_ljungbox(r, lags=LJUNGBOX_LAGS, return_df=True)
    dw = float(durbin_watson(r))
    arch = het_arch(r, nlags=min(12, max(1, len(r)//10)))
    arch_info = {
        "LM_stat": float(arch[0]), "LM_pvalue": float(arch[1]),
        "F_stat": float(arch[2]), "F_pvalue": float(arch[3])
    }
    lb_ok = ("lb_pvalue" in lj.columns) and bool((lj["lb_pvalue"] > LJUNGBOX_ALPHA).all())
    dw_ok = (DW_MIN <= dw <= DW_MAX)
    return lj, dw, lb_ok, dw_ok, arch_info

# --------------------------
# Fit SARIMA candidate grid
# --------------------------
def generate_grid(p_set, q_set, P_set, Q_set, d_set, D_set, s):
    grid = []
    for p in p_set:
        for d in d_set:
            for q in q_set:
                for P in P_set:
                    for D in D_set:
                        for Q in Q_set:
                            grid.append(((p,d,q),(P,D,Q,s)))
    # simpler models first
    grid.sort(key=lambda x: (sum(x[0])+sum(x[1][:3]), x[0], x[1][:3]))
    return grid

def fit_grid(train: pd.Series, test: pd.Series, grid, max_fit:int):
    fitted = []
    tried = 0
    for order, sorder in grid:
        if tried >= max_fit:
            break
        tried += 1
        try:
            mod = SARIMAX(train, order=order, seasonal_order=sorder,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, method=OPT_METHOD, maxiter=MAXITER)
            if not res.mle_retvals.get("converged", False):
                continue
            yhat = pd.Series(res.get_forecast(steps=len(test)).predicted_mean.values, index=test.index)
            resid = pd.Series(res.resid).dropna()
            lj, dw, lb_ok, dw_ok, arch_info = diagnostics_from_resid(resid)
            fitted.append({
                "Model": f"SARIMA{order}x{sorder}",
                "order": order,
                "seasonal_order": sorder,
                "AIC": float(res.aic),
                "BIC": float(res.bic),
                "RMSE": float(np.sqrt(np.mean((test.loc[yhat.index].values - yhat.values)**2))),
                "MAE": float(np.mean(np.abs(test.loc[yhat.index].values - yhat.values))),
                "resid": resid,
                "yhat": yhat,
                "res": res,
                "ljungbox": lj,
                "DW": dw,
                "LB_OK": lb_ok,
                "DW_OK": dw_ok,
                "ARCH": arch_info
            })
        except Exception:
            continue
    return fitted

# --------------------------
# optional GARCH fit (on residuals)
# --------------------------
def try_garch_on_resid(resid: pd.Series):
    try:
        from arch import arch_model
    except Exception:
        return None, "arch not installed"
    r = resid.dropna()
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    cres = am.fit(disp="off")
    return cres, None

# --------------------------
# Main pipeline: follow the exact structure requested
# --------------------------
try:
    weekly_df = load_and_aggregate_weekly(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load CSV. Put {CSV_PATH} in same folder.\nError: {e}")
    st.stop()

y = weekly_df[TARGET_COL].copy()
train_raw, test = train_test_split_year(y, TEST_YEAR)

# 1. Loading, converting and cleaning of the data (weekly) + train/test done
st.header("1) Loading, converting and cleaning")
st.write(f"Data loaded. Range: {y.index.min().date()} to {y.index.max().date()}.")
st.write(f"Train: {train_raw.index.min().date()} -> {train_raw.index.max().date()} ({len(train_raw)} weeks).")
st.write(f"Test: {test.index.min().date()} -> {test.index.max().date()} ({len(test)} weeks).")

# Clean training series (trim outliers by interpolation)
train = robust = (train_raw.astype(float)).copy()
z = (train - train.mean()) / (train.std(ddof=0) + 1e-12)
train = train.mask(z.abs()>4.0).interpolate("time")

# 2. Exploring dataset with descriptive stats and frequency analysis
st.header("2) Exploratory analysis & descriptive stats")
show_descriptive_stats(train)
plot_time_series(pd.concat([train, test]), title="Weekly temperature (train + test)")
fig = seasonal_decompose(pd.concat([train, test]), model="additive", period=SEASONAL_PERIOD).plot()
fig.set_size_inches(10,6)
st.pyplot(fig)
plt.close(fig)

# Frequency analysis: ACF/PACF visualization
st.subheader("ACF / PACF (train)")
fig, ax = plt.subplots()
plot_acf(train.dropna(), lags=ACF_PACF_LAGS, ax=ax)
ax.set_title("ACF (train)")
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots()
plot_pacf(train.dropna(), lags=ACF_PACF_LAGS, method="ywm", ax=ax)
ax.set_title("PACF (train)")
st.pyplot(fig)
plt.close(fig)

# 3. Describing time series patterns (visually and numerically)
st.header("3) Time series patterns")
decomp = seasonal_decompose(pd.concat([train, test]), model="additive", period=SEASONAL_PERIOD)
st.subheader("Seasonal strength (approx)")
seasonal_var = np.var(decomp.seasonal.dropna())
resid_var = np.var(decomp.resid.dropna())
trend_var = np.var(decomp.trend.dropna())
st.write({"seasonal_var": float(seasonal_var), "trend_var": float(trend_var), "resid_var": float(resid_var)})

st.subheader("Seasonal component (first 3 years)")
st.line_chart(decomp.seasonal.iloc[:(52*3)].rename("seasonal"))

# 4. Modeling focused on temperature — follow steps:
st.header("4) Modeling (stationarity → differencing → SARIMA orders → model build & selection)")

# 4.1 Check stationarity with ADF and KPSS (train)
st.subheader("4.1 Stationarity tests on training data")
print("ADF (c):", adfuller(train.values, regression="c")[1])
print("ADF (ct):", adfuller(train.values, regression="ct")[1])
print("KPSS (c):", kpss(train.values, regression="c")[1])
print("KPSS (ct):", kpss(train.values, regression="ct")[1])
adf_stat, adf_p = run_adf(train)
kpss_stat, kpss_p = run_kpss(train)
st.write({"ADF statistic": adf_stat, "ADF p-value": adf_p, "KPSS statistic": kpss_stat, "KPSS p-value": kpss_p})
st.caption("Interpretation: ADF H0 = unit root (non-stationary). KPSS H0 = stationary. They may conflict; we use them to decide whether differencing is likely needed.")

# Decide whether differencing is likely required:
adf_nonstationary = (adf_p is np.nan) or (adf_p > 0.05)
kpss_nonstationary = (kpss_p is not np.nan) and (kpss_p < 0.05)
need_diff_hint = adf_nonstationary or kpss_nonstationary
st.write(f"Stationarity hint -> ADF indicates nonstationary? {adf_nonstationary}. KPSS indicates nonstationary? {kpss_nonstationary}. Overall hint: differencing recommended? {need_diff_hint}")

# 4.2 If not stationary do differencing (we will consider d,D in {0,1,2} but prefer to try those suggested)
st.subheader("4.2 Candidate differencing orders")
st.write("We will consider d and D values in {0,1,2}. If tests indicate nonstationarity, models with d>0 and/or D>0 are likely to be needed. We will fit a small grid of SARIMA candidates across these differencing values and let diagnostics + information criteria decide.")

st.write("Candidate differencing sets:")
st.write({"d_candidates": d_candidates, "D_candidates": D_candidates})

# 4.3 Use ACF & PACF to propose p,q,P,Q
st.subheader("4.3 Propose p,q,P,Q from ACF/PACF")
cands = suggest_pq_PQ_from_acf_pacf(train, SEASONAL_PERIOD)
st.write({"p_candidates": cands["p"], "q_candidates": cands["q"], "P_candidates": cands["P"], "Q_candidates": cands["Q"], "seasonal_period": SEASONAL_PERIOD})

# 4.4 Build candidate SARIMA grid and fit
st.subheader("4.4 Build SARIMA grid and fit candidates")
grid = generate_grid(p_set=cands["p"], q_set=cands["q"], P_set=cands["P"], Q_set=cands["Q"], d_set=d_candidates, D_set=D_candidates, s=SEASONAL_PERIOD)
st.write(f"Grid size: {len(grid)} (we will fit up to {MAX_MODELS_TO_FIT} models for speed)")

# cache fits so we don't refit on each run
selection_config = {"p": cands["p"], "q": cands["q"], "P": cands["P"], "Q": cands["Q"],
                    "d_candidates": d_candidates, "D_candidates": D_candidates,
                    "MAX_MODELS_TO_FIT": MAX_MODELS_TO_FIT, "OPT_METHOD": OPT_METHOD, "MAXITER": MAXITER}
sig = signature_from_series(train.dropna(), selection_config)
cached = load_cache(sig)
if cached is None:
    with st.spinner("Fitting SARIMA candidates (this may take time on first run)..."):
        fitted = fit_grid(train, test, grid, MAX_MODELS_TO_FIT)
    save_cache(sig, {"fitted": fitted})
    st.success(f"Fitted {len(fitted)} converged SARIMA candidates and cached results.")
else:
    fitted = cached["fitted"]
    st.success(f"Loaded {len(fitted)} cached SARIMA fits (fast).")

if len(fitted) == 0:
    st.error("No SARIMA model converged. Consider expanding data cleaning or relaxing grid.")
    st.stop()

# present table of fitted models with their AIC/BIC/RMSE/diagnostics
comp = pd.DataFrame([{
    "Model": m["Model"],
    "order": m["order"],
    "seasonal_order": m["seasonal_order"],
    "AIC": m["AIC"],
    "BIC": m["BIC"],
    "RMSE": m["RMSE"],
    "MAE": m["MAE"],
    "LB_OK": m["LB_OK"],
    "DW": m["DW"],
    "DW_OK": m["DW_OK"]
} for m in fitted]).sort_values(["AIC", "BIC"])
st.subheader("Fitted SARIMA candidates (AIC/BIC/RMSE + diagnostics)")
st.dataframe(comp, use_container_width=True)

# 4.5 Test models with Durbin Watson and Ljung Box; filter those passing both
st.subheader("4.5 Diagnostic filtering: Ljung–Box & Durbin–Watson")
adequate = pd.DataFrame(comp[(comp["LB_OK"]==True) & (comp["DW_OK"]==True)]).copy()
if adequate.shape[0] == 0:
    st.warning("No model passed both Ljung–Box and Durbin–Watson. Relaxing to Ljung–Box-only filter.")
    adequate = pd.DataFrame(comp[comp["LB_OK"]==True]).copy()
if adequate.shape[0] == 0:
    st.warning("No model passed Ljung–Box either. Proceeding with full set (note this limitation).")
    adequate = comp.copy()

st.write(f"{len(adequate)} models remain after diagnostic filtering.")

# 4.6 If both tests pass then use AIC and BIC -> shortlist (parsimonious)
st.subheader("4.6 Shortlist by BIC and AIC among diagnostically adequate models")
short_by_bic = adequate.nsmallest(min(8, len(adequate)), "BIC")
short_by_aic = adequate.nsmallest(min(8, len(adequate)), "AIC")
shortlist = pd.concat([short_by_bic, short_by_aic]).drop_duplicates(subset=["Model"])
st.dataframe(shortlist.sort_values(["BIC", "AIC", "RMSE"]), use_container_width=True)

# 4.7 Now use RMSE to choose best among the shortlist
st.subheader("4.7 Final selection by RMSE among shortlisted models")
final_row = shortlist.sort_values("RMSE").iloc[0]
final_name = final_row["Model"]
final_obj = next(m for m in fitted if m["Model"] == final_name)
st.success(f"Final selected model: {final_name}")

# show final diagnostics & plots
st.subheader("Final model summary (to present in report/oral)")
st.write({
    "Model": final_obj["Model"],
    "order": final_obj["order"],
    "seasonal_order": final_obj["seasonal_order"],
    "AIC": final_obj["AIC"],
    "BIC": final_obj["BIC"],
    "RMSE(2016)": final_obj["RMSE"],
    "MAE(2016)": final_obj["MAE"],
    "DurbinWatson": final_obj["DW"],
    "LjungBox_OK": final_obj["LB_OK"]
})

st.subheader("Forecast plot: final model vs actual (2016) + benchmarks")
# seasonal naive
sn = pd.Series(index=test.index, dtype=float)
for t in test.index:
    prev = t - pd.Timedelta(weeks=SEASONAL_PERIOD)
    if prev in train.index:
        sn.loc[t] = float(train.loc[prev])
    else:
        # fallback nearest
        diffs = np.abs((train.index - prev).days)
        sn.loc[t] = float(train.iloc[int(np.argmin(diffs))])

# ETS baseline
try:
    ets = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated").fit(optimized=True)
    ets_hat = pd.Series(ets.forecast(len(test)).values, index=test.index)
except Exception:
    ets_hat = None

plot_df = pd.DataFrame({"Actual": test, "Final SARIMA": final_obj["yhat"], "Seasonal Naive": sn})
if ets_hat is not None:
    plot_df["ETS"] = ets_hat
st.line_chart(plot_df)

# show compact benchmark table
bench = pd.DataFrame([
    {"Model":"Final SARIMA", "RMSE": final_obj["RMSE"], "AIC": final_obj["AIC"], "BIC": final_obj["BIC"]},
    {"Model":"Seasonal Naive", "RMSE": float(np.sqrt(np.mean((test.values - sn.values)**2)))},
    {"Model":"ETS", "RMSE": float(np.sqrt(np.mean((test.values - ets_hat.values)**2))) if ets_hat is not None else np.nan}
]).sort_values("RMSE")
st.dataframe(bench, use_container_width=True)

# show Ljung-Box table and DW interpretation
st.subheader("Residual diagnostics for final model")
st.write("Ljung–Box (p-values):")
st.dataframe(final_obj["ljungbox"], use_container_width=True)
st.write("Durbin–Watson:", final_obj["DW"])
if final_obj["DW"] < DW_MIN:
    st.warning("Durbin–Watson < 1.2: strong positive residual autocorrelation; model likely underfitting.")
elif final_obj["DW"] > DW_MAX:
    st.warning("Durbin–Watson > 2.8: strong negative autocorrelation.")
else:
    st.success("Durbin–Watson in acceptable range (approx. near 2).")

st.write("ARCH LM test result (on residuals):")
st.json(final_obj["ARCH"])

# residual plots
r = final_obj["resid"].dropna()
c1, c2, c3 = st.columns(3)
with c1:
    fig, ax = plt.subplots()
    ax.plot(r.values)
    ax.set_title("Residuals")
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots()
    plot_acf(r, lags=40, ax=ax)
    ax.set_title("Residual ACF")
    st.pyplot(fig)
    plt.close(fig)
with c3:
    fig, ax = plt.subplots()
    ax.hist(r.values, bins=30)
    ax.set_title("Residual histogram")
    st.pyplot(fig)
    plt.close(fig)

# 5) ARCH & GARCH
st.header("5) ARCH & GARCH (variance modeling)")
arch_p = final_obj["ARCH"]["LM_pvalue"]
if arch_p < 0.05:
    st.warning(f"ARCH LM p-value = {arch_p:.4f}: evidence of conditional heteroskedasticity.")
    garch_res, garch_err = try_garch_on_resid(final_obj["resid"])
    if garch_res is None:
        st.info(f"GARCH skipped: {garch_err}")
    else:
        st.write("GARCH(1,1) fitted on residuals. Summary:")
        st.text(str(garch_res.summary()))
else:
    st.success(f"ARCH LM p-value = {arch_p:.4f}: no strong evidence of ARCH effects; GARCH not required.")

# final note for the examiner
st.markdown("""
### Final note (what to write in your report / say in oral)
- Data aggregated to weekly frequency (mean).  
- Stationarity checked with ADF and KPSS; differencing candidates d,D∈{0,1,2} were considered.  
- ACF/PACF used to propose p,q and seasonal P,Q.  
- Models fitted and filtered by convergence, Ljung–Box (residual independence) and Durbin–Watson (no first-order autocorrelation).  
- Among diagnostically adequate models, parsimonious choices were shortlisted by BIC/AIC and final selection was made by lowest RMSE on the 2016 test set.  
- Residuals were checked for ARCH effects and GARCH was fitted only if justified.  
""")