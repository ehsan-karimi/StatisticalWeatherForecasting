# Tourism forecasting project (Italy monthly arrivals, pre-COVID: up to 2019-12)
# End-to-end: load -> clean -> EDA -> stationarity tests -> baselines -> SARIMA + ETS
# -> evaluation -> diagnostics -> plots

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


# -----------------------------
# 1) Load, clean, aggregate
# -----------------------------
CSV_PATH = "arrivals_1989_2024.csv"

COUNTRY = "Italy"
END_DATE = "2019-12-31"      # pre-COVID cut
FREQ = "MS"                  # Month Start frequency
SEASONAL_PERIOD = 12         # monthly seasonality
TEST_MONTHS = 24             # last 24 months as test (you can change to 12)

def load_italy_monthly_arrivals(csv_path: str, country: str = "Italy", end_date: str = "2019-12-31") -> pd.Series:
    df = pd.read_csv(csv_path)

    # Normalize country matching
    mask = df["country"].astype(str).str.strip().str.lower() == country.strip().lower()
    d = df.loc[mask].copy()

    # month is month name (e.g., January). Build datetime at month start.
    d["date"] = pd.to_datetime(
        d["year"].astype(str) + "-" + d["month"].astype(str),
        format="%Y-%B",
        errors="coerce",
    )
    d = d.dropna(subset=["date"])

    # Ensure arrivals numeric
    d["arrivals"] = pd.to_numeric(d["arrivals"], errors="coerce")
    d = d.dropna(subset=["arrivals"])

    # Aggregate total arrivals per month
    s = d.groupby("date")["arrivals"].sum().sort_index()

    # Regularize to monthly index (fill missing months if any)
    idx = pd.date_range(s.index.min(), s.index.max(), freq=FREQ)
    s = s.reindex(idx)

    # Time interpolation for any gaps, then cut to end_date
    s = s.interpolate("time")
    s = s.loc[:pd.to_datetime(end_date)]

    s.name = "arrivals"
    return s

y = load_italy_monthly_arrivals(CSV_PATH, COUNTRY, END_DATE)

print("Series name:", y.name)
print("Date range:", y.index.min().date(), "to", y.index.max().date())
print("Observations:", len(y))
print(y.head(), "\n")
print(y.tail())


# -----------------------------
# 2) Train/Test split
# -----------------------------
if TEST_MONTHS >= len(y):
    raise ValueError("TEST_MONTHS is too large for the dataset length.")

train = y.iloc[:-TEST_MONTHS].copy()
test = y.iloc[-TEST_MONTHS:].copy()

print("\nTrain range:", train.index.min().date(), "to", train.index.max().date(), "N=", len(train))
print("Test  range:", test.index.min().date(), "to", test.index.max().date(), "N=", len(test))


# -----------------------------
# 3) EDA plots
# -----------------------------
plt.figure()
plt.plot(y.index, y.values)
plt.title(f"Monthly Tourist Arrivals – {COUNTRY} (pre-COVID)")
plt.xlabel("Date")
plt.ylabel("Arrivals")
plt.show()

plt.figure()
plt.plot(train.index, train.values, label="Train")
plt.plot(test.index, test.values, label="Test")
plt.title("Train/Test Split")
plt.xlabel("Date")
plt.ylabel("Arrivals")
plt.legend()
plt.show()

# Seasonal decomposition (try additive and multiplicative)
# Multiplicative needs strictly positive values; arrivals should be >0
decomp_add = seasonal_decompose(y, model="additive", period=SEASONAL_PERIOD)
decomp_mul = seasonal_decompose(y, model="multiplicative", period=SEASONAL_PERIOD)

decomp_add.plot()
plt.suptitle("Seasonal Decomposition (Additive)", y=1.02)
plt.show()

decomp_mul.plot()
plt.suptitle("Seasonal Decomposition (Multiplicative)", y=1.02)
plt.show()


# -----------------------------
# 4) Stationarity tests
# -----------------------------
def adf_test(series: pd.Series, regression: str = "c"):
    # regression: "c" constant, "ct" constant+trend
    res = adfuller(series.dropna().values, regression=regression, autolag="AIC")
    return {"stat": res[0], "pvalue": res[1], "lags": res[2], "nobs": res[3]}

def kpss_test(series: pd.Series, regression: str = "c"):
    # regression: "c" level stationary, "ct" trend stationary
    stat, pvalue, lags, crit = kpss(series.dropna().values, regression=regression, nlags="auto")
    return {"stat": stat, "pvalue": pvalue, "lags": lags, "crit": crit}

print("\nADF (c):", adf_test(train, "c"))
print("ADF (ct):", adf_test(train, "ct"))
print("\nKPSS (c):", kpss_test(train, "c"))
print("KPSS (ct):", kpss_test(train, "ct"))

# ACF/PACF on train
LAGS = 48
plt.figure()
plot_acf(train, lags=LAGS, zero=False)
plt.title("ACF (Train)")
plt.show()

plt.figure()
plot_pacf(train, lags=LAGS, zero=False, method="ywm")
plt.title("PACF (Train)")
plt.show()


# -----------------------------
# 5) Forecast baselines
# -----------------------------
def forecast_naive_last(train: pd.Series, steps: int) -> pd.Series:
    return pd.Series([train.iloc[-1]] * steps, index=pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq=FREQ))

def forecast_seasonal_naive(train: pd.Series, steps: int, seasonal_period: int = 12) -> pd.Series:
    # Repeat last seasonal_period values
    last_season = train.iloc[-seasonal_period:].values
    vals = np.resize(last_season, steps)
    return pd.Series(vals, index=pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq=FREQ))

def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)

naive_fc = forecast_naive_last(train, len(test))
snaive_fc = forecast_seasonal_naive(train, len(test), SEASONAL_PERIOD)

# Align to test index
naive_fc = naive_fc.reindex(test.index)
snaive_fc = snaive_fc.reindex(test.index)

baseline_results = []
baseline_results.append(("Naive", mae(test, naive_fc), rmse(test, naive_fc), mape(test, naive_fc)))
baseline_results.append(("Seasonal Naive (12)", mae(test, snaive_fc), rmse(test, snaive_fc), mape(test, snaive_fc)))

print("\nBaselines:")
for name, _mae, _rmse, _mape in baseline_results:
    print(f"{name:20s}  MAE={_mae:,.2f}  RMSE={_rmse:,.2f}  MAPE={_mape:.2f}%")

plt.figure()
plt.plot(train.index, train.values, label="Train")
plt.plot(test.index, test.values, label="Test")
plt.plot(test.index, naive_fc.values, label="Naive")
plt.plot(test.index, snaive_fc.values, label="Seasonal Naive (12)")
plt.title("Baselines vs Test")
plt.xlabel("Date")
plt.ylabel("Arrivals")
plt.legend()
plt.show()


# -----------------------------
# 6) Model 1: ETS / Holt-Winters
# -----------------------------
# Often tourism is multiplicative seasonality; try both
def fit_ets(train: pd.Series, seasonal: str = "mul"):
    # trend can be 'add' or None; we'll use additive trend here, seasonal as chosen
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)

ets_mul = fit_ets(train, seasonal="mul")
ets_add = fit_ets(train, seasonal="add")

ets_mul_fc = pd.Series(ets_mul.forecast(len(test)), index=test.index)
ets_add_fc = pd.Series(ets_add.forecast(len(test)), index=test.index)

print("\nETS results:")
for name, fc in [("ETS(add trend, mul season)", ets_mul_fc), ("ETS(add trend, add season)", ets_add_fc)]:
    print(f"{name:28s}  MAE={mae(test, fc):,.2f}  RMSE={rmse(test, fc):,.2f}  MAPE={mape(test, fc):.2f}%")


# -----------------------------
# 7) Model 2: SARIMA grid search (small, sane grid)
# -----------------------------
# You can expand these ranges if needed, but keep it small for speed.
p = [0, 1, 2, 3]
d = [0, 1]
q = [0, 1, 2, 3]

P = [0, 1]
D = [0, 1]
Q = [0, 1]

def fit_sarima(train: pd.Series, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    return res

best = {"aic": np.inf, "order": None, "seasonal_order": None, "model": None}

for pi in p:
    for di in d:
        for qi in q:
            for Pi in P:
                for Di in D:
                    for Qi in Q:
                        order = (pi, di, qi)
                        seasonal_order = (Pi, Di, Qi, SEASONAL_PERIOD)
                        try:
                            res = fit_sarima(train, order, seasonal_order)
                            if res.aic < best["aic"]:
                                best.update({"aic": res.aic, "order": order, "seasonal_order": seasonal_order, "model": res})
                        except Exception:
                            continue

print("\nBest SARIMA by AIC:")
print("AIC:", best["aic"])
print("order:", best["order"], "seasonal_order:", best["seasonal_order"])

sarima_best = best["model"]
sarima_fc = sarima_best.get_forecast(steps=len(test))
sarima_mean = pd.Series(sarima_fc.predicted_mean, index=test.index)
sarima_ci = sarima_fc.conf_int()
sarima_ci.index = test.index

print("\nBest SARIMA test metrics:")
print(f"MAE={mae(test, sarima_mean):,.2f}  RMSE={rmse(test, sarima_mean):,.2f}  MAPE={mape(test, sarima_mean):.2f}%")


# -----------------------------
# 8) Compare models on test
# -----------------------------
models = [
    ("Naive", naive_fc),
    ("Seasonal Naive (12)", snaive_fc),
    ("ETS mul", ets_mul_fc),
    ("ETS add", ets_add_fc),
    ("SARIMA best", sarima_mean),
]

results = []
for name, fc in models:
    results.append((name, mae(test, fc), rmse(test, fc), mape(test, fc)))

res_df = pd.DataFrame(results, columns=["model", "MAE", "RMSE", "MAPE_%"]).sort_values("RMSE")
print("\nModel comparison (sorted by RMSE):")
print(res_df.to_string(index=False))

plt.figure()
plt.plot(train.index, train.values, label="Train")
plt.plot(test.index, test.values, label="Test")
for name, fc in models:
    if name in ["SARIMA best", "ETS mul"]:  # show the top contenders + keep plot readable
        plt.plot(test.index, fc.values, label=name)

plt.title("Forecasts vs Actual (Test)")
plt.xlabel("Date")
plt.ylabel("Arrivals")
plt.legend()
plt.show()

# SARIMA with confidence intervals
plt.figure()
plt.plot(train.index, train.values, label="Train")
plt.plot(test.index, test.values, label="Test")
plt.plot(test.index, sarima_mean.values, label="SARIMA forecast")
plt.fill_between(
    test.index,
    sarima_ci.iloc[:, 0].values,
    sarima_ci.iloc[:, 1].values,
    alpha=0.2,
    label="95% CI",
)
plt.title("SARIMA Forecast with 95% CI")
plt.xlabel("Date")
plt.ylabel("Arrivals")
plt.legend()
plt.show()


# -----------------------------
# 9) Diagnostics for best SARIMA
# -----------------------------
resid = sarima_best.resid.dropna()

plt.figure()
plt.plot(resid.index, resid.values)
plt.title("SARIMA Residuals over Time")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.show()

plt.figure()
plot_acf(resid, lags=48, zero=False)
plt.title("Residual ACF")
plt.show()

lb = acorr_ljungbox(resid, lags=[12, 24, 36], return_df=True)
dw = durbin_watson(resid.values)

print("\nResidual diagnostics:")
print("Durbin-Watson:", float(dw))
print("\nLjung-Box (residual autocorrelation):")
print(lb)

# Optional: QQ plot (normality check)
import statsmodels.api as sm
plt.figure()
sm.qqplot(resid.values, line="s")
plt.title("Residual QQ Plot")
plt.show()