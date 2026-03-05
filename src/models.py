from typing import Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

@dataclass
class ModelResult:
    name: str
    yhat: pd.Series
    aic: Optional[float]
    bic: Optional[float]
    residuals: Optional[pd.Series]
    notes: str = ""

def seasonal_naive_forecast(train, test, seasonal_period=52, name="SeasonalNaive"):
    # forecast each test point as value from one season ago
    shifted = train.shift(seasonal_period)
    yhat = shifted.reindex(test.index)
    return ModelResult(name=name, yhat=yhat, aic=None, bic=None, residuals=(test - yhat), notes=f"period={seasonal_period}")

def mean_forecast(train, test, name="Mean"):
    yhat = pd.Series(train.mean(), index=test.index)
    return ModelResult(name=name, yhat=yhat, aic=None, bic=None, residuals=(test - yhat))

def arima_forecast(train, test, order=(1,1,1), name=None):
    name = name or f"ARIMA{order}"
    model = ARIMA(train, order=order)
    res = model.fit()
    fc = res.get_forecast(steps=len(test))
    yhat = pd.Series(fc.predicted_mean.values, index=test.index)
    residuals = res.resid
    return ModelResult(name=name, yhat=yhat, aic=float(res.aic), bic=float(res.bic), residuals=residuals)

def ets_forecast(train, test, seasonal_period=52, trend="add", seasonal="add", name="ETS"):
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_period,
        initialization_method="estimated",
    )
    res = model.fit(optimized=True)
    yhat = pd.Series(res.forecast(len(test)).values, index=test.index)
    # ETS in statsmodels doesn't always expose AIC/BIC consistently across versions, so keep None if missing
    aic = float(getattr(res, "aic", np.nan))
    bic = float(getattr(res, "bic", np.nan))
    aic = None if np.isnan(aic) else aic
    bic = None if np.isnan(bic) else bic
    residuals = train - res.fittedvalues
    return ModelResult(name=name, yhat=yhat, aic=aic, bic=bic, residuals=residuals, notes=f"{trend}/{seasonal}, period={seasonal_period}")

def sarima_forecast(train, test, order=(1,1,1), seasonal_order=(1,1,1,52), name=None):
    name = name or f"SARIMA{order}x{seasonal_order}"
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=len(test))
    yhat = pd.Series(fc.predicted_mean.values, index=test.index)
    residuals = res.resid
    return ModelResult(name=name, yhat=yhat, aic=float(res.aic), bic=float(res.bic), residuals=residuals)

def sarima_grid_search(train, p=(0,1,2), d=(0,1), q=(0,1,2), P=(0,1), D=(0,1), Q=(0,1), s=52, top_k=5):
    """
    Returns top_k models by AIC.
    """
    results = []
    for pi in p:
        for di in d:
            for qi in q:
                for Pi in P:
                    for Di in D:
                        for Qi in Q:
                            try:
                                res = SARIMAX(
                                    train,
                                    order=(pi,di,qi),
                                    seasonal_order=(Pi,Di,Qi,s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                results.append(((pi,di,qi),(Pi,Di,Qi,s), float(res.aic), float(res.bic)))
                            except Exception:
                                continue
    results.sort(key=lambda x: x[2])  # by AIC
    return results[:top_k]

# ----------------- ARCH / GARCH (optional) -----------------

def garch_forecast(train, test, p=1, q=1, mean="AR", ar_lags=1, name=None):
    """
    Forecast mean with a simple AR mean (optional), volatility with GARCH(p,q).
    This requires `arch` package. If not installed, raise informative error.
    """
    try:
        from arch import arch_model
    except Exception as e:
        raise RuntimeError("arch package not installed. Add `arch>=6.3` to requirements.txt and pip install it.") from e

    name = name or f"GARCH({p},{q}) mean={mean}"

    # Use returns-like series for variance modeling. For temperature, we model residuals around mean.
    # A pragmatic approach:
    y = train.copy()

    # Fit ARCH model
    am = arch_model(y, mean=mean, lags=ar_lags if mean == "AR" else 0, vol="GARCH", p=p, q=q, dist="normal")
    res = am.fit(disp="off")

    # Mean forecast:
    # arch forecasts give mean + variance forecasts. We'll use mean.
    f = res.forecast(horizon=len(test), reindex=False)
    # f.mean is (T x horizon) in some versions; use last row
    mean_fc = f.mean.iloc[-1].values
    yhat = pd.Series(mean_fc, index=test.index)

    # no AIC/BIC shown consistently, but res has aic/bic
    aic = float(getattr(res, "aic", np.nan))
    bic = float(getattr(res, "bic", np.nan))
    aic = None if np.isnan(aic) else aic
    bic = None if np.isnan(bic) else bic

    residuals = res.resid
    return ModelResult(name=name, yhat=yhat, aic=aic, bic=bic, residuals=residuals, notes=f"vol=GARCH({p},{q})")