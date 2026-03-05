import numpy as np

def mae(y_true, y_pred):
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def smape(y_true, y_pred):
    y_true, y_pred = _align(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom < 1e-12, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def _align(y_true, y_pred):
    # assumes pandas Series with index
    common = y_true.index.intersection(y_pred.index)
    return y_true.loc[common].values, y_pred.loc[common].values