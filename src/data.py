import pandas as pd
import numpy as np

def load_weather_csv(file, datetime_col="Date Time"):
    df = pd.read_csv(file)

    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found. Columns: {list(df.columns)}")

    # ✅ your CSV is DD.MM.YYYY HH:MM:SS
    df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True, errors="coerce")

    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col)
    df = df.set_index(datetime_col)

    # remove duplicate timestamps if any
    df = df[~df.index.duplicated(keep="first")]

    return df


def weekly_aggregate(df, target_col="T (degC)"):
    """
    Converts to weekly frequency.
    Uses mean for continuous variables (temperature), which is standard for weather.
    """
    if target_col not in df.columns:
        # Try common alternatives
        candidates = ["degC", "Temp", "Temperature", "T", "temp", "temperature"]
        for c in candidates:
            if c in df.columns:
                target_col = c
                break
        else:
            raise ValueError(
                f"Target column not found. Tried '{target_col}' and common alternatives. "
                f"Available columns: {list(df.columns)}"
            )

    # Keep only numeric columns for aggregation
    numeric = df.select_dtypes(include=[np.number]).copy()

    # Weekly mean; you can switch to 'W-SUN' or 'W-MON' if you want consistent week boundary
    weekly = numeric.resample("W").mean()

    # keep target name stable
    if target_col not in weekly.columns:
        # if target was not numeric in original, it won't be here
        raise ValueError(f"Target column '{target_col}' is not numeric or missing after aggregation.")

    return weekly, target_col


def split_train_test(weekly_df, test_year=2016, target_col="degC"):
    """
    Train: all dates strictly before Jan 1 test_year
    Test: all weeks within test_year
    """
    y = weekly_df[target_col].dropna()

    train = y[y.index < f"{test_year}-01-01"]
    test = y[(y.index >= f"{test_year}-01-01") & (y.index < f"{test_year+1}-01-01")]

    if len(train) < 60:
        raise ValueError("Training set too small. Check date parsing or resampling.")
    if len(test) < 10:
        raise ValueError("Test set too small. Check the specified test_year or data coverage.")

    return train, test


def robust_outlier_handling(y, z_thresh=4.0):
    """
    Safer than dropping rows: marks extreme points as NaN then interpolates in time.
    """
    y = y.copy()
    z = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    y = y.mask(z.abs() > z_thresh)
    y = y.interpolate(method="time")
    return y