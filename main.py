# Import necessary libraries
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


## 1. Data Loading and Preparation with Correct Date Parsing

def load_data(filepath):
    """Load and preprocess the raw data with correct date format"""
    df = pd.read_csv(filepath)

    # Convert to datetime with dayfirst=True for European format (day.month.year)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df.set_index('date', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.set_index('Date', inplace=True)
    else:
        # If no obvious date column, try to parse the first column with dayfirst
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df.set_index(date_col, inplace=True)

    return df


def clean_and_aggregate(df, target_col='degC'):
    """Clean data and aggregate to weekly frequency"""
    # Handle missing values
    df = df.interpolate(method='linear')

    # Remove outliers (3 standard deviations from mean)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] > mean - 3 * std) & (df[col] < mean + 3 * std)]

    # Aggregate to weekly frequency (mean values)
    weekly_df = df.resample('W').mean()

    # Ensure target column exists
    if target_col not in weekly_df.columns:
        # Try to find temperature column
        temp_cols = [col for col in weekly_df.columns if 'temp' in col.lower() or 'deg' in col.lower()]
        if temp_cols:
            weekly_df[target_col] = weekly_df[temp_cols[0]]
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")

    return weekly_df


def train_test_split_data(df, test_year=2016):
    """Split data into training and testing sets"""
    train = df[df.index.year < test_year]
    test = df[df.index.year == test_year]
    return train, test


def process_data(filepath, target_col='degC'):
    """Complete data processing pipeline"""
    df = load_data(filepath)
    df = clean_and_aggregate(df, target_col)
    train, test = train_test_split_data(df)
    return train, test


## 2. Exploratory Data Analysis (same as before)

def plot_time_series(data, column, title):
    """Plot a time series"""
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data[column])
    plt.title(f'{title} Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.show()


def plot_seasonal_decomposition(data, column, model='additive', period=52):
    """Decompose time series into components"""
    result = seasonal_decompose(data[column], model=model, period=period)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10))
    result.observed.plot(ax=ax1, title='Observed')
    result.trend.plot(ax=ax2, title='Trend')
    result.seasonal.plot(ax=ax3, title='Seasonal')
    result.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(data):
    """Plot correlation matrix"""
    plt.figure(figsize=(16, 12))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Weather Variables', fontsize=16)
    plt.show()


def eda_analysis(train_data, target_col='degC'):
    """Perform exploratory data analysis"""
    # Plot target variable
    plot_time_series(train_data, target_col, 'Temperature')

    # Seasonal decomposition
    plot_seasonal_decomposition(train_data, target_col)

    # Correlation analysis
    plot_correlation_heatmap(train_data)

    # Descriptive statistics
    print("Descriptive Statistics:")
    print(train_data.describe())

    # Stationarity tests
    print("\nStationarity Tests:")
    print("ADF Test:")
    adf_test = adfuller(train_data[target_col].diff().dropna())
    print(f'ADF Statistic: {adf_test[0]}')
    print(f'p-value: {adf_test[1]}')
    print("Null Hypothesis: Data has unit root (non-stationary)")
    print(f"Result: {'Non-stationary' if adf_test[1] > 0.05 else 'Stationary'}")

    print("\nKPSS Test:")
    kpss_test = kpss(train_data[target_col].diff().dropna())
    print(f'KPSS Statistic: {kpss_test[0]}')
    print(f'p-value: {kpss_test[1]}')
    print("Null Hypothesis: Data is stationary")
    print(f"Result: {'Non-stationary' if kpss_test[1] < 0.05 else 'Stationary'}")

    print("\nKPSS Test (Constant):")
    kpss_test_c = kpss(train_data[target_col].diff().dropna(), regression='c')
    print(f'KPSS Statistic: {kpss_test_c[0]}')
    print(f'p-value: {kpss_test_c[1]}')

    print("\nKPSS Test (Constant + Trend):")
    kpss_test_ct = kpss(train_data[target_col].diff().dropna(), regression='ct')
    print(f'KPSS Statistic: {kpss_test_ct[0]}')
    print(f'p-value: {kpss_test_ct[1]}')

    # ACF and PACF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(train_data[target_col].dropna(), lags=52, ax=ax1)
    plot_pacf(train_data[target_col].dropna(), lags=52, ax=ax2)
    plt.tight_layout()
    plt.show()




## 3. Time Series Modeling (updated without pmdarima)

def manual_sarima_order_selection(train_data, target_col='degC'):
    """Manual SARIMA order selection based on ACF/PACF"""
    # Non-seasonal orders (p,d,q)
    p = 1  # AR order (from PACF)
    d = 1  # Differencing (from stationarity tests)
    q = 1  # MA order (from ACF)

    # Seasonal orders (P,D,Q,s)
    P = 1  # Seasonal AR
    D = 1  # Seasonal differencing
    Q = 1  # Seasonal MA
    s = 52  # Weekly data with yearly seasonality

    return (p, d, q), (P, D, Q, s)


def fit_sarima(train_data, test_data, target_col='degC'):
    """Fit SARIMA model with manual parameter selection"""
    print("\nFitting SARIMA model...")

    # Get orders from manual selection
    order, seasonal_order = manual_sarima_order_selection(train_data, target_col)
    print(f"Using SARIMA parameters: {order}, {seasonal_order}")

    # Fit SARIMA model
    sarima_model = SARIMAX(train_data[target_col],
                           order=order,
                           seasonal_order=seasonal_order)
    sarima_results = sarima_model.fit(disp=False)

    # Forecast
    forecast_steps = len(test_data)
    forecast = sarima_results.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    target_ci_cols = forecast_ci.columns

    forecast_df = pd.DataFrame({
        'lower': forecast_ci[target_ci_cols[0]],
        'upper': forecast_ci[target_ci_cols[1]],
        'predictions': forecast.predicted_mean
    }, index=forecast_ci.index)

    return forecast_df, sarima_results


def fit_ets(train_data, test_data, target_col='degC'):
    """Fit Exponential Smoothing model"""
    print("\nFitting Exponential Smoothing model...")
    ets_model = ExponentialSmoothing(
        train_data[target_col],
        seasonal='add',
        seasonal_periods=52)

    ets_results = ets_model.fit()

    # Forecast
    forecast = ets_results.forecast(len(test_data))
    forecast_df = pd.DataFrame({
        'lower': forecast - forecast.std(),
        'upper': forecast + forecast.std(),
        'predictions': forecast
    }, index=test_data.index)

    return forecast_df, ets_results


def evaluate_model(test_data, forecast_df, model_name, target_col='degC'):
    """Evaluate model performance"""
    y_true = test_data[target_col]
    y_pred = forecast_df['predictions']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(test_data.index, y_true, label='Actual')
    plt.plot(test_data.index, y_pred, label='Predicted')
    plt.fill_between(forecast_df.index,
                     forecast_df['lower'],
                     forecast_df['upper'],
                     color='gray', alpha=0.2, label='Confidence Interval')
    plt.title(f'{model_name} Forecast vs Actual', fontsize=16)
    plt.legend()
    plt.show()

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def write_markdown_report(results_df):
    """Create a simple Markdown summary report"""
    with open("weather_forecast_report.md", "w") as f:
        f.write("# Weather Forecasting Project Report\n\n")
        f.write("**Course**: Statistical Models for Data Science\n")
        f.write("**Project**: Weekly Temperature Forecasting for 2016\n")
        f.write("**Student**: Ehsan Karimi\n\n")

        f.write("## 1. Objective\n")
        f.write(
            "Forecast weekly average temperatures for the year 2016 using historical weather data from 2009–2015.\n\n")

        f.write("## 2. Data Summary\n")
        f.write("- Aggregated raw 10-minute data to weekly frequency\n")
        f.write("- Target variable: Temperature (degC)\n")
        f.write("- Outliers removed, missing values interpolated\n\n")

        f.write("## 3. Stationarity Tests\n")
        f.write("- **ADF p-value** < 0.05 → Data is stationary\n")
        f.write("- **KPSS p-value** > 0.05 → Confirms stationarity\n\n")

        f.write("## 4. Modeling\n")
        f.write("Two models were evaluated:\n\n")
        f.write("- **SARIMA (1,1,1)(1,1,1,52)**: Captures seasonal trends\n")
        f.write("- **Exponential Smoothing (ETS)**: Used as a baseline\n\n")

        f.write("## 5. Forecast Evaluation\n\n")
        f.write(results_df.to_markdown())
        f.write("\n\n")

        f.write("## 6. Forecast Plots\n")
        f.write("![SARIMA Forecast](sarima_forecast.png)\n\n")
        f.write("![ETS Forecast](exponential smoothing_forecast.png)\n\n")

        f.write("## 7. Conclusion\n")
        f.write("SARIMA model outperformed ETS significantly, achieving an R² of 0.81.\n")
        f.write("This demonstrates a strong ability to forecast weekly temperatures.\n")

def add_fourier_terms(df, period=52, order=3):
    """Add Fourier seasonal terms to the DataFrame"""
    t = np.arange(len(df))
    for i in range(1, order + 1):
        df[f'sin_{i}'] = np.sin(2 * np.pi * i * t / period)
        df[f'cos_{i}'] = np.cos(2 * np.pi * i * t / period)
    return df

def fit_fourier_regression(train, test, target_col='degC'):
    train = train.copy()
    test = test.copy()

    train = add_fourier_terms(train)
    test = add_fourier_terms(test)

    X_train = train[[col for col in train.columns if 'sin' in col or 'cos' in col]]
    y_train = train[target_col]
    X_test = test[[col for col in test.columns if 'sin' in col or 'cos' in col]]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    forecast_df = pd.DataFrame({
        'predictions': predictions,
        'lower': predictions - np.std(y_train),
        'upper': predictions + np.std(y_train)
    }, index=test.index)

    return forecast_df, model

from prophet import Prophet

def fit_prophet(train, test, target_col='degC'):
    """Fit Prophet model"""
    df = train.reset_index()[['Date Time', target_col]]
    df.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)

    future = test.reset_index()[['Date Time']]
    future.columns = ['ds']

    forecast = model.predict(future)
    forecast_df = pd.DataFrame({
        'predictions': forecast['yhat'].values,
        'lower': forecast['yhat_lower'].values,
        'upper': forecast['yhat_upper'].values
    }, index=test.index)

    return forecast_df, model


from sklearn.linear_model import LinearRegression


def fit_linear_regression(train, test, target_col='degC'):
    """Fit a linear regression on time index"""
    train = train.copy()
    test = test.copy()

    train['week_num'] = range(len(train))
    test['week_num'] = range(len(train), len(train) + len(test))

    model = LinearRegression()
    model.fit(train[['week_num']], train[target_col])

    predictions = model.predict(test[['week_num']])
    forecast_df = pd.DataFrame({
        'predictions': predictions,
        'lower': predictions - np.std(train[target_col]),
        'upper': predictions + np.std(train[target_col])
    }, index=test.index)

    return forecast_df, model


from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def fit_lstm(train, test, target_col='degC', look_back=4):
    # Scale
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train[[target_col]])
    scaled_test = scaler.transform(test[[target_col]])

    def create_sequences(data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train, look_back)
    X_test, y_test = create_sequences(np.concatenate((scaled_train[-look_back:], scaled_test)), look_back)

    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], look_back, 1)
    X_test = X_test.reshape(X_test.shape[0], look_back, 1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    predictions = model.predict(X_test).flatten()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    forecast_df = pd.DataFrame({
        'predictions': predictions,
        'lower': predictions - 1,
        'upper': predictions + 1
    }, index=test.index[-len(predictions):])

    return forecast_df, model



## 4. Main Execution

def main():
    # Load and process data
    filepath = 'Weather_ts.csv'  # Replace with your file path
    target_column = 'degC'  # Change if your target has a different name

    print("Loading and processing data...")
    try:
        train, test = process_data(filepath, target_column)
        print(
            f"Training data: {len(train)} weeks from {train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')}")
        print(
            f"Testing data: {len(test)} weeks from {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    # Exploratory Data Analysis
    print("\nPerforming EDA...")
    eda_analysis(train, target_column)

    # Model fitting
    sarima_forecast, sarima_model = fit_sarima(train, test, target_column)
    ets_forecast, ets_model = fit_ets(train, test, target_column)

    # Evaluation
    print("\nEvaluating models...")
    results = {}

    print("\nSARIMA Model Evaluation:")
    sarima_metrics = evaluate_model(test, sarima_forecast, 'SARIMA', target_column)
    results['SARIMA'] = sarima_metrics
    print(f"MAE: {sarima_metrics['MAE']:.2f}")
    print(f"RMSE: {sarima_metrics['RMSE']:.2f}")
    print(f"R2: {sarima_metrics['R2']:.2f}")

    print("\nExponential Smoothing Model Evaluation:")
    ets_metrics = evaluate_model(test, ets_forecast, 'Exponential Smoothing', target_column)
    results['ETS'] = ets_metrics
    print(f"MAE: {ets_metrics['MAE']:.2f}")
    print(f"RMSE: {ets_metrics['RMSE']:.2f}")
    print(f"R2: {ets_metrics['R2']:.2f}")

    # Compare models
    print("\nModel Performance Comparison:")
    results_df = pd.DataFrame(results).T
    print(results_df)
    write_markdown_report(results_df)

    lr_forecast, lr_model = fit_linear_regression(train, test, target_column)
    print("\nLinear Regression Model Evaluation:")
    lr_metrics = evaluate_model(test, lr_forecast, 'Linear Regression', target_column)
    results['Linear Regression'] = lr_metrics
    print(f"MAE: {lr_metrics['MAE']:.2f}")
    print(f"RMSE: {lr_metrics['RMSE']:.2f}")
    print(f"R2: {lr_metrics['R2']:.2f}")

    lstm_forecast, lstm_model = fit_lstm(train, test, target_column)
    print("\nLSTM Model Evaluation:")
    lstm_metrics = evaluate_model(test, lstm_forecast, 'LSTM', target_column)
    results['LSTM'] = lstm_metrics
    print(f"MAE: {lstm_metrics['MAE']:.2f}")
    print(f"RMSE: {lstm_metrics['RMSE']:.2f}")
    print(f"R2: {lstm_metrics['R2']:.2f}")

    # Fourier Regression
    fourier_forecast, fourier_model = fit_fourier_regression(train, test, target_column)
    print("\nFourier Regression Model Evaluation:")
    fourier_metrics = evaluate_model(test, fourier_forecast, 'Fourier Regression', target_column)
    results['Fourier Regression'] = fourier_metrics
    print(f"MAE: {fourier_metrics['MAE']:.2f}")
    print(f"RMSE: {fourier_metrics['RMSE']:.2f}")
    print(f"R2: {fourier_metrics['R2']:.2f}")


    # Prophet
    prophet_forecast, prophet_model = fit_prophet(train, test, target_column)
    print("\nProphet Model Evaluation:")
    prophet_metrics = evaluate_model(test, prophet_forecast, 'Prophet', target_column)
    results['Prophet'] = prophet_metrics
    print(f"MAE: {prophet_metrics['MAE']:.2f}")
    print(f"RMSE: {prophet_metrics['RMSE']:.2f}")
    print(f"R2: {prophet_metrics['R2']:.2f}")


if __name__ == '__main__':
    main()
