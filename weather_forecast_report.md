# Weather Forecasting Project Report

**Course**: Statistical Models for Data Science
**Project**: Weekly Temperature Forecasting for 2016
**Student**: Ehsan Karimi

## 1. Objective
Forecast weekly average temperatures for the year 2016 using historical weather data from 2009–2015.

## 2. Data Summary
- Aggregated raw 10-minute data to weekly frequency
- Target variable: Temperature (degC)
- Outliers removed, missing values interpolated

## 3. Stationarity Tests
- **ADF p-value** < 0.05 → Data is stationary
- **KPSS p-value** > 0.05 → Confirms stationarity

## 4. Modeling
Two models were evaluated:

- **SARIMA (1,1,1)(1,1,1,52)**: Captures seasonal trends
- **Exponential Smoothing (ETS)**: Used as a baseline

## 5. Forecast Evaluation

|        |     MAE |    RMSE |       R2 |
|:-------|--------:|--------:|---------:|
| SARIMA | 2.34171 | 2.88992 | 0.812486 |
| ETS    | 4.89204 | 5.39575 | 0.346321 |

## 6. Forecast Plots
![SARIMA Forecast](sarima_forecast.png)

![ETS Forecast](exponential smoothing_forecast.png)

## 7. Conclusion
SARIMA model outperformed ETS significantly, achieving an R² of 0.81.
This demonstrates a strong ability to forecast weekly temperatures.
