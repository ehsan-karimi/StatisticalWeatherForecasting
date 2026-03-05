import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_series(y, title="Time series"):
    fig = plt.figure()
    plt.plot(y.index, y.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y.name if y.name else "value")
    plt.tight_layout()
    return fig

def decompose_weekly(y, period=52):
    # additive decomposition is usually fine for temperature
    res = seasonal_decompose(y, model="additive", period=period)
    fig = res.plot()
    fig.set_size_inches(10, 7)
    fig.tight_layout()
    return fig

def acf_pacf_figs(y, lags=60):
    fig1 = plt.figure()
    plot_acf(y.dropna(), lags=lags)
    plt.tight_layout()

    fig2 = plt.figure()
    plot_pacf(y.dropna(), lags=lags, method="ywm")
    plt.tight_layout()

    return fig1, fig2