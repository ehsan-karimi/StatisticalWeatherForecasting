import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf

def residual_diagnostics(residuals, lags=(10, 20, 30)):
    """
    Returns:
      - Ljung-Box table
      - Durbin-Watson statistic
      - ARCH LM test
    """
    r = pd.Series(residuals).dropna()

    lb = acorr_ljungbox(r, lags=list(lags), return_df=True)

    dw = float(durbin_watson(r))

    # ARCH LM test
    # returns: (LM stat, LM pvalue, F stat, F pvalue)
    arch_lm = het_arch(r, nlags=min(12, max(1, len(r)//10)))
    arch_info = {
        "LM_stat": float(arch_lm[0]),
        "LM_pvalue": float(arch_lm[1]),
        "F_stat": float(arch_lm[2]),
        "F_pvalue": float(arch_lm[3]),
    }

    return lb, dw, arch_info


def residual_plots(residuals, acf_lags=40):
    r = pd.Series(residuals).dropna()

    fig1 = plt.figure()
    plt.plot(r.values)
    plt.title("Residuals over time")
    plt.tight_layout()

    fig2 = plt.figure()
    plot_acf(r, lags=acf_lags)
    plt.title("Residual ACF")
    plt.tight_layout()

    fig3 = plt.figure()
    plt.hist(r.values, bins=30)
    plt.title("Residual histogram")
    plt.tight_layout()

    return fig1, fig2, fig3