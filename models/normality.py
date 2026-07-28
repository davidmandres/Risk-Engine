import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

from data.data import get_multiple_stocks_data

# setup
plt.style.use("dark_background")
mpl.rcParams['font.family'] = 'serif'
significance_level = 0.05
tickers = ["SPY", "GLD", "NVDA", "GOOGL", "BTC-USD"]
data = get_multiple_stocks_data(tickers, "2016-06-01", "2026-06-01", "1d")

# computations
# simple comps: prices + log returns
prices = {tickers[i]: data[tickers[i]]["Close"].values for i in range(len(tickers))}
log_returns = {ticker: np.diff(np.log(prices[ticker])) for ticker in tickers}

# complex comps: kurtosis, skewness, normality tests e.g. JB test, Ljung-Box test, normal test
excess_kurtosis = {ticker: stats.kurtosis(log_returns[ticker]) for ticker in tickers}
skewness = {ticker: stats.skew(log_returns[ticker]) for ticker in tickers}
jb_test_results = {ticker: stats.jarque_bera(log_returns[ticker]) for ticker in tickers}
ljung_box_test_results = {ticker: sm.stats.diagnostic.acorr_ljungbox(log_returns[ticker], lags=[10], return_df=True) for ticker in tickers}
normal_tests_results = {ticker: stats.normaltest(log_returns[ticker]) for ticker in tickers}

# plotting, for every 3 tickers, change rows
fig, axs = plt.subplots(len(tickers) // 3 + (1 if len(tickers) % 3 else 0), 3, figsize=(18, 9))
fig.suptitle(f"Distribution of Log Returns (01/06/2016 - 01/06/2026) for Selected Assets (Normalized), Significance Level: {significance_level}", fontsize=16)

for i, ticker in enumerate(tickers):
  # plot the distribution of log returns, and overlay a normal distribution for comparison
  
  axs[i // 3, i % 3].hist(log_returns[ticker], bins=50, alpha=0.7, color='blue', density=True, edgecolor='black')
  axs[i // 3, i % 3].set_xlabel('Log Returns')
  axs[i // 3, i % 3].set_ylabel('Frequency')
  axs[i // 3, i % 3].set_title(f'Distribution of Log Returns for {ticker}')
  # Overlay a normal distribution
  normal_x_axis= np.linspace(log_returns[ticker].min(), log_returns[ticker].max(), 100)
  normal_pdf_line = stats.norm.pdf(normal_x_axis, log_returns[ticker].mean(), log_returns[ticker].std())
  axs[i // 3, i % 3].plot(normal_x_axis, normal_pdf_line, 'y--', linewidth=2)

  # print the stats in the title or as text on the plot
  stats_text = f"Kurtosis: {excess_kurtosis[ticker]:.2f}\nSkewness: {skewness[ticker]:.2f}\nJB p-value: {jb_test_results[ticker][1]:.2e}\nLjung-Box p-value: {ljung_box_test_results[ticker]['lb_pvalue'].values[0]:.2e}\nNormal Test p-value: {normal_tests_results[ticker][1]:.2e}"
  axs[i // 3, i % 3].text(0.05, 0.95, stats_text, transform=axs[i // 3, i % 3].transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()