import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# core params
S0 = np.random.uniform(0.0000001, 500)
T = 1
N = DAILY = 252 # T is the time horizon in years, the N is the num of divisions you want to make on T to make delta_t (num of steps), the time step, 252 is for daily
delta_t = T / N
num_configs = 10
num_paths_per_config = 10

# model params
mu = np.random.uniform(0.1, 0.6, size=(num_configs)) # expected annual return
sigma = np.random.uniform(0, 1, size=(num_configs)) # annualized sd

# reshaping these so that they broadcast well in the .cumsum later, with z
mu = mu[:, np.newaxis, np.newaxis]
sigma = sigma[:, np.newaxis, np.newaxis]

z = np.random.standard_normal((num_configs, num_paths_per_config, N))

# S(t) = S(t-1) * exp((mu - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * z)), call the argument of exp a(t)
# which after manipulation gives S(t) = S(0) * exp(SUM a(t))

# constant mu and sigma
# S = np.empty((num_paths, N + 1))
# S[:, 0] = S0
# S[:, 1:] = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * z, axis=1)) # 1 = row-wise

# different configs of mu & sigma
S = np.empty((num_configs, num_paths_per_config, N + 1))
S[:, :, 0] = S0
S[:, :, 1:] = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * z, axis=2)) # 1 = row-wise

# log return r_t = ln(S(t) / S(t-1)) = ln(S(t)) - ln(S(t-1)), is normal for geometric brownian motion 
log_S = np.diff(np.log(S), axis=2)

# stats stuff
annualized_mean_log_S = log_S.mean(axis=2).mean(axis=1) * DAILY # take the mean/std per config, and then the mean of the configs
annualized_vol_log_S = log_S.std(axis=2).mean(axis=1) * np.sqrt(DAILY)

# sharpe ratio = (R - R_f) / sigma_R, where R is the avg return of asset, R_f risk-free rate, sigma_R = std of returns
R_f = 0.03 # example risk-free
sharpe = (annualized_mean_log_S - R_f) / annualized_vol_log_S # make sure the timeframes align, remember annuities

# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
plt.style.use("dark_background")
mpl.rcParams['font.family'] = 'serif'

_, axes = plt.subplots(2, 2, figsize=(18, 9))

# 1. Price paths
stock_paths = axes[0, 0].plot(np.linspace(0, T, N + 1), S.mean(axis=1).T, alpha=0.4)
axes[0, 0].set_title("Average GBM Stock Paths")
axes[0, 0].set_xlabel("Time (years)")
axes[0, 0].set_ylabel("Price")
axes[0, 0].grid(True)

stock_path_colors = [stock_path.get_color() for stock_path in stock_paths]

# 2. Log return histogram
axes[0, 1].hist(log_S.flatten(), bins=num_configs * num_paths_per_config, edgecolor='black', alpha=0.7)
axes[0, 1].set_title("Histogram of Log Returns")
axes[0, 1].set_xlabel("Log Return")
axes[0, 1].set_ylabel("Frequency")

# 3. Stats plot
axes[1, 0].scatter(annualized_vol_log_S, annualized_mean_log_S, alpha=0.6, color=stock_path_colors)
axes[1, 0].set_xlabel("Annualized Volatility")
axes[1, 0].set_ylabel("Annualized Mean Return")
axes[1, 0].set_title("Mean-Volatility of Simulated Paths")
axes[1, 0].grid(True)

# 4. Sharpe plot
sharpe_ratio_bars = axes[1, 1].bar(range(num_paths_per_config), sharpe, color=stock_path_colors)
axes[1, 1].set_xticks([])
axes[1, 1].set_ylabel("Sharpe Ratio")
axes[1, 1].set_title("Sharpe Ratios of Stock Paths")

for i, bar in enumerate(sharpe_ratio_bars):
    yval = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, yval + 0.02 if yval > 0 else yval - 0.3, f"{sharpe[i]:.2f}", 
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()