import numpy as np
import pandas as pd

from data.data import get_multiple_stocks_data

import numpy.typing as npt

class GBMRiskEngine:
  ROLLING_PERIOD = 60 # in days
  N = DAILY = 252
  SIGNIFICANCE_LEVEL = 0.05
  N_PATHS = 5

  def simulate_euler_maruyama(self, S0, mu, sigma, T, N, n_paths) -> npt.NDArray[np.float64]:
    """
    Simulates Geometric Brownian Motion (GBM) paths using the Euler-Maruyama method.
    
    Parameters:
    - S0: Initial stock price
    - mu: Drift coefficient (expected return)
    - sigma: Diffusion coefficient (volatility)
    - T: Total time horizon (in years)
    - N: Number of time steps
    - n_paths: Number of simulation paths
    """
    # euler-maruyama method
    N = int(np.floor(N))
    z = np.random.standard_normal((n_paths, N))
    delta_t = T / N
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0

    """
    General Euler-Maruyama method: X_n+1 = X_n + a(X_n, t_n) * delta_t + b(X_n, t_n) * sqrt(delta_t) * z_n
    For GBM, we have: S_n+1 = S_n + mu * S_n * delta_t + sigma * S_n * sqrt(delta_t) * z_n
    This can be rearranged to: S_n+1 = S_n * (1 + mu * delta_t + sigma * sqrt(delta_t) * z_n)
    """

    S[:, 1:] = S0 * np.cumprod(1 + mu * delta_t + sigma * np.sqrt(delta_t) * z, axis=1)

    return S

  def var(self, paths: npt.NDArray[np.float64], alpha: float) -> float:
    """
    Calculates the Value at Risk (VaR) for a given set of simulated paths and a specified confidence level (alpha) using the Monte Carlo method.

    Parameters:
    - paths: Simulated paths of the portfolio value
    - alpha: Significance level (e.g., 0.05 for 95% confidence level)

    Returns:
    - VaR: The Value at Risk at the specified confidence level
    """

    """
    Formal def: 
    "VaR is formally defined as the smallest loss that will not be exceeded with a
    given probability over a specified time frame. Formally, for a portfolio with returns
    R ={R1,R2,...,Rn}, the VaR at a confidence level α over a time horizon T is
    expressed as VaRα = inf{x ∈ R : P (L ≤ x) ≥ α}

    Here, L = P0 − PT represents the loss, where P0 is the initial value of the
    portfolio and PT is its value at time T . This definition ensures that there is at most
    (1 − α) probability that the loss L exceeds the VaR."

    Monte Carlo method uses simulated paths of the portfolio value and calculates the VaR based on the simulated losses:

    "Formally, if portfolio returns follow a multivariate normal distribution R ∼
    N(mu, SIGMA), we simulate N returns R_1, R_2, ..., R_N. For each simulated return R_i,
    we compute the loss L_i = −P_0 * R_i, where P_0 is the initial portfolio value. The VaR
    at the confidence level α is then calculated as
    VaR_α = Percentile(100 × (1 − α))
    where L = { L_1, L_2, ..., L_N } represents the set of simulated losses.
    """

    returns = np.diff(np.log(paths))  # Calculate log returns (since returns are lognormal and we want normal returns) and convert to percentage
    P_0 = paths[0, 0]  # Initial portfolio value, each path starts with the same S_0
    losses = -P_0 * returns  # Calculate losses based on returns
    print(losses)
    
    return np.percentile(losses, (1 - alpha) * 100)

  def expected_shortfall(self, paths, alpha = SIGNIFICANCE_LEVEL, var = None) -> float:
    # "Besides, VaR only considers extreme risk up to the specified quantile level when
    # it comes to capturing the tail risk while ignoring further extreme losses that occur
    # beyond the VaR threshold. To address this limitation, Expected Shortfall (ES) is
    # often used alongside VaR. ES measures the average loss that occurs when losses
    # exceed the VaR level, offering a more comprehensive understanding of potential
    # extreme losses. It is defined as
    # ES_α = E[L | L > VaR_α]"

    if var is None:
      var = self.var(paths, alpha)

    returns = np.diff(np.log(paths))
    losses = returns[returns < 0] * 100

    losses_exceeding_var = losses[losses < var] # since losses are negative, we want the losses that are less than the var (which is also negative)

    if len(losses_exceeding_var) == 0:
      return 0.0
    
    return losses_exceeding_var.mean()

risk_eng = GBMRiskEngine()

tickers = ["SPY", "GLD", "NVDA", "GOOGL", "BTC-USD"]
data: dict[str, pd.DataFrame] = get_multiple_stocks_data(tickers, "2016-06-01", "2026-06-01", "1d")

dates: dict[str, pd.Index] = {}
prices: dict[str, pd.Series] = {}
log_returns: dict[str, pd.DataFrame] = {}
annualized_rolling_vols: dict[str, pd.DataFrame] = {}
em_paths: dict[str, dict[str, npt.NDArray[np.float64]]] = {}
vars: dict[str, dict[str, dict[str, float]]] = {}
expected_shortfalls: dict[str, dict[str, dict[str, float]]] = {}

for ticker in tickers:
  curr_data = data[ticker]

  dates[ticker] = curr_data.index

  prices[ticker] = curr_data["Close"]
  log_returns[ticker] = pd.DataFrame({"Log Returns": np.diff(np.log(prices[ticker]))}, index=dates[ticker][1:])

  annualized_rolling_vols[ticker] = log_returns[ticker].rolling(window=risk_eng.ROLLING_PERIOD).std().dropna() * np.sqrt(risk_eng.DAILY)
  annualized_rolling_vols[ticker].rename(columns={"Log Returns": "Annualized Rolling Volatility"}, inplace=True)

  em_paths[ticker] = {
    "1d": risk_eng.simulate_euler_maruyama(S0=prices[ticker].iloc[-1], mu=log_returns[ticker]["Log Returns"].mean(), sigma=annualized_rolling_vols[ticker]["Annualized Rolling Volatility"].iloc[-1], T=1, N=risk_eng.N, n_paths=risk_eng.N_PATHS),
    "10d": risk_eng.simulate_euler_maruyama(S0=prices[ticker].iloc[-1], mu=log_returns[ticker]["Log Returns"].mean(), sigma=annualized_rolling_vols[ticker]["Annualized Rolling Volatility"].iloc[-1], T=1, N=risk_eng.N / 10, n_paths=risk_eng.N_PATHS)
  }

  vars[ticker] = {
    "1d": {
      "var_95": risk_eng.var(paths=em_paths[ticker]["1d"], alpha=0.05),
      "var_99": risk_eng.var(paths=em_paths[ticker]["1d"], alpha=0.01)
    },
    "10d": {
      "var_95": risk_eng.var(paths=em_paths[ticker]["10d"], alpha=0.05),
      "var_99": risk_eng.var(paths=em_paths[ticker]["10d"], alpha=0.01)
    }
  }

  expected_shortfalls[ticker] = {
    "1d": {
      "es_95": risk_eng.expected_shortfall(paths=em_paths[ticker]["1d"], var=vars[ticker]["1d"]["var_95"]),
      "es_99": risk_eng.expected_shortfall(paths=em_paths[ticker]["1d"], var=vars[ticker]["1d"]["var_99"])
    },
    "10d": {
      "es_95": risk_eng.expected_shortfall(paths=em_paths[ticker]["10d"], var=vars[ticker]["10d"]["var_95"]),
      "es_99": risk_eng.expected_shortfall(paths=em_paths[ticker]["10d"], var=vars[ticker]["10d"]["var_99"])
    }
  }

print(vars)





