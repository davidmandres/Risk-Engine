import numpy as np
import pandas as pd
import numpy.typing as npt

from data.data import get_multiple_stocks_data


class GBMRiskEngine:
    ROLLING_PERIOD = 60
    DAILY = 252
    N_PATHS = 10_000

    def simulate_euler_maruyama(
        self, 
        S0: float,
        mu_annual: float,
        sigma_annual: float,
        T: float,
        N: int,
        n_paths: int
    ) -> npt.NDArray[np.float64]:
      N = int(np.floor(N))
      z = np.random.standard_normal((n_paths, N))
      delta_t = T / N
      paths = np.zeros((n_paths, N + 1))
      paths[:, 0] = S0
      paths[:, 1:] = S0 * np.cumprod(1 + mu_annual * delta_t + sigma_annual * np.sqrt(delta_t) * z, axis=1)
      return paths

    def simulate_gbm(
        self,
        S0: float,
        mu_annual: float,
        sigma_annual: float,
        T: float,
        N: int,
        n_paths: int
    ) -> npt.NDArray[np.float64]:
        N = int(N)
        dt = T / N
        z = np.random.standard_normal((n_paths, N))

        log_increments = (
            (mu_annual - 0.5 * sigma_annual**2) * dt
            + sigma_annual * np.sqrt(dt) * z
        )

        paths = np.zeros((n_paths, N + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
        return paths

    def var_es(
        self,
        paths: npt.NDArray[np.float64],
        time_horizon: int,
        alpha: float
    ) -> tuple[float, float]:
        if time_horizon < 1 or time_horizon >= paths.shape[1]:
            raise ValueError("time_horizon must be between 1 and number of simulated steps")

        P0 = paths[0, 0]
        losses = P0 - paths[:, time_horizon]

        var = np.percentile(losses, (1 - alpha) * 100)
        es = losses[losses >= var].mean()

        return float(var), float(es)


risk_eng = GBMRiskEngine()

tickers = ["SPY", "GLD", "NVDA", "GOOGL", "BTC-USD"]
data = get_multiple_stocks_data(tickers, "2016-06-01", "2026-06-01", "1d")

prices = {}
paths_by_ticker = {}
var_results = {}
es_results = {}

for ticker in tickers:
    curr_data = data[ticker]
    prices[ticker] = curr_data["Close"]

    lr = np.diff(np.log(prices[ticker].values))

    mu_annual = lr.mean() * risk_eng.DAILY
    sigma_annual = lr.rolling(risk_eng.ROLLING_PERIOD).std().dropna().iloc[-1] * np.sqrt(risk_eng.DAILY)

    paths = risk_eng.simulate_gbm(
        S0=float(prices[ticker].iloc[-1]),
        mu_annual=float(mu_annual),
        sigma_annual=float(sigma_annual),
        T=1.0,
        N=risk_eng.DAILY,
        n_paths=risk_eng.N_PATHS
    )
    paths_by_ticker[ticker] = paths

    v95_1, e95_1 = risk_eng.var_es(paths, 1, 0.05)
    v99_1, e99_1 = risk_eng.var_es(paths, 1, 0.01)
    v95_10, e95_10 = risk_eng.var_es(paths, 10, 0.05)
    v99_10, e99_10 = risk_eng.var_es(paths, 10, 0.01)

    var_results[ticker] = {
        "1d": {"var_95": v95_1, "var_99": v99_1},
        "10d": {"var_95": v95_10, "var_99": v99_10},
    }

    es_results[ticker] = {
        "1d": {"es_95": e95_1, "es_99": e99_1},
        "10d": {"es_95": e95_10, "es_99": e99_10},
    }