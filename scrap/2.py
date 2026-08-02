import numpy as np
import pandas as pd
import numpy.typing as npt

from data.data import get_multiple_stocks_data

class GBMRiskEngine:
    ROLLING_PERIODS = (20, 60)
    TIME_HORIZONS = (1, 10)
    ALPHAS = (0.05, 0.01)
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
log_returns = {}
sigmas_by_window = {}
paths_by_ticker = {}
results = {}

for ticker in tickers:
    curr_data = data[ticker]
    prices[ticker] = curr_data["Close"]

    lr = pd.Series(
        np.diff(np.log(prices[ticker].values)),
        index=prices[ticker].index[1:]
    )
    log_returns[ticker] = lr

    mu_annual = float(lr.mean() * risk_eng.DAILY)

    sigmas_by_window[ticker] = {
        f"vol_{window}d": float(
            lr.rolling(window).std().dropna().iloc[-1] * np.sqrt(risk_eng.DAILY)
        )
        for window in risk_eng.ROLLING_PERIODS
    }

    paths_by_ticker[ticker] = {}
    results[ticker] = {}

    for window, sigma_annual in sigmas_by_window[ticker].items():
        paths = risk_eng.simulate_gbm(
            S0=prices[ticker].iloc[-1],
            mu_annual=mu_annual,
            sigma_annual=sigma_annual,
            T=1.0,
            N=risk_eng.DAILY,
            n_paths=risk_eng.N_PATHS
        )

        paths_by_ticker[ticker][window] = paths
        results[ticker][window] = {}

        for horizon in risk_eng.TIME_HORIZONS:
            horizon_key = f"{horizon}d"
            results[ticker][window][horizon_key] = {}

            for alpha in risk_eng.ALPHAS:
                var, es = risk_eng.var_es(paths, horizon, alpha)
                conf = int((1 - alpha) * 100)

                results[ticker][window][horizon_key][f"var_{conf}"] = var
                results[ticker][window][horizon_key][f"es_{conf}"] = es


                