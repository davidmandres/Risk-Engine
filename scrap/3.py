# EWMA — Volatility Dynamics
import numpy as np
import pandas as pd
import numpy.typing as npt

from data.data import get_multiple_stocks_data

class EWMAEngine:
    ROLLING_PERIODS = (20, 60)
    TIME_HORIZONS = (1, 10)
    ALPHAS = (0.05, 0.01)
    DAILY = 252
    N_PATHS = 10_000
    LAMBDA_ = 0.94

    def ewma_volatility(self, log_returns: pd.Series, lambda_: float = LAMBDA_) -> float:
        if (not 0 < lambda_ <= 1):
            raise ValueError("Lambda must be between 0 and 1")

        squared_returns = log_returns ** 2
        last_return = squared_returns.iloc[-1]
        squared_returns = squared_returns.shift(1)
        squared_returns = squared_returns.fillna(value=0)
        ewma_var = squared_returns.ewm(alpha=1 - lambda_, adjust=False).mean().iloc[-1] 

        forecast = (last_return * (1 - lambda_) + ewma_var  * lambda_) ** 0.5 * np.sqrt(self.DAILY)

        return forecast

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
        es = losses[losses > var].mean()

        return float(var), float(es)



ewma_eng = EWMAEngine()

tickers = ["SPY", "GLD", "NVDA", "GOOGL", "BTC-USD"]
data = get_multiple_stocks_data(tickers, "2016-06-01", "2026-06-01", "1d")

prices = {}
log_returns = {}
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

    mu_annual = lr.mean() * ewma_eng.DAILY 

    sigma_annual = ewma_eng.ewma_volatility(log_returns=lr)

    paths = ewma_eng.simulate_gbm(
        S0=prices[ticker].iloc[-1],
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
        T=1.0,
        N=ewma_eng.DAILY,
        n_paths=ewma_eng.N_PATHS
    )

    paths_by_ticker[ticker] = paths
    results[ticker] = {}

    for horizon in ewma_eng.TIME_HORIZONS:
        horizon_key = f"{horizon}d"
        results[ticker][horizon_key] = {}

        for alpha in ewma_eng.ALPHAS:
            var, es = ewma_eng.var_es(paths, horizon, alpha)
            conf = int((1 - alpha) * 100)

            results[ticker][horizon_key][f"var_{conf}"] = var
            results[ticker][horizon_key][f"es_{conf}"] = es

   
print("EWMA Results:")
for ticker in tickers:
    print(f"\nTicker: {ticker}")
    for horizon in ewma_eng.TIME_HORIZONS:
        horizon_key = f"{horizon}d"
        print(f"  Horizon: {horizon_key}")
        for alpha in ewma_eng.ALPHAS:
            conf = int((1 - alpha) * 100)
            var = results[ticker][horizon_key][f"var_{conf}"]
            es = results[ticker][horizon_key][f"es_{conf}"]
            print(f"    Confidence Level: {conf}%")
            print(f"      VaR: {var:.2f}")
            print(f"      ES: {es:.2f}")
        
                