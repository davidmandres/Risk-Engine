import numpy as np
import pandas as pd
import numpy.typing as npt
import scipy.stats as stats

from data.data import get_multiple_stocks_data


class BacktestingEngine:
    ROLLING_PERIODS = (20, 60)
    TIME_HORIZONS = (1, 10)
    ALPHAS = (0.95, 0.99)  # the confidence levels for VaR and ES calculations
    DAILY = 252
    N_PATHS = 10_000
    LAMBDA_ = 0.94
    T = 1.0    # Time to maturity in years for the simulation
    N = 252    # Number of time steps in the simulation (daily steps for 1 year)

    def get_conf(self, alpha: float) -> int:
        if alpha not in self.ALPHAS:
            raise ValueError(f"Alpha must be one of {self.ALPHAS}")
        return int(alpha * 100)

    def ewma_volatility(self, log_returns: pd.Series, lambda_: float = LAMBDA_) -> float:
        if not 0 < lambda_ <= 1:
            raise ValueError("Lambda must be between 0 and 1")

        squared_returns = log_returns ** 2
        last_return = squared_returns.iloc[-1]
        squared_returns = squared_returns.shift(1)
        squared_returns = squared_returns.fillna(value=log_returns.mean() ** 2)
        ewma_var = squared_returns.ewm(alpha=1 - lambda_, adjust=False).mean().iloc[-1]

        forecast = (last_return * (1 - lambda_) + ewma_var * lambda_) ** 0.5 * np.sqrt(self.DAILY)
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
        log_increments = (mu_annual - 0.5 * sigma_annual ** 2) * dt + sigma_annual * np.sqrt(dt) * z
        paths = np.zeros((n_paths, N + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
        return paths

    # --- Backtesting statistical tests (verified correct against independent manual re-derivation) ---

    def kupiec_pof_test(
        self,
        violations: int,
        observations: int,
        alpha: float
    ) -> tuple[float, float]:
        if observations <= 0:
            raise ValueError("Number of observations must be positive")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")

        non_violations = observations - violations  # n - n_1, i.e. n_0
        p_hat = violations / observations

        if p_hat == 0 or p_hat == 1:
            # log(0) is undefined; by convention, treat the null-model likelihood
            # ratio as the limiting case since n_1*ln(p_hat) -> 0 as p_hat -> 0.
            likelihood_ratio = -2 * (
                violations * np.log(1 - alpha) + non_violations * np.log(alpha)
            )
        else:
            likelihood_ratio = -2 * (
                violations * np.log(1 - alpha) + non_violations * np.log(alpha)
                - violations * np.log(p_hat) - non_violations * np.log(1 - p_hat)
            )

        # P-value based on Chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(likelihood_ratio, df=1)
        return float(likelihood_ratio), float(p_value)

    def christoffersen_independence_test(
        self,
        violations: npt.NDArray[np.int64]
    ) -> tuple[float, float]:
        if len(violations) < 2:
            raise ValueError("At least two observations are required for the Christoffersen test.")

        # n_ij denotes the number of days where I_{t-1}=i and I_t=j (I_t is the
        # violation indicator, 1=violation, 0=no violation), for i,j in {0,1}.
        n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))

        n0 = n00 + n01
        n1 = n10 + n11

        if n0 == 0 or n1 == 0:
            # No transitions observed out of one of the states -- test undefined.
            return float("nan"), float("nan")

        pi0_hat = n01 / n0
        pi1_hat = n11 / n1
        pi_hat = (n01 + n11) / (n0 + n1)

        def safe_term(count, prob):
            return count * np.log(prob) if count > 0 else 0.0

        likelihood_ratio = -2 * (
            safe_term(n00 + n10, 1 - pi_hat) + safe_term(n01 + n11, pi_hat)
            - safe_term(n00, 1 - pi0_hat) - safe_term(n01, pi0_hat)
            - safe_term(n10, 1 - pi1_hat) - safe_term(n11, pi1_hat)
        )

        p_value = 1 - stats.chi2.cdf(likelihood_ratio, df=1)
        return float(likelihood_ratio), float(p_value)

    # --- The actual backtest: walk-forward over real historical data, no lookahead ---

    def walk_forward_backtest(
        self,
        log_returns: pd.Series,
        P0_series: pd.Series,
        alpha: float,
        burn_in: int = 100,
        lambda_: float = LAMBDA_
    ) -> dict[str, npt.NDArray]:
        """
        Walk-forward backtesting of 1-day parametric VaR forecasts using EWMA volatility.
        At each day t, only data up to and including t is used to forecast day t+1 --
        no lookahead. Returns violations (0/1 series), var_series, and dates.
        """
        n = len(log_returns)
        violations = []
        var_series = []

        for t in range(burn_in, n - 1):
            history = log_returns.iloc[:t + 1]

            mu_annual = history.mean() * self.DAILY
            sigma_annual = self.ewma_volatility(history, lambda_=lambda_)

            P0 = P0_series.iloc[t + 1]  # # price known when the forecast is made, accounts for log return at t being price[t+1] / price[t]
            delta_t = 1 / self.DAILY  # 1-day horizon
            z_alpha = stats.norm.ppf(alpha)

            adjusted_mu = delta_t * (mu_annual - 0.5 * sigma_annual ** 2)
            adjusted_sigma = sigma_annual * np.sqrt(delta_t)
            var = -P0 * (adjusted_mu - adjusted_sigma * z_alpha)

            # actual realized loss on the next observed price
            P1 = P0_series.iloc[t + 2]
            realized_loss = P0 - P1

            violations.append(int(realized_loss > var))
            var_series.append(var)

        return {
            "violations": np.array(violations, dtype=int),
            "var_series": np.array(var_series),
            "dates": log_returns.index[burn_in + 1:n]
        }


# ----------------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------------
backtesting_eng = BacktestingEngine()
tickers = ["SPY", "GLD", "NVDA", "GOOGL", "BTC-USD"]
data = get_multiple_stocks_data(tickers, "2016-06-01", "2026-06-01", "1d")

prices = {}
log_returns = {}
paths_by_ticker = {}
backtest_results = {} # walk-forward backtest: backtest_results[ticker][f"conf_{conf}"]

for ticker in tickers:
    curr_data = data[ticker]
    prices[ticker] = curr_data["Close"]

    lr = pd.Series(
        np.diff(np.log(prices[ticker].values)),
        index=prices[ticker].index[1:]
    )
    log_returns[ticker] = lr

    # --- Backtest: ONE walk-forward result per (ticker, alpha) ---
    backtest_results[ticker] = {}
    for alpha in backtesting_eng.ALPHAS:
        conf = backtesting_eng.get_conf(alpha)
        result = backtesting_eng.walk_forward_backtest(
            log_returns[ticker], prices[ticker], alpha=alpha, burn_in=100
        )

        kupiec_lr, kupiec_p = backtesting_eng.kupiec_pof_test(
            violations=int(result["violations"].sum()),
            observations=len(result["violations"]),
            alpha=alpha
        )
        christoffersen_lr, christoffersen_p = backtesting_eng.christoffersen_independence_test(
            result["violations"]
        )

        n_obs = len(result["violations"])
        n_viol = int(result["violations"].sum())

        backtest_results[ticker][f"conf_{conf}"] = {
            "n_obs": n_obs,
            "violations": n_viol,
            "violation_rate": n_viol / n_obs,
            "target_rate": 1 - alpha,
            "kupiec_lr": kupiec_lr,
            "kupiec_p": kupiec_p,
            "christoffersen_lr": christoffersen_lr,
            "christoffersen_p": christoffersen_p,
        }

# ----------------------------------------------------------------------------
# Table 1: Walk-forward backtest results (Kupiec POF + Christoffersen independence)
# ----------------------------------------------------------------------------
bt_rows = []
for ticker in tickers:
    for conf_key, stats_dict in backtest_results[ticker].items():
        conf = conf_key.replace("conf_", "")
        bt_rows.append({
            "Ticker": ticker,
            "Confidence": f"{conf}%",
            "Observations": stats_dict["n_obs"],
            "Violations": stats_dict["violations"],
            "Violation Rate": stats_dict["violation_rate"],
            "Target Rate": stats_dict["target_rate"],
            "Kupiec LR": stats_dict["kupiec_lr"],
            "Kupiec p-value": stats_dict["kupiec_p"],
            "Christoffersen LR": stats_dict["christoffersen_lr"],
            "Christoffersen p-value": stats_dict["christoffersen_p"],
        })

backtest_table = pd.DataFrame(bt_rows).sort_values(["Ticker", "Confidence"]).reset_index(drop=True)


def highlight_reject(val):
    """Flags p-values below 0.05 (reject H0 at 5% significance) in red; pass in blue."""
    if pd.isna(val):
        return "background-color: #888888; color: white;"
    return "background-color: #b40426; color: white;" if val < 0.05 else "background-color: #3b4cc0; color: white;"


styled_backtest = (
    backtest_table.style
    .format({
        "Violation Rate": "{:.4f}",
        "Target Rate": "{:.4f}",
        "Kupiec LR": "{:.4f}",
        "Kupiec p-value": "{:.4f}",
        "Christoffersen LR": "{:.4f}",
        "Christoffersen p-value": "{:.4f}",
    })
    .map(highlight_reject, subset=["Kupiec p-value", "Christoffersen p-value"])
    .set_properties(**{"text-align": "center"})
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center"), ("background-color", "#2c3e50"),
                                       ("color", "white"), ("font-weight", "bold"), ("padding", "6px 10px")]},
        {"selector": "td", "props": [("padding", "6px 10px")]},
        {"selector": "table", "props": [("border-collapse", "collapse"),
                                          ("font-family", "Arial, sans-serif"), ("font-size", "13px")]},
        {"selector": "caption", "props": [("font-size", "14px"), ("font-weight", "bold"), ("padding", "8px 0")]},
    ])
    .set_caption("Walk-Forward Backtest: Kupiec POF & Christoffersen Independence Tests (red = reject H0 at 5%)")
)

# In a notebook: just put `styled_comparison` and `styled_backtest` as the last
# expression in their own cells to render. Kept as separate tables deliberately --
# the point-estimate table answers "what is my risk right now," the backtest
# table answers "does my model's track record hold up historically." Don't
# merge them; they measure different things.
