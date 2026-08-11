# EWMA — Volatility Dynamics
import numpy as np
import pandas as pd
import numpy.typing as npt
import scipy.stats as stats

from data.data import get_multiple_stocks_data

# reuseable class for GBM simulation and risk metrics calculation
class BacktestingEngine:
    ROLLING_PERIODS = (20, 60)
    TIME_HORIZONS = (1, 10)
    ALPHAS = (0.95, 0.99) # the confidence levels for VaR and ES calculations
    DAILY = 252
    N_PATHS = 10_000
    LAMBDA_ = 0.94
    T = 1.0  # Time to maturity in years for the simulation
    N = 252  # Number of time steps in the simulation (daily steps for 1 year)

    def get_conf(self, alpha: float) -> int:
        if alpha not in self.ALPHAS:
            raise ValueError(f"Alpha must be one of {self.ALPHAS}")
        return int(alpha * 100)

    def ewma_volatility(self, log_returns: pd.Series, lambda_: float = LAMBDA_) -> float:
        if (not 0 < lambda_ <= 1):
            raise ValueError("Lambda must be between 0 and 1")

        squared_returns = log_returns ** 2
        last_return = squared_returns.iloc[-1]
        squared_returns = squared_returns.shift(1)
        squared_returns = squared_returns.fillna(value=log_returns.mean() ** 2)
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

    def backtested_var_es(
        self,
        paths: npt.NDArray[np.float64],
        time_horizon: int,
        alpha: float
    ) -> tuple[float, float, dict[str, float]]:
        if time_horizon < 1 or time_horizon >= paths.shape[1]:
            raise ValueError("time_horizon must be between 1 and number of simulated steps")

        P0 = paths[0, 0]
        losses = P0 - paths[:, time_horizon]

        var = np.percentile(losses, alpha * 100)
        es = losses[losses > var].mean()

        kupiec_lr_stat, kupiec_p_value, christoffersen_lr_stat, christoffersen_p_value = self.backtester(losses, var, alpha)
        backtest_results = {
            "kupiec_lr_stat": float(kupiec_lr_stat),
            "kupiec_p_value": float(kupiec_p_value),
            "christoffersen_lr_stat": float(christoffersen_lr_stat),
            "christoffersen_p_value": float(christoffersen_p_value)
        }

        return float(var), float(es), backtest_results

    def backtested_var_es_parametric(
        self,
        paths: npt.NDArray[np.float64],
        mu: float,
        sigma: float,
        T: float,
        N: int,
        time_horizon: int,
        alpha: float
    ) -> tuple[float, float, dict[str, float]]:
        P0 = paths[0, 0]
        delta_t = (T / N) * time_horizon
        z_alpha = stats.norm.ppf(alpha)
        adjusted_mu = delta_t * (mu - 0.5 * sigma ** 2)
        adjusted_sigma = sigma * np.sqrt(delta_t)
        z_star_alpha = stats.norm.pdf(z_alpha) / (1 - alpha)

        var = -P0 * (adjusted_mu - adjusted_sigma * z_alpha)
        es = -P0 * (adjusted_mu - adjusted_sigma * z_star_alpha)

        losses = P0 - paths[:, time_horizon]

        kupiec_lr_stat, kupiec_p_value, christoffersen_lr_stat, christoffersen_p_value = self.backtester(losses, var, alpha)
        backtest_results = {
            "kupiec_lr_stat": float(kupiec_lr_stat),
            "kupiec_p_value": float(kupiec_p_value),
            "christoffersen_lr_stat": float(christoffersen_lr_stat),
            "christoffersen_p_value": float(christoffersen_p_value)
        }
        return float(var), float(es), backtest_results

    def kupiec_pof_test(self, violations, observations, alpha):
        if observations <= 0:
            raise ValueError("Number of observations must be positive")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")

        non_violations = observations - violations # n - n_1, i.e. n_0, n is # of observations, n_1 is # of violations, n_0 is # of non-violations
        p_hat = violations / observations

        if p_hat == 0 or p_hat == 1:
        # log(0) is undefined; the LR statistic is degenerate at the boundary.
        # By convention, treat the null-model likelihood ratio as the limiting case.
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

    def christoffersen_independence_test(self, violations: npt.NDArray[np.int64]) -> tuple[float, float]:
        if len(violations) < 2:
            raise ValueError("At least two observations are required for the Christoffersen test.")

        # nij denotes the number of days where I_{t-1}=i and I_t=j (where I_t is the indicator variable for a violation, 1=violation, 0=no violation), for i,j in {0,1}, e.g. n01 is the number of days where there was no violation the previous day and a violation today. Note n_{i0}+n_{i1} is the total number of days following a day in state i, which we will denote n_i.
        n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))

        n0 = n00 + n01
        n1 = n10 + n11

        if n0 == 0 or n1 == 0:
        # No transitions were ever observed out of one of the states
        # (e.g. never two consecutive violations) — test is undefined; skip it.
            return float("nan"), float("nan")

        pi0_hat = n01 / n0 if n0 > 0 else 0
        pi1_hat = n11 / n1 if n1 > 0 else 0
        pi_hat = (n01 + n11) / (n0 + n1)

        def safe_term(count, prob):
            return count * np.log(prob) if count > 0 else 0.0

        likelihood_ratio = -2 * (
            safe_term(n00 + n10, 1 - pi_hat) + safe_term(n01 + n11, pi_hat)
            - safe_term(n00, 1 - pi0_hat) - safe_term(n01, pi0_hat)
            - safe_term(n10, 1 - pi1_hat) - safe_term(n11, pi1_hat)
        )

        # P-value based on Chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(likelihood_ratio, df=1)

        return float(likelihood_ratio), float(p_value)

    def backtester(self, losses: npt.NDArray[np.float64], var: float, alpha: float) -> tuple[float, float, float, float]:
        violations = np.sum(losses > var)
        observations = len(losses)

        kupiec_lr_stat, kupiec_p_value = self.kupiec_pof_test(violations, observations, alpha)
        christoffersen_lr_stat, christoffersen_p_value = self.christoffersen_independence_test(np.array(losses > var, dtype=int))

        return float(kupiec_lr_stat), float(kupiec_p_value), float(christoffersen_lr_stat), float(christoffersen_p_value)

# initialize the risk engine and variables to store data
backtesting_eng = BacktestingEngine()

tickers = ["SPY", "GLD", "NVDA", "GOOGL", "BTC-USD"]
data = get_multiple_stocks_data(tickers, "2016-06-01", "2026-06-01", "1d")

prices = {}
log_returns = {}
paths_by_ticker = {}
results = {}

# populate the dictionaries with prices, log returns, simulated paths, and risk metrics
for ticker in tickers:
    curr_data = data[ticker]
    prices[ticker] = curr_data["Close"]

    lr = pd.Series(
        np.diff(np.log(prices[ticker].values)),
        index=prices[ticker].index[1:]
    )
    log_returns[ticker] = lr

    mu_annual = lr.mean() * backtesting_eng.DAILY 

    sigma_annual = backtesting_eng.ewma_volatility(log_returns=lr)

    paths = backtesting_eng.simulate_gbm(
        S0=prices[ticker].iloc[-1],
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
        T=1.0,
        N=backtesting_eng.DAILY,
        n_paths=backtesting_eng.N_PATHS
    )

    paths_by_ticker[ticker] = paths
    solving_methods = ["monte_carlo", "parametric"]
    results[ticker] = {method: {} for method in solving_methods}
         
    for horizon in backtesting_eng.TIME_HORIZONS:
            horizon_key = f"{horizon}d"
            for method in solving_methods:
                results[ticker][method][horizon_key] = {}
    
            for alpha in backtesting_eng.ALPHAS:
                print(f"Ticker: {ticker}, Horizon: {horizon_key}, Alpha: {alpha}\n")
                var, es, backtest_results = backtesting_eng.backtested_var_es(paths, horizon, alpha)
                print("---------- PARAMETRIC ----------")
                para_var, para_es, para_backtest_results = backtesting_eng.backtested_var_es_parametric(paths, mu_annual, sigma_annual, backtesting_eng.T, backtesting_eng.N, horizon, alpha)
                conf = backtesting_eng.get_conf(alpha)

                results[ticker]["monte_carlo"][horizon_key][f"var_{conf}"] = var
                results[ticker]["monte_carlo"][horizon_key][f"es_{conf}"] = es
                results[ticker]["parametric"][horizon_key][f"var_{conf}"] = para_var
                results[ticker]["parametric"][horizon_key][f"es_{conf}"] = para_es

                # backtesting the results using Kupiec + Christoffersen tests
                results[ticker]["monte_carlo"][horizon_key][f"kupiec_lr_stat_{conf}"] = backtest_results["kupiec_lr_stat"]
                results[ticker]["monte_carlo"][horizon_key][f"kupiec_p_value_{conf}"] = backtest_results["kupiec_p_value"]
                results[ticker]["parametric"][horizon_key][f"kupiec_lr_stat_{conf}"] = para_backtest_results["kupiec_lr_stat"]
                results[ticker]["parametric"][horizon_key][f"kupiec_p_value_{conf}"] = para_backtest_results["kupiec_p_value"]

                results[ticker]["monte_carlo"][horizon_key][f"christoffersen_lr_stat_{conf}"] = backtest_results["christoffersen_lr_stat"]
                results[ticker]["monte_carlo"][horizon_key][f"christoffersen_p_value_{conf}"] = backtest_results["christoffersen_p_value"]
                results[ticker]["parametric"][horizon_key][f"christoffersen_lr_stat_{conf}"] = para_backtest_results["christoffersen_lr_stat"]
                results[ticker]["parametric"][horizon_key][f"christoffersen_p_value_{conf}"] = para_backtest_results["kupiec_p_value"]

# TODO: change this for historical results


# ------------------------------------------------------------------
# 1. Flatten the nested `results` dict into a tidy long-format table
# ------------------------------------------------------------------
metric_labels = {
    "var_95": "VaR 95%",
    "es_95": "ES 95%",
    "var_99": "VaR 99%",
    "es_99": "ES 99%",
}
metrics_order = ["var_95", "es_95", "var_99", "es_99"]

rows = []
for ticker in tickers:
    for horizon_key in results[ticker]["monte_carlo"]:
        for metric in metrics_order:
            mc_val = results[ticker]["monte_carlo"][horizon_key][metric]
            para_val = results[ticker]["parametric"][horizon_key][metric]
            rows.append({
                "Ticker": ticker,
                "Horizon": horizon_key,
                "Metric": metric_labels[metric],
                "Monte Carlo": mc_val,
                "Parametric": para_val,
                "Difference (MC - CF)": mc_val - para_val,
            })

comparison_table = pd.DataFrame(rows)

# ------------------------------------------------------------------
# 2. Enforce sensible ordering (numeric horizon, fixed metric order)
# ------------------------------------------------------------------
horizon_order = sorted(
    comparison_table["Horizon"].unique(),
    key=lambda x: int(x.replace("d", ""))
)
metric_order = [metric_labels[m] for m in metrics_order]

comparison_table["Horizon"] = pd.Categorical(
    comparison_table["Horizon"], categories=horizon_order, ordered=True
)
comparison_table["Metric"] = pd.Categorical(
    comparison_table["Metric"], categories=metric_order, ordered=True
)

comparison_table = comparison_table.sort_values(
    ["Ticker", "Horizon", "Metric"]
).reset_index(drop=True)

comparison_grouped = comparison_table.set_index(["Ticker", "Horizon", "Metric"])

# ------------------------------------------------------------------
# 3. Detect group boundaries so we know where to draw borders
# ------------------------------------------------------------------
idx = comparison_grouped.index
ticker_vals = idx.get_level_values("Ticker").to_numpy()
horizon_vals = idx.get_level_values("Horizon").to_numpy()
n = len(idx)

new_ticker = np.zeros(n, dtype=bool)   # True where a new Ticker group starts
new_horizon = np.zeros(n, dtype=bool)  # True where a new Horizon sub-group starts
for i in range(1, n):
    new_ticker[i] = ticker_vals[i] != ticker_vals[i - 1]
    new_horizon[i] = new_ticker[i] or (horizon_vals[i] != horizon_vals[i - 1])

TICKER_BORDER = "border-top: 3px solid #4695f0;"   # thick dark line between tickers
HORIZON_BORDER = "border-top: 1px solid #b0b7bd;"  # thin gray line between horizons

def body_border_func():
    """Applies the right border to every data cell in a row, based on group boundaries."""
    counter = {"i": 0}
    def func(row):
        i = counter["i"]
        counter["i"] += 1
        css = TICKER_BORDER if new_ticker[i] else (HORIZON_BORDER if new_horizon[i] else "")
        return [css] * len(row)
    return func

def index_border_func(level_changed_array, border_css):
    """Same border logic, applied to the sparse index cells (Ticker/Horizon columns)."""
    def func(s):
        return [border_css if level_changed_array[i] else "" for i in range(len(s))]
    return func

# ------------------------------------------------------------------
# 4. Style: number formatting, diff color gradient, header, borders,
#    and a hover effect that also darkens the gradient-colored cells
# ------------------------------------------------------------------
CMAP = "coolwarm"

diff_vals = comparison_grouped["Difference (MC - CF)"]

styled = (
    comparison_grouped.style
    .format({
        "Monte Carlo": "{:.4f}",
        "Parametric": "{:.4f}",
        "Difference (MC - CF)": "{:+.4f}",
    })
    .background_gradient(
            subset=["Difference (MC - CF)"],
            cmap=CMAP,          # note: NOT coolwarm_r, see below
            gmap=diff_vals.abs(),     # color driven by |difference|, not the signed value
            vmin=0,                  # zero deviation -> low end of cmap
            vmax=diff_vals.abs().max()  # biggest deviation -> high end
        )
    .set_properties(**{"text-align": "center"})
    .apply(body_border_func(), axis=1)
    .apply_index(index_border_func(new_horizon, HORIZON_BORDER), axis=0, level=1)
    .apply_index(index_border_func(new_horizon, HORIZON_BORDER), axis=0, level=2)
    .apply_index(index_border_func(new_ticker, TICKER_BORDER), axis=0, level=0)
    .apply_index(index_border_func(new_ticker, TICKER_BORDER), axis=0, level=1)
    .apply_index(index_border_func(new_ticker, TICKER_BORDER), axis=0, level=2)
    .set_table_styles([
        {"selector": "th", "props": [
            ("text-align", "center"),
            ("background-color", "#2c3e50"),
            ("color", "white"),
            ("font-weight", "bold"),
            ("padding", "6px 10px"),
        ]},
        {"selector": "td", "props": [("padding", "6px 10px")]},
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("font-family", "Arial, sans-serif"),
            ("font-size", "13px"),
        ]},
        # Hover for plain (non-gradient) cells: simple background tint.
        {"selector": "tbody tr", "props": [("transition", "background-color 0.3s ease")]},
        {"selector": "tbody tr:hover", "props": [("background-color", "#494a4b")]},
        # Hover overlay for gradient cells: an inset box-shadow layers a
        # translucent dark tint ON TOP of the inline background-color that
        # background_gradient() sets, since background-color itself can't
        # be reliably overridden by a plain CSS hover rule once it's inline.
        {"selector": "tbody tr td", "props": [("transition", "box-shadow 0.3s ease")]},
        {"selector": "tbody tr:hover td", "props": [
            ("box-shadow", "inset 0 0 0 9999px rgba(0,0,0,0.12)")
        ]},
        {"selector": "caption", "props": [
                    ("font-size", "14px"),
                    ("font-weight", "bold"),
                    ("padding", "8px 0"),
                ]},
    ], overwrite=False)  # overwrite=False keeps this appended, not replacing prior rules
    .set_caption("VaR / ES Comparison — Monte Carlo vs Parametric/Closed-Form")
)

import webbrowser
import os

html = styled.to_html()  # renders the full styled table to an HTML string

output_path = os.path.abspath("scrap/4.html")
with open(output_path, "w") as f:
    f.write(html)

# webbrowser.open(f"file://{output_path}")  # auto-opens it in your default browser
        