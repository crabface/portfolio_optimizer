"""Compare the cvxpy implementation against PyPortfolioOpt on the same inputs."""
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

import optimizer as opt
from data import annualized_stats, get_prices

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "GLD"]
START, END = "2019-01-01", "2024-12-31"
RF = 0.0


def compare(name, mine, theirs, mu, Sigma):
    df = pd.DataFrame({"mine": mine, "pypfopt": theirs}, index=mu.index).round(4)
    df["diff"] = df["mine"] - df["pypfopt"]
    print(f"\n== {name} ==")
    print(df)
    for label, w in [("mine", mine), ("pypfopt", theirs)]:
        r, v, s = opt.portfolio_performance(np.asarray(w), mu.values, Sigma.values, RF)
        print(f"{label:>8}: return {r:.4f}  vol {v:.4f}  sharpe {s:.4f}")


if __name__ == "__main__":
    prices = get_prices(TICKERS, START, END)
    # Feed both sides the *same* mu and Sigma, so any difference is in the optimizer.
    mu, Sigma = annualized_stats(prices)

    ef = EfficientFrontier(mu, Sigma)
    ef.min_volatility()
    compare("min volatility", opt.min_volatility(mu.values, Sigma.values),
            pd.Series(ef.clean_weights()), mu, Sigma)

    ef = EfficientFrontier(mu, Sigma)
    ef.max_sharpe(risk_free_rate=RF)
    compare("max sharpe", opt.max_sharpe(mu.values, Sigma.values, RF),
            pd.Series(ef.clean_weights()), mu, Sigma)
