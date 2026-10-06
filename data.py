"""Price loading and return/covariance estimation."""
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def get_prices(tickers, start, end):
    """Adjusted close prices, one column per ticker."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices[tickers].dropna()


def annualized_stats(prices):
    """Return (mu, Sigma): annualized mean returns and sample covariance.

    Uses simple daily returns, matching PyPortfolioOpt's defaults
    (mean_historical_return uses compounding, so expect small mu differences
    there -- that's one of the things worth investigating).
    """
    returns = prices.pct_change().dropna()
    mu = returns.mean() * TRADING_DAYS
    Sigma = returns.cov() * TRADING_DAYS
    return mu, Sigma
