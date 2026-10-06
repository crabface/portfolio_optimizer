"""Mean-variance optimization with cvxpy -- implement these yourself.

Notation: w = weights (n,), mu = expected returns (n,), Sigma = covariance (n, n).
Portfolio return = mu @ w, variance = w @ Sigma @ w (use cp.quad_form).
Long-only, fully invested: sum(w) == 1, w >= 0.
"""
import cvxpy as cp
import numpy as np


def min_volatility(mu, Sigma, max_weight=1.0):
    """minimize  w' Sigma w   s.t. sum(w) = 1, 0 <= w <= max_weight.

    Returns weights as a numpy array.
    """
    raise NotImplementedError


def efficient_return(mu, Sigma, target_return, max_weight=1.0):
    """Min variance subject to mu @ w >= target_return (plus the usual constraints).

    Sweeping target_return from the min-vol return to max(mu) traces the frontier.
    """
    raise NotImplementedError


def efficient_frontier(mu, Sigma, n_points=50, max_weight=1.0):
    """Return arrays (vols, rets) along the frontier using efficient_return."""
    raise NotImplementedError


def max_sharpe(mu, Sigma, rf=0.0, max_weight=1.0):
    """Maximize (mu @ w - rf) / sqrt(w' Sigma w).

    That ratio isn't convex, so you can't hand it to cvxpy directly. Standard trick
    (Cornuejols & Tutuncu): substitute y = k * w with k > 0, fix the excess return to 1:

        minimize  y' Sigma y
        s.t.      (mu - rf) @ y == 1
                  sum(y) == k,  y >= 0,  k >= 0
                  (max weight: y <= max_weight * k)

    then recover w = y / k. Think about why this is equivalent.
    """
    raise NotImplementedError


def portfolio_performance(w, mu, Sigma, rf=0.0):
    """Return (expected_return, volatility, sharpe)."""
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ Sigma @ w))
    return ret, vol, (ret - rf) / vol
