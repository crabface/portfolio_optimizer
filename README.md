# Portfolio Optimizer

Mean-variance portfolio optimization built from scratch with [cvxpy](https://www.cvxpy.org/), and checked against [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/).

Enter a set of tickers and a date range. The app estimates expected returns and the covariance matrix from historical prices, then traces the efficient frontier. It also finds the **maximum-Sharpe** and **minimum-volatility** portfolios.

## Status

- [x] Price data (yfinance, adjusted closes) and annualized return/covariance estimates
- [x] Verification script comparing results against PyPortfolioOpt
- [ ] Minimum-volatility portfolio (cvxpy)
- [ ] Efficient frontier
- [ ] Maximum-Sharpe portfolio (via the Cornuéjols–Tütüncu reformulation)
- [ ] Streamlit frontend
- [ ] Stretch goals: Ledoit-Wolf covariance shrinkage, per-asset weight caps, risk parity, Black-Litterman

## Project layout

| File | Purpose |
|---|---|
| `data.py` | Downloads prices; computes annualized mean returns and covariance |
| `optimizer.py` | Optimization routines written with cvxpy |
| `verify.py` | Runs both implementations on identical inputs and prints the weight differences |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python verify.py
```

## The approach

Long-only, fully invested portfolios: weights `w` with `sum(w) = 1` and `w >= 0`.

- **Minimum volatility:** minimize `w' Σ w`.
- **Efficient frontier:** minimize `w' Σ w` subject to `μ' w >= target`, sweeping the target return.
- **Maximum Sharpe:** the Sharpe ratio is not convex. Substitute `y = k·w` and fix the excess return to 1, which turns the problem into a convex quadratic program.
