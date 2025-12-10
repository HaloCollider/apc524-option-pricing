# Option Pricing

A Python package for pricing and analyzing plain-vanilla options under the Black-Scholes-Merton framework. The package provides closed-form pricing and Greeks computation for European and digital options, Monte Carlo path simulation, and a delta-hedging engine for backtesting hedging strategies and studying P&L distributions.

This project was developed as the final project for APC524 (Software Engineering for Scientific Computing) at Princeton University.

## Features

- **Black-Scholes pricing**: Compute closed-form prices for European call and put options using the Black-Scholes-Merton formula, with support for continuous dividend yields.
- **Greeks computation**: Calculate all primary Greeks (delta, gamma, vega, theta, rho) in a single vectorized pass for efficient sensitivity analysis.
- **Digital options**: Price and hedge cash-or-nothing digital calls and puts with analytical formulas.
- **Monte Carlo simulation**: Simulate geometric Brownian motion (GBM) paths with configurable time steps and random seeds for reproducibility.
- **Delta hedging engine**: Run Monte Carlo delta-hedging experiments to analyze the P&L distribution of a hedged option position under various market conditions, including volatility misspecification scenarios.

## Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/HaloCollider/apc524-option-pricing.git
cd apc524-option-pricing
pip install -e .
```

### Dependencies

- Python >= 3.13
- NumPy >= 1.26
- SciPy >= 1.11
- Pandas >= 2.0
- Matplotlib >= 3.5

## Usage

### Pricing a European Option

Use `black_scholes_price` to compute the fair value of a European call or put:

```python
from option_pricing import black_scholes_price

price = black_scholes_price(
    spot=100,           # current stock price
    strike=100,         # strike price
    maturity=1.0,       # time to expiry in years
    rate=0.05,          # risk-free interest rate
    volatility=0.2,     # annualized volatility
    option_type="call", # "call" or "put"
)
print(f"Option price: {price:.4f}")
```

### Computing Greeks

Use `black_scholes_greeks` to obtain delta, gamma, vega, theta, and rho:

```python
from option_pricing import black_scholes_greeks

greeks = black_scholes_greeks(
    spot=100,
    strike=100,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    option_type="call",
)
delta, gamma, vega, theta, rho = greeks
print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}")
```

### Running a Delta Hedging Simulation

The `bs_delta_hedge` function runs a Monte Carlo delta-hedging experiment. It simulates underlying price paths, rebalances the hedge at a specified frequency, and returns summary statistics of the hedging P&L:

```python
from option_pricing import bs_delta_hedge

result = bs_delta_hedge(
    product_type="european",  # "european" or "digital"
    payoff_type="call",       # "call" or "put"
    strike=100,
    expiry=1.0,
    rate=0.05,
    spot=100,
    sim_vol=0.2,              # volatility used to simulate paths
    mark_vol=0.2,             # volatility used for pricing and hedging
    n_paths=100_000,          # number of Monte Carlo paths
    hedge_freq=252,           # rebalancing frequency (252 = daily)
    seed=42,                  # for reproducibility
)

print(f"P&L Mean: {result['pnl_mean']:.4f}")
print(f"P&L Std:  {result['pnl_std']:.4f}")
```

The returned dictionary also includes histogram data (`hist_bin_edges` and `hist_counts`) for visualizing the P&L distribution.

## Project Structure

```
├── src/option_pricing/       # Main package
│   ├── pricing.py            # Black-Scholes pricing and Greeks
│   ├── api.py                # High-level delta hedging API
│   ├── hedging/              # Delta hedging engine
│   └── stats/                # Statistics utilities (histograms, moments)
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks with experiments
└── docs/                     # Sphinx documentation
```

## License

MIT
