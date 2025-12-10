from __future__ import annotations

import numpy as np

from .hedging.bs_mc_simple_delta_hedger import BsMcSimpleDeltaHedger, ProductType
from .pricing import OptionType
from .stats.histogram_calculator import HistogramCalculator, MeanVarCalculator


def bs_delta_hedge(
    *,
    product_type: ProductType,
    payoff_type: OptionType,
    strike: float,
    expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
    spot: float,
    sim_vol: float,
    mark_vol: float,
    n_steps: int = 252,
    hedge_freq: int = 252,
    n_paths: int = 100_000,
    hist_min: float = -5.0,
    hist_max: float = 5.0,
    n_bins: int = 50,
    seed: int | None = None,
) -> dict[str, object]:
    """Run a Black-Scholes Monte Carlo delta-hedging experiment.

    The function constructs the appropriate hedging engine and statistics
    calculators, performs the simulation, and returns summary statistics
    for the P&L distribution of a delta-hedged long option position.

    Parameters
    ----------
    product_type : {"european", "digital"}
        Type of option payoff.
    payoff_type : {"call", "put"}
        Direction of the payoff.
    strike : float
        Strike price of the option.
    expiry : float
        Time to maturity in years.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float, default=0.0
        Continuous dividend or convenience yield.
    spot : float
        Initial underlying price.
    sim_vol : float
        Volatility used to simulate the underlying paths.
    mark_vol : float
        Volatility used for Black-Scholes pricing and delta calculations.
    n_steps : int, default=252
        Base number of time steps per year for the simulation grid.
    hedge_freq : int, default=252
        Number of hedge rebalancings per year.
    n_paths : int, default=100_000
        Number of Monte Carlo paths.
    hist_min : float, default=-5.0
        Lower bound of the P&L histogram range.
    hist_max : float, default=5.0
        Upper bound of the P&L histogram range.
    n_bins : int, default=50
        Number of histogram bins.
    seed : int, optional
        Random number generator seed.

    Returns
    -------
    dict
        Dictionary with keys ``"pnl_mean"``, ``"pnl_std"``,
        ``"hist_bin_edges"``, and ``"hist_counts"``.
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if hist_max <= hist_min:
        raise ValueError("hist_max must be greater than hist_min.")

    hedger = BsMcSimpleDeltaHedger(
        product_type=product_type,
        payoff_type=payoff_type,
        strike=strike,
        expiry=expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        spot=spot,
        sim_vol=sim_vol,
        mark_vol=mark_vol,
        n_steps=n_steps,
        seed=seed,
    )

    mean_var = MeanVarCalculator(n_vars=1)
    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1, dtype=float)
    histogram = HistogramCalculator(bin_edges=bin_edges, n_vars=1)

    hedger.hedge(stats_calcs=[mean_var, histogram], hedge_freq=hedge_freq, n_paths=n_paths)

    mv_results = mean_var.results()
    counts, edges = histogram.results()

    mean = float(mv_results["mean"][0])
    std = float(mv_results["std"][0])
    hist_counts = counts[:, 0].astype(int)

    return {
        "pnl_mean": mean,
        "pnl_std": std,
        "hist_bin_edges": edges,
        "hist_counts": hist_counts,
    }
