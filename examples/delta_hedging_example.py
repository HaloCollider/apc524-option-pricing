#!/usr/bin/env python3
"""
Delta Hedging Example
=====================

This script demonstrates the basic functionality of the option_pricing package
by running a delta-hedging simulation and visualizing the P&L distribution.

To run this example:
    python examples/delta_hedging_example.py

With custom parameters:
    python examples/delta_hedging_example.py --spot 100 --strike 100 --expiry 1.0 \
        --rate 0.05 --volatility 0.2 --n_paths 10000 --hedge_freq 252 --seed 42

The script will:
1. Price a European call option using Black-Scholes
2. Compute and display the option's Greeks
3. Run a Monte Carlo delta-hedging simulation
4. Plot the resulting P&L distribution histogram
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from option_pricing import (
    black_scholes_greeks,
    black_scholes_price,
    bs_delta_hedge,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delta hedging simulation example for option_pricing package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spot", type=float, default=100.0, help="Current stock price")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price")
    parser.add_argument("--expiry", type=float, default=1.0, help="Time to expiry in years")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free interest rate")
    parser.add_argument("--volatility", type=float, default=0.2, help="Annualized volatility")
    parser.add_argument("--n_paths", type=int, default=10_000, help="Number of Monte Carlo paths")
    parser.add_argument("--hedge_freq", type=int, default=252, help="Rebalancing frequency per year")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="examples/delta_hedging_pnl.png", help="Output file for histogram")
    parser.add_argument("--no-plot", action="store_true", help="Skip displaying the plot (still saves to file)")
    return parser.parse_args()


def main():
    args = parse_args()

    spot = args.spot
    strike = args.strike
    expiry = args.expiry
    rate = args.rate
    volatility = args.volatility
    n_paths = args.n_paths
    hedge_freq = args.hedge_freq
    seed = args.seed

    # ==========================================================================
    # Part 1: Black-Scholes Pricing
    # ==========================================================================
    print("=" * 60)
    print("Black-Scholes Option Pricing")
    print("=" * 60)

    call_price = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=expiry,
        rate=rate,
        volatility=volatility,
        option_type="call",
    )
    put_price = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=expiry,
        rate=rate,
        volatility=volatility,
        option_type="put",
    )

    print(f"Spot:       {spot:.2f}")
    print(f"Strike:     {strike:.2f}")
    print(f"Expiry:     {expiry:.2f} years")
    print(f"Rate:       {rate:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print()
    print(f"Call Price: {float(call_price):.4f}")
    print(f"Put Price:  {float(put_price):.4f}")

    # ==========================================================================
    # Part 2: Greeks Computation
    # ==========================================================================
    print()
    print("=" * 60)
    print("Greeks (Call Option)")
    print("=" * 60)

    greeks = black_scholes_greeks(
        spot=spot,
        strike=strike,
        maturity=expiry,
        rate=rate,
        volatility=volatility,
        option_type="call",
    )
    delta, gamma, vega, theta, rho = greeks

    print(f"Delta: {float(delta):+.4f}  (sensitivity to spot)")
    print(f"Gamma: {float(gamma):+.4f}  (sensitivity of delta to spot)")
    print(f"Vega:  {float(vega):+.4f}  (sensitivity to volatility)")
    print(f"Theta: {float(theta):+.4f}  (time decay per year)")
    print(f"Rho:   {float(rho):+.4f}  (sensitivity to interest rate)")

    # ==========================================================================
    # Part 3: Delta Hedging Simulation
    # ==========================================================================
    print()
    print("=" * 60)
    print("Delta Hedging Simulation")
    print("=" * 60)
    print(f"Running {n_paths:,} Monte Carlo paths with {hedge_freq}x/year rebalancing...")
    print()

    result = bs_delta_hedge(
        product_type="european",
        payoff_type="call",
        strike=strike,
        expiry=expiry,
        rate=rate,
        spot=spot,
        sim_vol=volatility,   # Simulated volatility = mark volatility (no misspec)
        mark_vol=volatility,
        n_paths=n_paths,
        hedge_freq=hedge_freq,
        n_bins=50,
        hist_min=-3.0,
        hist_max=3.0,
        seed=seed,
    )

    print(f"P&L Mean:     {result['pnl_mean']:+.4f}")
    print(f"P&L Std Dev:  {result['pnl_std']:.4f}")

    # ==========================================================================
    # Part 4: Visualization
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    edges = result["hist_bin_edges"]
    counts = result["hist_counts"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax.bar(centers, counts, width=width * 0.9, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(result["pnl_mean"], color="red", linestyle="--", linewidth=2, label=f"Mean = {result['pnl_mean']:.3f}")
    ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

    ax.set_xlabel("P&L (normalized)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Delta-Hedged European Call P&L Distribution\n"
        f"(S={spot}, K={strike}, σ={volatility:.0%}, {n_paths:,} paths, {hedge_freq}x/year hedging)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print()
    print(f"Histogram saved to: {args.output}")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()

