#!/usr/bin/env python3
"""
Delta Hedging Animation
=======================

This script creates an animated visualization of the delta hedging process,
showing how a hedged portfolio evolves over time as the underlying price moves.

To run this example:
    python examples/delta_hedging_animation.py

With custom parameters:
    python examples/delta_hedging_animation.py --n_paths 5 --volatility 0.3 --fps 30

The animation shows:
- Multiple simulated stock price paths
- The option delta evolving with the stock price
- The cumulative P&L of the delta-hedged portfolio
- Real-time tracking of how the hedge performs
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from option_pricing import (
    black_scholes_greeks,
    black_scholes_price,
    simulate_gbm_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animated delta hedging visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spot", type=float, default=100.0, help="Initial stock price")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price")
    parser.add_argument("--expiry", type=float, default=1.0, help="Time to expiry in years")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--volatility", type=float, default=0.2, help="Volatility")
    parser.add_argument("--n_paths", type=int, default=5, help="Number of paths to animate")
    parser.add_argument("--n_steps", type=int, default=252, help="Number of time steps")
    parser.add_argument("--hedge_freq", type=int, default=21, help="Hedge rebalancing frequency (steps)")
    parser.add_argument("--fps", type=int, default=20, help="Animation frames per second")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Save animation to file (e.g., animation.mp4 or animation.gif)")
    return parser.parse_args()


def compute_hedge_pnl(path, times, strike, rate, volatility, hedge_freq_steps):
    """
    Compute the delta-hedged P&L at each time step for a single path.
    
    Returns arrays of: deltas, option_values, hedge_portfolio_values, pnl
    """
    n_steps = len(path) - 1
    dt = times[1] - times[0]
    expiry = times[-1]
    
    deltas = np.zeros(n_steps + 1)
    option_values = np.zeros(n_steps + 1)
    hedge_values = np.zeros(n_steps + 1)
    bank = np.zeros(n_steps + 1)
    
    # Initial position
    s0 = path[0]
    ttm0 = expiry
    price0 = float(black_scholes_price(s0, strike, ttm0, rate, volatility, "call"))
    greeks0 = black_scholes_greeks(s0, strike, ttm0, rate, volatility, "call")
    delta0 = float(greeks0[0])
    
    deltas[0] = delta0
    option_values[0] = price0
    bank[0] = price0 - delta0 * s0  # Cash from selling option, buying delta shares
    hedge_values[0] = delta0 * s0 + bank[0]
    
    current_delta = delta0
    current_bank = bank[0]
    last_hedge_step = 0
    
    for i in range(1, n_steps + 1):
        s = path[i]
        t = times[i]
        ttm = expiry - t
        
        # Bank accrues interest
        current_bank *= np.exp(rate * dt)
        
        if ttm > 1e-8:
            price = float(black_scholes_price(s, strike, ttm, rate, volatility, "call"))
            greeks = black_scholes_greeks(s, strike, ttm, rate, volatility, "call")
            new_delta = float(greeks[0])
        else:
            # At expiry
            price = max(s - strike, 0)
            new_delta = 1.0 if s > strike else 0.0
        
        option_values[i] = price
        
        # Rebalance hedge at specified frequency
        if (i - last_hedge_step) >= hedge_freq_steps and ttm > 1e-8:
            trade = new_delta - current_delta
            current_bank -= trade * s
            current_delta = new_delta
            last_hedge_step = i
        
        deltas[i] = current_delta
        bank[i] = current_bank
        hedge_values[i] = current_delta * s + current_bank
    
    # P&L = hedge portfolio value - option value (we are long hedge, short option)
    # For long option perspective: option_value - hedge_cost
    pnl = hedge_values - option_values
    
    return deltas, option_values, hedge_values, pnl


def main():
    args = parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    # Simulate paths
    print(f"Simulating {args.n_paths} price paths...")
    paths = simulate_gbm_paths(
        spot=args.spot,
        maturity=args.expiry,
        rate=args.rate,
        volatility=args.volatility,
        steps=args.n_steps,
        paths=args.n_paths,
        rng=rng,
    )
    
    times = np.linspace(0, args.expiry, args.n_steps + 1)
    
    # Compute hedge P&L for each path
    print("Computing delta hedges...")
    all_deltas = []
    all_option_values = []
    all_hedge_values = []
    all_pnls = []
    
    for i in range(args.n_paths):
        deltas, opt_vals, hedge_vals, pnl = compute_hedge_pnl(
            paths[i], times, args.strike, args.rate, args.volatility, args.hedge_freq
        )
        all_deltas.append(deltas)
        all_option_values.append(opt_vals)
        all_hedge_values.append(hedge_vals)
        all_pnls.append(pnl)
    
    all_deltas = np.array(all_deltas)
    all_pnls = np.array(all_pnls)
    
    # Set up the figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Stock prices
    ax2 = fig.add_subplot(gs[0, 1])  # Delta
    ax3 = fig.add_subplot(gs[1, :])  # P&L
    
    # Colors for paths
    colors = plt.cm.tab10(np.linspace(0, 1, args.n_paths))
    
    # Initialize plot elements
    ax1.set_xlim(0, args.expiry)
    ax1.set_ylim(paths.min() * 0.9, paths.max() * 1.1)
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Stock Price")
    ax1.set_title("Stock Price Paths")
    ax1.axhline(args.strike, color="red", linestyle="--", alpha=0.5, label=f"Strike = {args.strike}")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    
    ax2.set_xlim(0, args.expiry)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Delta")
    ax2.set_title("Option Delta (Hedge Ratio)")
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(alpha=0.3)
    
    pnl_range = max(abs(all_pnls.min()), abs(all_pnls.max())) * 1.2
    ax3.set_xlim(0, args.expiry)
    ax3.set_ylim(-pnl_range, pnl_range)
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("P&L")
    ax3.set_title("Delta-Hedged Portfolio P&L")
    ax3.axhline(0, color="black", linestyle="-", alpha=0.5)
    ax3.grid(alpha=0.3)
    
    # Create line objects for animation
    price_lines = [ax1.plot([], [], color=colors[i], linewidth=1.5, alpha=0.8)[0] for i in range(args.n_paths)]
    delta_lines = [ax2.plot([], [], color=colors[i], linewidth=1.5, alpha=0.8)[0] for i in range(args.n_paths)]
    pnl_lines = [ax3.plot([], [], color=colors[i], linewidth=1.5, alpha=0.8, label=f"Path {i+1}")[0] for i in range(args.n_paths)]
    
    # Current position markers
    price_dots = [ax1.plot([], [], 'o', color=colors[i], markersize=8)[0] for i in range(args.n_paths)]
    delta_dots = [ax2.plot([], [], 'o', color=colors[i], markersize=8)[0] for i in range(args.n_paths)]
    pnl_dots = [ax3.plot([], [], 'o', color=colors[i], markersize=8)[0] for i in range(args.n_paths)]
    
    ax3.legend(loc="upper left", ncol=min(args.n_paths, 5))
    
    # Time indicator
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12, fontweight="bold")
    
    fig.suptitle(
        f"Delta Hedging Simulation: S₀={args.spot}, K={args.strike}, σ={args.volatility:.0%}, r={args.rate:.1%}",
        fontsize=14, fontweight="bold"
    )
    
    def init():
        for line in price_lines + delta_lines + pnl_lines:
            line.set_data([], [])
        for dot in price_dots + delta_dots + pnl_dots:
            dot.set_data([], [])
        time_text.set_text("")
        return price_lines + delta_lines + pnl_lines + price_dots + delta_dots + pnl_dots + [time_text]
    
    def animate(frame):
        idx = frame + 1  # frame 0 shows up to index 1
        t_current = times[:idx]
        
        for i in range(args.n_paths):
            price_lines[i].set_data(t_current, paths[i, :idx])
            delta_lines[i].set_data(t_current, all_deltas[i, :idx])
            pnl_lines[i].set_data(t_current, all_pnls[i, :idx])
            
            # Update current position dots
            price_dots[i].set_data([times[idx-1]], [paths[i, idx-1]])
            delta_dots[i].set_data([times[idx-1]], [all_deltas[i, idx-1]])
            pnl_dots[i].set_data([times[idx-1]], [all_pnls[i, idx-1]])
        
        time_text.set_text(f"t = {times[idx-1]:.3f} years  |  Days = {int(times[idx-1] * 252)}")
        
        return price_lines + delta_lines + pnl_lines + price_dots + delta_dots + pnl_dots + [time_text]
    
    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=args.n_steps,
        interval=1000 // args.fps,
        blit=True
    )
    
    if args.output:
        print(f"Saving animation to {args.output}...")
        if args.output.endswith('.gif'):
            anim.save(args.output, writer='pillow', fps=args.fps)
        else:
            anim.save(args.output, writer='ffmpeg', fps=args.fps)
        print(f"Animation saved to: {args.output}")
    else:
        print("Displaying animation (close window to exit)...")
        plt.show()


if __name__ == "__main__":
    main()
