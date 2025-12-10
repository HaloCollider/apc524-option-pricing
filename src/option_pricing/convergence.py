"""
Convergence analysis tools for PDE solvers.

Provides utilities to study convergence rates and generate visualizations
for finite difference schemes applied to barrier option pricing.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def convergence_study(
    pricing_function: Callable[[int, int], float],
    grid_points_range: list[int] | NDArray[np.int_],
    time_steps_multiplier: float = 1.0,
    reference_value: float | None = None,
) -> pd.DataFrame:
    """
    Perform a convergence study for a PDE pricing method.

    Tests the pricing function across multiple grid refinements and computes
    errors relative to a reference value. Estimates convergence rates from
    successive refinements.

    Parameters
    ----------
    pricing_function : callable
        Function with signature (grid_points: int, time_steps: int) -> float
        that returns the option price for given discretization.
    grid_points_range : list or ndarray
        Sequence of grid point counts to test. Should be increasing.
    time_steps_multiplier : float, default=1.0
        Ratio of time_steps to grid_points. For example, 1.0 means
        time_steps = grid_points.
    reference_value : float, optional
        True or highly accurate reference price. If None, uses the finest
        grid result as reference.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - grid_points: Number of spatial grid points
        - time_steps: Number of time steps
        - price: Computed option price
        - error: Absolute error vs reference (if reference provided)
        - relative_error: Relative error as percentage (if reference provided)
        - convergence_rate: Estimated order of convergence (where applicable)

    Examples
    --------
    >>> def price_fn(n_grid, n_time):
    ...     return solve_barrier_pde(..., grid_points=n_grid, time_steps=n_time)
    >>> results = convergence_study(price_fn, [50, 100, 200, 400], reference_value=8.0)
    >>> results[['grid_points', 'price', 'error']]
    """
    grid_points = np.asarray(grid_points_range, dtype=int)
    n_refinements = len(grid_points)

    if n_refinements < 2:
        raise ValueError("grid_points_range must contain at least 2 values")

    results = {
        "grid_points": [],
        "time_steps": [],
        "price": [],
    }

    # Compute prices for each refinement
    for n_grid in grid_points:
        n_time = max(1, int(n_grid * time_steps_multiplier))
        try:
            price = pricing_function(n_grid, n_time)
            results["grid_points"].append(n_grid)
            results["time_steps"].append(n_time)
            results["price"].append(price)
        except Exception as e:
            print(f"Warning: Failed to compute price for grid_points={n_grid}: {e}")

    if len(results["price"]) < 2:
        raise RuntimeError("Failed to compute prices for sufficient grid refinements")

    df = pd.DataFrame(results)

    # Use finest grid as reference if not provided
    if reference_value is None:
        reference_value = df["price"].iloc[-1]

    # Compute errors
    df["error"] = np.abs(df["price"] - reference_value)
    df["relative_error"] = 100 * df["error"] / np.abs(reference_value)

    # Estimate convergence rates
    convergence_rates = [np.nan]  # First entry has no previous to compare
    for i in range(1, len(df)):
        error_ratio = df["error"].iloc[i - 1] / df["error"].iloc[i]
        grid_ratio = df["grid_points"].iloc[i] / df["grid_points"].iloc[i - 1]

        if error_ratio > 0 and grid_ratio > 1:
            # Order of convergence: error ~ h^p, so log(ratio) = p * log(grid_ratio)
            rate = np.log(error_ratio) / np.log(grid_ratio)
            convergence_rates.append(rate)
        else:
            convergence_rates.append(np.nan)

    df["convergence_rate"] = convergence_rates

    return df


def plot_convergence(
    df: pd.DataFrame,
    log_scale: bool = True,
    title: str = "PDE Convergence Analysis",
) -> tuple:
    """
    Create convergence plots from study results.

    Generates two plots:
    1. Price vs grid points
    2. Error vs grid points (log-log if log_scale=True)

    Parameters
    ----------
    df : pandas.DataFrame
        Results from convergence_study().
    log_scale : bool, default=True
        Use log-log scale for error plot.
    title : str, default="PDE Convergence Analysis"
        Title for the plots.

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes objects.

    Notes
    -----
    Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Price convergence
    ax1 = axes[0]
    ax1.plot(df["grid_points"], df["price"], "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Grid Points", fontsize=12)
    ax1.set_ylabel("Option Price", fontsize=12)
    ax1.set_title(f"{title}\nPrice Convergence", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error convergence
    ax2 = axes[1]
    if log_scale and "error" in df.columns:
        ax2.loglog(df["grid_points"], df["error"], "s-", linewidth=2, markersize=8, color="red")
        ax2.set_xlabel("Grid Points (log scale)", fontsize=12)
        ax2.set_ylabel("Absolute Error (log scale)", fontsize=12)

        # Add reference lines for common convergence orders
        grid_range = df["grid_points"].values
        if len(grid_range) > 1:
            error_ref = df["error"].iloc[0]
            grid_ref = df["grid_points"].iloc[0]

            for order in [1, 2]:
                reference_errors = error_ref * (grid_ref / grid_range) ** order
                ax2.plot(
                    grid_range,
                    reference_errors,
                    "--",
                    alpha=0.5,
                    label=f"Order {order}",
                )

        ax2.legend(fontsize=10)
    else:
        ax2.plot(df["grid_points"], df["error"], "s-", linewidth=2, markersize=8, color="red")
        ax2.set_xlabel("Grid Points", fontsize=12)
        ax2.set_ylabel("Absolute Error", fontsize=12)

    ax2.set_title(f"{title}\nError Convergence", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def extrapolate_richardson(
    coarse_price: float,
    fine_price: float,
    convergence_order: float = 2.0,
    refinement_ratio: float = 2.0,
) -> float:
    """
    Apply Richardson extrapolation to improve accuracy.

    Richardson extrapolation uses results from two different grid sizes to
    estimate the limit as the grid spacing approaches zero.

    Parameters
    ----------
    coarse_price : float
        Price computed on coarser grid.
    fine_price : float
        Price computed on finer grid.
    convergence_order : float, default=2.0
        Expected order of convergence (p in error ~ h^p).
    refinement_ratio : float, default=2.0
        Ratio of coarse to fine grid spacing (typically 2).

    Returns
    -------
    float
        Extrapolated price estimate.

    Notes
    -----
    The Richardson extrapolation formula is:
        P_extrap = P_fine + (P_fine - P_coarse) / (r^p - 1)

    where r is the refinement ratio and p is the convergence order.

    Examples
    --------
    >>> extrapolate_richardson(coarse_price=8.1, fine_price=8.05, convergence_order=2)
    8.033333333333333
    """
    denominator = refinement_ratio**convergence_order - 1.0
    if abs(denominator) < 1e-10:
        raise ValueError(
            f"Invalid refinement ratio {refinement_ratio} or convergence order {convergence_order}"
        )

    extrapolated = fine_price + (fine_price - coarse_price) / denominator
    return float(extrapolated)


def estimate_convergence_order(
    errors: NDArray[np.float64],
    grid_sizes: NDArray[np.float64],
) -> float:
    """
    Estimate convergence order from error measurements.

    Fits a power law error ~ h^p to the data using least squares on log-log scale.

    Parameters
    ----------
    errors : ndarray
        Absolute errors for different grid sizes.
    grid_sizes : ndarray
        Corresponding grid sizes (e.g., spatial step size h).

    Returns
    -------
    float
        Estimated convergence order p.

    Examples
    --------
    >>> errors = np.array([0.1, 0.025, 0.00625])
    >>> grid_sizes = np.array([0.1, 0.05, 0.025])
    >>> estimate_convergence_order(errors, grid_sizes)
    2.0
    """
    if len(errors) != len(grid_sizes):
        raise ValueError("errors and grid_sizes must have same length")

    if len(errors) < 2:
        raise ValueError("Need at least 2 data points to estimate convergence order")

    # Filter out zero or negative values
    valid_mask = (errors > 0) & (grid_sizes > 0)
    if valid_mask.sum() < 2:
        raise ValueError("Need at least 2 positive error and grid size values")

    log_errors = np.log(errors[valid_mask])
    log_sizes = np.log(grid_sizes[valid_mask])

    # Fit: log(error) = log(C) + p * log(h)
    # Using least squares: p = cov(log_h, log_e) / var(log_h)
    coeffs = np.polyfit(log_sizes, log_errors, deg=1)
    convergence_order = coeffs[0]

    return float(convergence_order)
