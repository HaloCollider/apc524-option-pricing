"""
Core option pricing utilities implemented with NumPy.

The module exposes analytical Black-Scholes-Merton valuations alongside
Monte Carlo simulation helpers for geometric Brownian motion paths.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

OptionType = Literal["call", "put"]


def standard_normal_cdf(x: ArrayLike) -> NDArray[np.float64]:
    """
    Evaluate the cumulative distribution function of a standard normal variable.

    Parameters
    ----------
    x : ArrayLike
        Scalar or array of evaluation points.

    Returns
    -------
    numpy.ndarray
        Array of CDF values with ``float64`` dtype.
    """

    values = np.asarray(x, dtype=np.float64)
    return ndtr(values)


def standard_normal_pdf(x: ArrayLike) -> NDArray[np.float64]:
    """
    Evaluate the probability density function of a standard normal variable.

    Parameters
    ----------
    x : ArrayLike
        Scalar or array of evaluation points.

    Returns
    -------
    numpy.ndarray
        Array of PDF values with ``float64`` dtype.
    """

    values = np.asarray(x, dtype=np.float64)
    normalization = 1.0 / np.sqrt(2.0 * np.pi)
    return normalization * np.exp(-0.5 * values**2)


def _validate_scalar(value: float, name: str, *, strictly_positive: bool = True) -> float:
    numeric = float(value)
    if strictly_positive and numeric <= 0:
        raise ValueError(f"{name} must be positive.")
    if not strictly_positive and numeric < 0:
        raise ValueError(f"{name} cannot be negative.")
    return numeric


def _as_float_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def _prepare_bsm_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    sigma = _validate_scalar(volatility, "volatility")

    spot_arr = _as_float_array(spot, "spot")
    strike_arr = _as_float_array(strike, "strike")
    maturity_arr = _as_float_array(maturity, "maturity")

    if np.any(spot_arr <= 0):
        raise ValueError("spot must be strictly positive.")
    if np.any(strike_arr <= 0):
        raise ValueError("strike must be strictly positive.")
    if np.any(maturity_arr <= 0):
        raise ValueError("maturity must be strictly positive.")

    sqrt_t = np.sqrt(maturity_arr)
    numerator = (
        np.log(spot_arr / strike_arr) + (rate - dividend_yield + 0.5 * sigma**2) * maturity_arr
    )
    d1 = numerator / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    discount_factor = np.exp(-rate * maturity_arr)
    dividend_discount = np.exp(-dividend_yield * maturity_arr)

    return (
        spot_arr,
        strike_arr,
        maturity_arr,
        sigma,
        sqrt_t,
        d1,
        d2,
        discount_factor,
        dividend_discount,
    )


def black_scholes_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> NDArray[np.float64]:
    """
    Compute the primary Black-Scholes-Merton Greeks in a single pass.

    Parameters
    ----------
    spot : ArrayLike
        Spot price(s). Must be strictly positive.
    strike : ArrayLike
        Strike price(s). Must be strictly positive.
    maturity : ArrayLike
        Time to maturity in years. Must be strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility. Must be strictly positive.
    option_type : {"call", "put"}, default="call"
        Determines whether delta, theta, and rho correspond to a call or put.
    dividend_yield : float, default=0.0
        Continuous dividend or convenience yield.

    Returns
    -------
    numpy.ndarray
        Array where the final axis stores ``[delta, gamma, vega, theta, rho]``.

    Raises
    ------
    ValueError
        If numeric inputs violate their constraints or the option type is invalid.

    Examples
    --------
    >>> greeks = black_scholes_greeks(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2)
    >>> greeks[..., 0]  # delta
    array(0.63683065)
    """

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'.")

    (
        spot_arr,
        strike_arr,
        maturity_arr,
        sigma,
        sqrt_t,
        d1,
        d2,
        discount_factor,
        dividend_discount,
    ) = _prepare_bsm_inputs(spot, strike, maturity, rate, volatility, dividend_yield)

    pdf = standard_normal_pdf(d1)
    if option_type == "call":
        delta = dividend_discount * standard_normal_cdf(d1)
    else:
        delta = dividend_discount * (standard_normal_cdf(d1) - 1.0)

    gamma = dividend_discount * pdf / (spot_arr * sigma * sqrt_t)
    vega = spot_arr * dividend_discount * pdf * sqrt_t

    pdf_term = -(spot_arr * dividend_discount * pdf * sigma) / (2.0 * sqrt_t)
    if option_type == "call":
        theta = (
            pdf_term
            - rate * strike_arr * discount_factor * standard_normal_cdf(d2)
            + dividend_yield * spot_arr * dividend_discount * standard_normal_cdf(d1)
        )
        rho = maturity_arr * strike_arr * discount_factor * standard_normal_cdf(d2)
    else:
        theta = (
            pdf_term
            + rate * strike_arr * discount_factor * standard_normal_cdf(-d2)
            - dividend_yield * spot_arr * dividend_discount * standard_normal_cdf(-d1)
        )
        rho = -maturity_arr * strike_arr * discount_factor * standard_normal_cdf(-d2)

    greeks = np.stack((delta, gamma, vega, theta, rho), axis=-1)
    return np.asarray(greeks, dtype=np.float64)


def black_scholes_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> NDArray[np.float64]:
    """
    Price a European call or put using the Black-Scholes-Merton model.

    Parameters
    ----------
    spot : ArrayLike
        Spot price(s) of the underlying asset. Must be strictly positive.
    strike : ArrayLike
        Strike price(s). Must be strictly positive.
    maturity : ArrayLike
        Time to maturity in years. Must be strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility. Must be strictly positive.
    option_type : {"call", "put"}, default="call"
        Selects the payoff to price.
    dividend_yield : float, default=0.0
        Continuous dividend or convenience yield.

    Returns
    -------
    numpy.ndarray
        Array of option values broadcast from the provided inputs.

    Raises
    ------
    ValueError
        If an input violates the constraints or the option type is invalid.

    Examples
    --------
    >>> black_scholes_price(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)
    array(10.45058357)
    """

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'.")

    (
        spot_arr,
        strike_arr,
        maturity_arr,
        sigma,
        sqrt_t,
        d1,
        d2,
        discount_factor,
        dividend_discount,
    ) = _prepare_bsm_inputs(spot, strike, maturity, rate, volatility, dividend_yield)

    if option_type == "call":
        price = (
            dividend_discount * standard_normal_cdf(d1) * spot_arr
            - discount_factor * standard_normal_cdf(d2) * strike_arr
        )
    else:
        price = (
            discount_factor * standard_normal_cdf(-d2) * strike_arr
            - dividend_discount * standard_normal_cdf(-d1) * spot_arr
        )

    return np.asarray(price, dtype=np.float64)


def black_scholes_digital_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> NDArray[np.float64]:
    """Price a European cash-or-nothing digital option under Black-Scholes-Merton.

    The payoff is ``1`` if the option finishes in-the-money and ``0`` otherwise.

    Parameters
    ----------
    spot : ArrayLike
        Spot price(s) of the underlying asset. Must be strictly positive.
    strike : ArrayLike
        Strike price(s). Must be strictly positive.
    maturity : ArrayLike
        Time to maturity in years. Must be strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility. Must be strictly positive.
    option_type : {"call", "put"}, default="call"
        Selects the payoff direction.
    dividend_yield : float, default=0.0
        Continuous dividend or convenience yield.

    Returns
    -------
    numpy.ndarray
        Array of option values broadcast from the provided inputs.

    Raises
    ------
    ValueError
        If an input violates the constraints or the option type is invalid.
    """

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'.")

    (
        spot_arr,
        strike_arr,
        maturity_arr,
        sigma,
        sqrt_t,
        d1,
        d2,
        discount_factor,
        dividend_discount,
    ) = _prepare_bsm_inputs(spot, strike, maturity, rate, volatility, dividend_yield)

    if option_type == "call":
        price = discount_factor * standard_normal_cdf(d2)
    else:
        price = discount_factor * standard_normal_cdf(-d2)

    return np.asarray(price, dtype=np.float64)


def black_scholes_digital_delta(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> NDArray[np.float64]:
    """Compute the Black-Scholes-Merton delta of a digital option.

    This corresponds to the derivative of the cash-or-nothing price with
    respect to the underlying spot price.

    Parameters
    ----------
    spot : ArrayLike
        Spot price(s) of the underlying asset. Must be strictly positive.
    strike : ArrayLike
        Strike price(s). Must be strictly positive.
    maturity : ArrayLike
        Time to maturity in years. Must be strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility. Must be strictly positive.
    option_type : {"call", "put"}, default="call"
        Selects the payoff direction.
    dividend_yield : float, default=0.0
        Continuous dividend or convenience yield.

    Returns
    -------
    numpy.ndarray
        Array of delta values broadcast from the provided inputs.

    Raises
    ------
    ValueError
        If an input violates the constraints or the option type is invalid.
    """

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'.")

    (
        spot_arr,
        strike_arr,
        maturity_arr,
        sigma,
        sqrt_t,
        d1,
        d2,
        discount_factor,
        dividend_discount,
    ) = _prepare_bsm_inputs(spot, strike, maturity, rate, volatility, dividend_yield)

    pdf_d2 = standard_normal_pdf(d2)
    base = discount_factor * pdf_d2 / (spot_arr * sigma * sqrt_t)

    if option_type == "call":
        delta = base
    else:
        delta = -base

    return np.asarray(delta, dtype=np.float64)


def simulate_gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    *,
    steps: int,
    paths: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Simulate geometric Brownian motion price paths.

    Parameters
    ----------
    spot : float
        Initial underlying price. Must be strictly positive.
    maturity : float
        Time horizon in years. Must be strictly positive.
    rate : float
        Continuously compounded drift parameter.
    volatility : float
        Annualized volatility. Must be non-negative.
    steps : int
        Number of timesteps per path. Must be positive.
    paths : int
        Number of independent simulation paths. Must be positive.
    rng : numpy.random.Generator, optional
        Source of randomness. If omitted, ``default_rng`` is used.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(paths, steps + 1)`` that stores the simulated spot
        trajectories, including the initial value at column zero.

    Raises
    ------
    ValueError
        If any numeric constraint is violated.
    """

    s0 = _validate_scalar(spot, "spot")
    t = _validate_scalar(maturity, "maturity")
    sigma = _validate_scalar(volatility, "volatility", strictly_positive=False)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if paths <= 0:
        raise ValueError("paths must be positive.")

    rng = rng or np.random.default_rng()

    dt = t / steps
    drift = (rate - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    shocks = rng.normal(loc=drift, scale=diffusion, size=(paths, steps))
    log_paths = np.cumsum(shocks, axis=1) + np.log(s0)

    simulated = np.empty((paths, steps + 1), dtype=np.float64)
    simulated[:, 0] = s0
    simulated[:, 1:] = np.exp(log_paths)
    return simulated


def monte_carlo_european_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    *,
    steps: int = 64,
    paths: int = 50_000,
    option_type: OptionType = "call",
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Estimate a European option price with Monte Carlo simulation.

    Parameters
    ----------
    spot : float
        Initial underlying price. Must be strictly positive.
    strike : float
        Strike price. Must be strictly positive.
    maturity : float
        Time to maturity in years. Must be strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility. Must be non-negative.
    steps : int, default=64
        Number of timesteps per path. Must be positive.
    paths : int, default=50_000
        Number of simulated paths. Must be positive.
    option_type : {"call", "put"}, default="call"
        Selects the payoff to evaluate.
    rng : numpy.random.Generator, optional
        Source of randomness. If omitted, ``default_rng`` is used.

    Returns
    -------
    tuple of float
        Discounted price estimate and its standard error.

    Raises
    ------
    ValueError
        If numeric inputs violate their constraints or the option type is invalid.
    """

    s0 = _validate_scalar(spot, "spot")
    k = _validate_scalar(strike, "strike")
    t = _validate_scalar(maturity, "maturity")
    sigma = _validate_scalar(volatility, "volatility", strictly_positive=False)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if paths <= 0:
        raise ValueError("paths must be positive.")

    trajectories = simulate_gbm_paths(
        s0,
        t,
        rate,
        sigma,
        steps=steps,
        paths=paths,
        rng=rng,
    )
    terminal_prices = trajectories[:, -1]

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - k, 0.0)
    else:
        payoffs = np.maximum(k - terminal_prices, 0.0)

    discount = np.exp(-rate * t)
    mean_payoff = np.mean(payoffs)
    price = float(discount * mean_payoff)

    if payoffs.size > 1:
        std = np.std(payoffs, ddof=1)
        std_err = float(discount * (std / np.sqrt(payoffs.size)))
    else:
        std_err = 0.0

    return price, std_err


def monte_carlo_digital_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    *,
    steps: int = 1,
    paths: int = 50_000,
    option_type: OptionType = "call",
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Estimate a European cash-or-nothing digital option price via Monte Carlo.

    The payoff is ``1`` if the option finishes in-the-money and ``0`` otherwise.

    Parameters
    ----------
    spot : float
        Initial underlying price. Must be strictly positive.
    strike : float
        Strike price. Must be strictly positive.
    maturity : float
        Time to maturity in years. Must be strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility. Must be non-negative.
    steps : int, default=1
        Number of timesteps per path. Must be positive. Digital payoff depends
        only on the terminal price so a single step is sufficient.
    paths : int, default=50_000
        Number of simulated paths. Must be positive.
    option_type : {"call", "put"}, default="call"
        Selects the payoff direction.
    rng : numpy.random.Generator, optional
        Source of randomness. If omitted, ``default_rng`` is used.

    Returns
    -------
    tuple of float
        Discounted price estimate and its standard error.

    Raises
    ------
    ValueError
        If numeric inputs violate their constraints or the option type is invalid.
    """

    s0 = _validate_scalar(spot, "spot")
    k = _validate_scalar(strike, "strike")
    t = _validate_scalar(maturity, "maturity")
    sigma = _validate_scalar(volatility, "volatility", strictly_positive=False)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if paths <= 0:
        raise ValueError("paths must be positive.")

    trajectories = simulate_gbm_paths(
        s0,
        t,
        rate,
        sigma,
        steps=steps,
        paths=paths,
        rng=rng,
    )
    terminal_prices = trajectories[:, -1]

    if option_type == "call":
        indicators = (terminal_prices > k).astype(np.float64)
    else:
        indicators = (terminal_prices < k).astype(np.float64)

    discount = np.exp(-rate * t)
    mean_ind = float(np.mean(indicators))
    price = float(discount * mean_ind)

    if indicators.size > 1:
        std = float(np.std(indicators, ddof=1))
        std_err = float(discount * (std / np.sqrt(indicators.size)))
    else:
        std_err = 0.0

    return price, std_err
