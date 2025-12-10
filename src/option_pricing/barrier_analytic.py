"""
Closed-form Black-Scholes formulas for continuously monitored barrier options.

Implements analytical pricing for all combinations of:
(call, put) × (up, down) × (in, out)

Based on formulas from Hull's "Options, Futures and Other Derivatives" (10th edition, section 26.9).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .pricing import black_scholes_price, standard_normal_cdf


def barrier_option_bs(
    option_type: str,
    barrier_type: str,
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Price a European barrier option using Black-Scholes closed-form formulas.

    This function prices continuously monitored barrier options for all combinations
    of call/put, up/down, and in/out barriers. Uses parity relationships where
    applicable: C_ui + C_uo = C (vanilla call) and P_di + P_do = P (vanilla put).

    Parameters
    ----------
    option_type : str
        Either "call" or "put" (case insensitive).
    barrier_type : str
        Two-character string specifying barrier type (case insensitive):
        "ui" (up-and-in), "uo" (up-and-out), "di" (down-and-in), "do" (down-and-out).
    spot : float
        Current underlying asset price. Must be strictly positive.
    strike : float
        Strike price. Must be strictly positive.
    barrier : float
        Barrier level. Must be strictly positive.
    maturity : float
        Time to expiration in years. Must be strictly positive.
    rate : float
        Continuously compounded risk-free interest rate.
    volatility : float
        Annualized volatility. Must be strictly positive.
    dividend_yield : float, default=0.0
        Continuous dividend yield.

    Returns
    -------
    float
        Present value of the barrier option.

    Raises
    ------
    ValueError
        If inputs violate constraints or barrier placement is invalid for option type.

    Examples
    --------
    >>> barrier_option_bs("call", "uo", 100, 100, 120, 1.0, 0.05, 0.25)
    7.964401294924583
    >>> barrier_option_bs("put", "di", 100, 100, 80, 1.0, 0.05, 0.25)
    2.7673623781934823
    """
    # Normalize inputs
    opt_type = option_type.lower().strip()
    barr_type = barrier_type.lower().strip()

    if opt_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if barr_type not in ("ui", "uo", "di", "do"):
        raise ValueError(
            f"barrier_type must be 'ui', 'uo', 'di', or 'do', got '{barrier_type}'"
        )

    if spot <= 0 or strike <= 0 or barrier <= 0:
        raise ValueError("spot, strike, and barrier must be strictly positive")

    if maturity <= 0:
        raise ValueError("maturity must be strictly positive")

    if volatility <= 0:
        raise ValueError("volatility must be strictly positive")

    # Validate barrier placement
    if barr_type in ("ui", "uo") and barrier <= spot:
        raise ValueError(f"Up barrier ({barrier}) must be above spot ({spot})")

    if barr_type in ("di", "do") and barrier >= spot:
        raise ValueError(f"Down barrier ({barrier}) must be below spot ({spot})")

    # Route to appropriate pricing function
    if opt_type == "call":
        if barr_type == "di":
            return _barrier_call_down_in(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )
        elif barr_type == "do":
            return _barrier_call_down_out(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )
        elif barr_type == "ui":
            return _barrier_call_up_in(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )
        else:  # uo
            return _barrier_call_up_out(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )
    else:  # put
        if barr_type == "ui":
            return _barrier_put_up_in(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )
        elif barr_type == "uo":
            return _barrier_put_up_out(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )
        elif barr_type == "di":
            return _barrier_put_down_in(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )
        else:  # do
            return _barrier_put_down_out(
                spot, strike, barrier, maturity, rate, volatility, dividend_yield
            )


def _barrier_call_down_in(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price down-and-in call using reflection formula."""
    if H > K:
        # When H > K, option already in-the-money, equals vanilla
        return float(black_scholes_price(S, K, T, r, sigma, "call", dividend_yield=q))

    # For H <= K: Reflection formula
    # C_di = (H/S)^(2λ) * [S*e^(-qT)*N(y) - K*e^(-rT)*N(y-σ√T)]
    lambda_val = _lambda(r, q, sigma)
    y = _y(H, S, K, r, q, sigma, T)  # ln(H²/(SK)) + ...
    
    price = (H / S) ** (2 * lambda_val) * (
        S * np.exp(-q * T) * standard_normal_cdf(y) 
        - K * np.exp(-r * T) * standard_normal_cdf(y - sigma * np.sqrt(T))
    )
    return float(max(price, 0.0))


def _barrier_call_down_out(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price down-and-out call using parity: C_do = C - C_di."""
    # Use parity for all cases
    vanilla = float(black_scholes_price(S, K, T, r, sigma, "call", dividend_yield=q))
    c_di = _barrier_call_down_in(S, K, H, T, r, sigma, q)
    return max(vanilla - c_di, 0.0)


def _barrier_call_up_in(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price up-and-in call: C_ui."""
    # Use parity: C_ui = C - C_uo
    vanilla = float(black_scholes_price(S, K, T, r, sigma, "call", dividend_yield=q))
    c_uo = _barrier_call_up_out(S, K, H, T, r, sigma, q)
    return max(vanilla - c_uo, 0.0)


def _barrier_call_up_out(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price up-and-out call using Reiner-Rubinstein formula."""
    if H <= K:
        # When barrier is at or below strike, option worthless
        return 0.0

    # Reiner-Rubinstein formula for up-and-out call (H > K)
    lambda_val = _lambda(r, q, sigma)
    
    # Calculate d parameters
    x1 = np.log(S/K) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    x2 = np.log(S/H) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    y1 = np.log(H**2/(S*K)) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    y2 = np.log(H/S) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    
    # Compute A, B, C, D terms
    A = S * np.exp(-q*T) * standard_normal_cdf(x1) - K * np.exp(-r*T) * standard_normal_cdf(x1 - sigma*np.sqrt(T))
    B = S * np.exp(-q*T) * standard_normal_cdf(x2) - K * np.exp(-r*T) * standard_normal_cdf(x2 - sigma*np.sqrt(T))
    C = S * np.exp(-q*T) * (H/S)**(2*lambda_val) * standard_normal_cdf(y1) - K * np.exp(-r*T) * (H/S)**(2*lambda_val-2) * standard_normal_cdf(y1 - sigma*np.sqrt(T))
    D = S * np.exp(-q*T) * (H/S)**(2*lambda_val) * standard_normal_cdf(y2) - K * np.exp(-r*T) * (H/S)**(2*lambda_val-2) * standard_normal_cdf(y2 - sigma*np.sqrt(T))
    
    price = A - B - C + D
    return float(max(price, 0.0))


def _barrier_put_up_in(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price up-in put: P_ui (valid for K <= H)."""
    if K > H:
        # When K > H, option already in-the-money, equals vanilla
        return float(black_scholes_price(S, K, T, r, sigma, "put", dividend_yield=q))

    # Standard formula for K <= H
    lambda_val = _lambda(r, q, sigma)
    y = _y(H, S, K, r, q, sigma, T)
    x1 = _x1(S, K, r, q, sigma, T)
    y1 = _y1(H, K, S, r, q, sigma, T)

    term1 = -S * np.exp(-q * T) * ((H / S) ** (2 * lambda_val)) * standard_normal_cdf(-y)
    term2 = K * np.exp(-r * T) * ((H / S) ** (2 * lambda_val - 2)) * standard_normal_cdf(
        -y + sigma * np.sqrt(T)
    )

    return float(term1 + term2)


def _barrier_put_up_out(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price up-and-out put: P_uo (valid for H <= K)."""
    if H > K:
        # When H > K, option is worthless (already knocked out)
        return 0.0

    # Use parity: P_uo = P - P_ui
    vanilla = float(black_scholes_price(S, K, T, r, sigma, "put", dividend_yield=q))
    p_ui = _barrier_put_up_in(S, K, H, T, r, sigma, q)
    return vanilla - p_ui


def _barrier_put_down_in(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price down-and-in put using Reiner-Rubinstein formula."""
    if H > K:
        # When H > K, option already in-the-money, equals vanilla
        return float(black_scholes_price(S, K, T, r, sigma, "put", dividend_yield=q))

    # Reiner-Rubinstein formula for down-and-in put (H < K)
    lambda_val = _lambda(r, q, sigma)
    
    # Calculate d parameters
    x1 = np.log(S/K) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    x2 = np.log(S/H) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    y1 = np.log(H**2/(S*K)) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    y2 = np.log(H/S) / (sigma*np.sqrt(T)) + lambda_val*sigma*np.sqrt(T)
    
    # Compute terms for down-and-in put
    # P_di = -S*e^(-qT)*N(-x2) + K*e^(-rT)*N(-x2+σ√T) + S*e^(-qT)*(H/S)^(2λ)*N(y2) - K*e^(-rT)*(H/S)^(2λ-2)*N(y2-σ√T)
    term1 = -S * np.exp(-q*T) * standard_normal_cdf(-x2)
    term2 = K * np.exp(-r*T) * standard_normal_cdf(-x2 + sigma*np.sqrt(T))
    term3 = S * np.exp(-q*T) * (H/S)**(2*lambda_val) * standard_normal_cdf(y2)
    term4 = K * np.exp(-r*T) * (H/S)**(2*lambda_val-2) * standard_normal_cdf(y2 - sigma*np.sqrt(T))
    
    price = term1 + term2 + term3 - term4
    return float(max(price, 0.0))


def _barrier_put_down_out(
    S: float, K: float, H: float, T: float, r: float, sigma: float, q: float
) -> float:
    """Price down-and-out put: P_do (valid for H <= K)."""
    if H >= K:
        # When H >= K, barrier is above strike - option knocks out before ITM
        return 0.0

    # Use parity: P_do = P - P_di
    vanilla = float(black_scholes_price(S, K, T, r, sigma, "put", dividend_yield=q))
    p_di = _barrier_put_down_in(S, K, H, T, r, sigma, q)
    return vanilla - p_di


# Helper functions for intermediate calculations
def _lambda(r: float, q: float, sigma: float) -> float:
    """Calculate lambda parameter: sqrt(mu² + 2r/σ²) where mu = (r-q-σ²/2)/σ²."""
    mu = (r - q - 0.5 * sigma**2) / (sigma**2)
    return np.sqrt(mu**2 + 2 * r / (sigma**2))


def _y(H: float, S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Calculate y parameter: [ln(H²/(SK)) + (r - q + sigma²/2)T] / (sigma√T)."""
    numerator = np.log(H**2 / (S * K)) + (r - q + 0.5 * sigma**2) * T
    return numerator / (sigma * np.sqrt(T))


def _x1(S: float, X: float, r: float, q: float, sigma: float, T: float) -> float:
    """Calculate x1 parameter: [ln(S/X) + (r - q + sigma²/2)T] / (sigma√T)."""
    numerator = np.log(S / X) + (r - q + 0.5 * sigma**2) * T
    return numerator / (sigma * np.sqrt(T))


def _y1(H: float, K: float, S: float, r: float, q: float, sigma: float, T: float) -> float:
    """Calculate y1 parameter: [ln(H²/(SK)) + (r - q + sigma²/2)T] / (sigma√T)."""
    numerator = np.log(H**2 / (S * K)) + (r - q + 0.5 * sigma**2) * T
    return numerator / (sigma * np.sqrt(T))
