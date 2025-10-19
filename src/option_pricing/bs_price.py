"""
Black–Scholes–Merton (BSM) formulas for European options with continuous dividend yield.

All inputs are real scalars. Time is in *years*; interest rates (r) and dividend yields (q)
are *continuously compounded*. Volatility (sigma) is the annualized standard deviation
of log-returns.

Parameter glossary (used throughout):
    S : Spot (underlying) price
    K : Strike price
    T : Time to maturity in years
    r : Risk-free rate (continuous compounding)
    sigma : Volatility
    q : Continuous dividend yield
    cp : +1 for a call, −1 for a put

Conventions & numerical notes:
    • To avoid division-by-zero and log singularities as T → 0 or sigma → 0,
      we clamp both at 1e-12 in _d1d2.
    • Normal CDF is implemented via erf for accuracy and speed, which suffices for
      practical option pricing ranges.
    • The “cp-coding” trick collapses call/put formulas into one line using cp ∈ {+1, −1}.
    • All Greeks here are model (risk-neutral) Greeks w.r.t. their canonical definitions.
      Theta is per *year* (divide by 365 or 252 for day-based units as needed).
"""

from math import log, sqrt, exp, erf, pi

# Standard normal CDF and PDF. We keep these as tiny lambdas for terseness.
# N(x) = Φ(x), n(x) = φ(x).
N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
n = lambda x: exp(-0.5 * x * x) / sqrt(2 * pi)


def _d1d2(S, K, T, r, sigma, q):
    """
    Compute the Black–Scholes d1 and d2 under a continuous dividend yield q.

    d1 = [ln(S/K) + (r − q + 0.5 σ²) T] / (σ √T)
    d2 = d1 − σ √T

    Numerical safeguards:
        • T and sigma are floored at 1e-12 to avoid 0/0 or log(·) issues at expiry or zero vol.

    Returns:
        (d1, d2)
    """
    T = max(T, 1e-12)
    sigma = max(sigma, 1e-12)
    a = (r - q + 0.5 * sigma ** 2) * T
    d1 = (log(S / K) + a) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def bs(S, K, T, r, sigma, cp=1, q=0.0):  # cp = +1 for call, −1 for put
    """
    Black–Scholes–Merton price for a European call/put with continuous dividend yield.

    Unified (cp-coded) closed form:
        V = cp * ( e^{−qT} S Φ(cp·d1) − e^{−rT} K Φ(cp·d2) )

    Args:
        S, K, T, r, sigma, q: See module docstring.
        cp: +1 for call, −1 for put.

    Returns:
        Option price (float).
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    Dq = exp(-q * T)   # dividend discount factor
    Dr = exp(-r * T)   # risk-free discount factor
    return cp * (Dq * S * N(cp * d1) - Dr * K * N(cp * d2))


def greeks(S, K, T, r, sigma, cp=1, q=0.0):
    """
    Compute the primary BSM Greeks under continuous dividend yield.

    Returns a 5-tuple: (delta, gamma, vega, theta, rho)

        Delta:
            Δ = cp · e^{−qT} Φ(cp·d1)
            (For calls: e^{−qT} Φ(d1); for puts: e^{−qT}(Φ(d1) − 1))

        Gamma (same for calls/puts):
            Γ = e^{−qT} φ(d1) / (S σ √T)

        Vega (same for calls/puts; w.r.t. σ, *not* per 1%):
            V = e^{−qT} S φ(d1) √T

        Theta (per *year*; sign is the usual convention of ∂V/∂t):
            Θ = − e^{−qT} S φ(d1) σ / (2 √T)
                + cp [ q e^{−qT} S Φ(cp·d1) − r e^{−rT} K Φ(cp·d2) ]

        Rho:
            ρ = cp · T · e^{−rT} K Φ(cp·d2)
            (For calls: +T K e^{−rT} Φ(d2); for puts: −T K e^{−rT} Φ(−d2))

    Note:
        Theta here is “calendar” theta per annum. For daily theta, divide by 365 (or 252).

    Returns:
        (delta, gamma, vega, theta, rho)
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    Dq = exp(-q * T)
    Dr = exp(-r * T)
    rt = sigma * sqrt(T)

    delta = Dq * N(cp * d1) * cp
    gamma = Dq * n(d1) / (S * rt)
    vega  = Dq * S * n(d1) * sqrt(T)
    theta = -Dq * S * n(d1) * sigma / (2 * sqrt(T)) + cp * (q * Dq * S * N(cp * d1) - r * Dr * K * N(cp * d2))
    rho   = cp * T * Dr * K * N(cp * d2)

    return delta, gamma, vega, theta, rho


def iv(price, S, K, T, r, cp=1, q=0.0, lo=1e-6, hi=5.0, tol=1e-8):
    """
    Implied volatility via robust bisection.

    Solves for σ such that bs(S, K, T, r, σ, cp, q) ≈ price on [lo, hi].

    Behavior:
        • If the initial bracket [lo, hi] does not straddle a root, returns the midpoint
          (best-effort fallback). This typically signals an arbitrage-violating price
          or a bracket that is too narrow for extreme parameters.
        • Up to 80 bisection iterations, terminating early once the interval width < tol.
        • Returns σ in absolute units (e.g., 0.2 for 20% annualized vol).

    Practical tips:
        • Reasonable brackets for equity options are often [1e−4, 3.0].
        • Ensure `price` is a *clean* European price (no early exercise, no discrete dividends).

    Args:
        price: Observed option price to invert.
        S, K, T, r, cp, q: As above.
        lo, hi: Initial volatility bracket.
        tol: Absolute tolerance on bracket width.

    Returns:
        Implied volatility (float).
    """
    f = lambda v: bs(S, K, T, r, v, cp, q) - price
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return (lo + hi) / 2  # no bracket: return midpoint as a conservative fallback

    for _ in range(80):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid
        if hi - lo < tol:
            break
    return (lo + hi) / 2
