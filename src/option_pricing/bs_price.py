from math import log, sqrt, exp, erf, pi

# Normal CDF and PDF
N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
n = lambda x: exp(-0.5 * x * x) / sqrt(2 * pi)

def _d1d2(S, K, T, r, sigma, q):
    T = max(T, 1e-12)
    sigma = max(sigma, 1e-12)
    a = (r - q + 0.5 * sigma ** 2) * T
    d1 = (log(S / K) + a) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2

def bs(S, K, T, r, sigma, cp=1, q=0.0):  # cp=+1 for call, -1 for put
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    Dq = exp(-q * T)
    Dr = exp(-r * T)
    return cp * (Dq * S * N(cp * d1) - Dr * K * N(cp * d2))

def greeks(S, K, T, r, sigma, cp=1, q=0.0):
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
    f = lambda v: bs(S, K, T, r, v, cp, q) - price
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return (lo + hi) / 2  # no bracket, best effort

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
