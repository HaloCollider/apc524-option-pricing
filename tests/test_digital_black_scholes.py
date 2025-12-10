from __future__ import annotations

import numpy as np
from option_pricing.pricing import (
    black_scholes_digital_delta,
    black_scholes_digital_price,
    black_scholes_price,
)


def test_digital_call_price_matches_strike_derivative():
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    vol = 0.2

    eps = 1e-4
    call_plus = black_scholes_price(
        spot=spot,
        strike=strike + eps,
        maturity=maturity,
        rate=rate,
        volatility=vol,
        option_type="call",
    )
    call_minus = black_scholes_price(
        spot=spot,
        strike=strike - eps,
        maturity=maturity,
        rate=rate,
        volatility=vol,
        option_type="call",
    )

    # dC/dK ≈ (C(K+eps) - C(K-eps)) / (2 eps) and
    # digital_call_price = -dC/dK.
    dcdk = (float(call_plus) - float(call_minus)) / (2.0 * eps)
    digital_price = float(
        black_scholes_digital_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol,
            option_type="call",
        )
    )

    assert np.isclose(digital_price, -dcdk, rtol=1e-4, atol=1e-6)


def test_digital_call_delta_matches_spot_derivative():
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    vol = 0.2

    eps = 1e-4
    price_plus = float(
        black_scholes_digital_price(
            spot=spot + eps,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol,
            option_type="call",
        )
    )
    price_minus = float(
        black_scholes_digital_price(
            spot=spot - eps,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol,
            option_type="call",
        )
    )

    numerical_delta = (price_plus - price_minus) / (2.0 * eps)
    analytic_delta = float(
        black_scholes_digital_delta(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol,
            option_type="call",
        )
    )

    assert np.isclose(analytic_delta, numerical_delta, rtol=1e-4, atol=1e-6)
