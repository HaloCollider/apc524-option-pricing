from __future__ import annotations

import numpy as np
from option_pricing import bs_delta_hedge


def test_bs_delta_hedge_basic_properties():
    result = bs_delta_hedge(
        product_type="european",
        payoff_type="call",
        strike=100.0,
        expiry=1.0,
        rate=0.01,
        dividend_yield=0.0,
        spot=100.0,
        sim_vol=0.2,
        mark_vol=0.2,
        n_steps=8,
        hedge_freq=4,
        n_paths=5000,
        hist_min=-2.0,
        hist_max=2.0,
        n_bins=20,
        seed=123,
    )

    mean = result["pnl_mean"]
    std = result["pnl_std"]
    counts = result["hist_counts"]
    edges = result["hist_bin_edges"]

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert counts.shape[0] == 20
    assert np.isclose(np.sum(counts), 5000)

    # With matching simulation and mark-to-market volatilities and
    # reasonably frequent hedging, the average P&L should be close to 0.
    assert abs(mean) < 0.05
    assert std > 0.0
    assert edges.shape[0] == 21
