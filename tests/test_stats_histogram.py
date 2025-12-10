from __future__ import annotations

import numpy as np
from option_pricing.stats.histogram_calculator import HistogramCalculator, MeanVarCalculator


def test_histogram_single_variable_counts():
    edges = np.array([-1.0, 0.0, 1.0])
    hist = HistogramCalculator(bin_edges=edges, n_vars=1)

    hist.add_sample(-0.5)
    hist.add_sample(0.5)

    counts, returned_edges = hist.results()

    assert counts.shape == (2, 1)
    # First sample in first bin, second sample in second bin.
    assert counts[0, 0] == 1
    assert counts[1, 0] == 1
    assert np.allclose(returned_edges, edges)


def test_histogram_multi_variable_counts():
    edges = np.array([-1.0, 0.0, 1.0])
    hist = HistogramCalculator(bin_edges=edges, n_vars=2)

    hist.add_sample([-0.5, 0.5])

    counts, _ = hist.results()

    assert counts.shape == (2, 2)
    assert counts[0, 0] == 1  # first variable in first bin
    assert counts[1, 1] == 1  # second variable in second bin


def test_mean_var_calculator_matches_numpy():
    rng = np.random.default_rng(123)
    data = rng.normal(size=1000)

    calc = MeanVarCalculator(n_vars=1)
    for x in data:
        calc.add_sample(x)

    results = calc.results()
    expected_mean = np.mean(data)
    expected_var = np.var(data, ddof=1)

    assert np.isclose(results["mean"][0], expected_mean, rtol=1e-6, atol=1e-6)
    assert np.isclose(results["variance"][0], expected_var, rtol=1e-6, atol=1e-6)
