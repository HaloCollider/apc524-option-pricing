import unittest

import numpy as np
import numpy.testing as npt

from option_pricing import (
    black_scholes_greeks,
    black_scholes_price,
    monte_carlo_digital_price,
    monte_carlo_european_price,
    simulate_gbm_paths,
    standard_normal_cdf,
    standard_normal_pdf,
)


class PricingTests(unittest.TestCase):
    def test_standard_normal_cdf_matches_reference_values(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        result = standard_normal_cdf(x)
        reference = np.array([0.15865525393145707, 0.5, 0.8413447460685429], dtype=np.float64)
        npt.assert_allclose(result, reference, rtol=0.0, atol=1e-12)

    def test_standard_normal_pdf_matches_reference_values(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        result = standard_normal_pdf(x)
        reference = np.array(
            [0.24197072451914337, 0.3989422804014327, 0.24197072451914337],
            dtype=np.float64,
        )
        npt.assert_allclose(result, reference, rtol=0.0, atol=1e-12)

    def test_black_scholes_price_matches_published_call_and_put(self) -> None:
        call = black_scholes_price(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
        )
        put = black_scholes_price(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            option_type="put",
        )
        npt.assert_allclose(call, np.array(10.450583572185565), rtol=0.0, atol=1e-12)
        npt.assert_allclose(put, np.array(5.573526022256971), rtol=0.0, atol=1e-12)

    def test_black_scholes_greeks_returns_expected_vector(self) -> None:
        greeks_call = black_scholes_greeks(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
        )
        expected_call = np.array(
            [
                0.6368306511756191,
                0.018762017345846895,
                37.52403469169379,
                -6.414027546438197,
                53.232481545376345,
            ]
        )
        npt.assert_allclose(greeks_call, expected_call, atol=1e-12)

        greeks_put = black_scholes_greeks(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            option_type="put",
        )
        expected_put = np.array(
            [
                -0.3631693488243809,
                0.018762017345846895,
                37.52403469169379,
                -1.657880423934626,
                -41.89046090469506,
            ]
        )
        npt.assert_allclose(greeks_put, expected_put, atol=1e-12)

    def test_simulate_gbm_paths_returns_expected_shape(self) -> None:
        rng = np.random.default_rng(0)
        paths = simulate_gbm_paths(
            spot=50.0,
            maturity=1.0,
            rate=0.02,
            volatility=0.3,
            steps=4,
            paths=3,
            rng=rng,
        )
        self.assertEqual(paths.shape, (3, 5))
        npt.assert_allclose(paths[:, 0], np.full(3, 50.0))

    def test_monte_carlo_pricer_tracks_black_scholes_value(self) -> None:
        rng = np.random.default_rng(42)
        price, stderr = monte_carlo_european_price(
            spot=105.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            steps=96,
            paths=80_000,
            rng=rng,
        )
        reference = float(
            black_scholes_price(
                spot=105.0,
                strike=100.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.2,
            )
        )
        tolerance = max(3.0 * stderr, 0.15)
        self.assertLess(abs(price - reference), tolerance)
        self.assertGreater(stderr, 0.0)

    def test_monte_carlo_digital_tracks_analytic_price(self) -> None:
        # Parameters
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        rate = 0.05
        vol = 0.2

        # Analytic digital prices for unit payoff: e^{-rT} N(±d2)
        sqrt_t = np.sqrt(maturity)
        d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * maturity) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t
        disc = np.exp(-rate * maturity)
        call_ref = float(disc * standard_normal_cdf(d2))
        put_ref = float(disc * standard_normal_cdf(-d2))

        rng = np.random.default_rng(123)
        call_price, call_se = monte_carlo_digital_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol,
            steps=1,
            paths=200_000,
            option_type="call",
            rng=rng,
        )
        self.assertLess(abs(call_price - call_ref), max(3.0 * call_se, 5e-3))
        self.assertGreater(call_se, 0.0)

        rng = np.random.default_rng(456)
        put_price, put_se = monte_carlo_digital_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol,
            steps=1,
            paths=200_000,
            option_type="put",
            rng=rng,
        )
        self.assertLess(abs(put_price - put_ref), max(3.0 * put_se, 5e-3))
        self.assertGreater(put_se, 0.0)


if __name__ == "__main__":
    unittest.main()
