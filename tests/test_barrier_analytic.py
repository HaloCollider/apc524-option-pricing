"""
Tests for analytical barrier option pricing formulas.
"""

import numpy as np
import pytest

from option_pricing.barrier_analytic import barrier_option_bs
from option_pricing.pricing import black_scholes_price


class TestBarrierOptionBS:
    """Test analytical barrier option pricing."""

    def test_call_up_and_out_basic(self):
        """Test up-and-out call with standard parameters."""
        price = barrier_option_bs(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
        )
        assert price > 0
        assert price < 15  # Should be less than vanilla call

    def test_put_down_and_out_basic(self):
        """Test down-and-out put with standard parameters."""
        price = barrier_option_bs(
            option_type="put",
            barrier_type="do",
            spot=100.0,
            strike=100.0,
            barrier=80.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
        )
        assert price > 0
        assert price < 15  # Should be less than vanilla put

    def test_parity_call_up(self):
        """Verify C_ui + C_uo = C (vanilla call)."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "barrier": 120.0,
            "maturity": 1.0,
            "rate": 0.05,
            "volatility": 0.25,
        }

        c_ui = barrier_option_bs(option_type="call", barrier_type="ui", **params)
        c_uo = barrier_option_bs(option_type="call", barrier_type="uo", **params)
        vanilla_call = float(
            black_scholes_price(
                params["spot"],
                params["strike"],
                params["maturity"],
                params["rate"],
                params["volatility"],
                "call",
            )
        )

        assert abs(c_ui + c_uo - vanilla_call) < 1e-6

    def test_parity_put_down(self):
        """Verify P_di + P_do = P (vanilla put)."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "barrier": 80.0,
            "maturity": 1.0,
            "rate": 0.05,
            "volatility": 0.25,
        }

        p_di = barrier_option_bs(option_type="put", barrier_type="di", **params)
        p_do = barrier_option_bs(option_type="put", barrier_type="do", **params)
        vanilla_put = float(
            black_scholes_price(
                params["spot"],
                params["strike"],
                params["maturity"],
                params["rate"],
                params["volatility"],
                "put",
            )
        )

        assert abs(p_di + p_do - vanilla_put) < 1e-6

    def test_case_insensitive_option_type(self):
        """Test that option_type is case insensitive."""
        params = {
            "barrier_type": "uo",
            "spot": 100.0,
            "strike": 100.0,
            "barrier": 120.0,
            "maturity": 1.0,
            "rate": 0.05,
            "volatility": 0.25,
        }

        price1 = barrier_option_bs(option_type="call", **params)
        price2 = barrier_option_bs(option_type="CALL", **params)
        price3 = barrier_option_bs(option_type="Call", **params)

        assert abs(price1 - price2) < 1e-10
        assert abs(price1 - price3) < 1e-10

    def test_case_insensitive_barrier_type(self):
        """Test that barrier_type is case insensitive."""
        params = {
            "option_type": "call",
            "spot": 100.0,
            "strike": 100.0,
            "barrier": 120.0,
            "maturity": 1.0,
            "rate": 0.05,
            "volatility": 0.25,
        }

        price1 = barrier_option_bs(barrier_type="uo", **params)
        price2 = barrier_option_bs(barrier_type="UO", **params)
        price3 = barrier_option_bs(barrier_type="Uo", **params)

        assert abs(price1 - price2) < 1e-10
        assert abs(price1 - price3) < 1e-10

    def test_invalid_option_type(self):
        """Test that invalid option_type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be"):
            barrier_option_bs(
                option_type="invalid",
                barrier_type="uo",
                spot=100.0,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_invalid_barrier_type(self):
        """Test that invalid barrier_type raises ValueError."""
        with pytest.raises(ValueError, match="barrier_type must be"):
            barrier_option_bs(
                option_type="call",
                barrier_type="invalid",
                spot=100.0,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_invalid_barrier_placement_up(self):
        """Test that up barrier below spot raises ValueError."""
        with pytest.raises(ValueError, match="Up barrier .* must be above spot"):
            barrier_option_bs(
                option_type="call",
                barrier_type="uo",
                spot=100.0,
                strike=100.0,
                barrier=90.0,  # Barrier below spot
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_invalid_barrier_placement_down(self):
        """Test that down barrier above spot raises ValueError."""
        with pytest.raises(ValueError, match="Down barrier .* must be below spot"):
            barrier_option_bs(
                option_type="put",
                barrier_type="do",
                spot=100.0,
                strike=100.0,
                barrier=110.0,  # Barrier above spot
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_negative_inputs(self):
        """Test that negative inputs raise ValueError."""
        with pytest.raises(ValueError):
            barrier_option_bs(
                option_type="call",
                barrier_type="uo",
                spot=-100.0,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_with_dividend_yield(self):
        """Test pricing with non-zero dividend yield."""
        price = barrier_option_bs(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
            dividend_yield=0.02,
        )
        assert price > 0

    def test_all_barrier_combinations(self):
        """Test that all 8 barrier combinations can be priced."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "rate": 0.05,
            "volatility": 0.25,
        }

        # Call options
        for barrier_type in ["ui", "uo"]:
            price = barrier_option_bs(
                option_type="call", barrier_type=barrier_type, barrier=120.0, **params
            )
            assert price >= 0

        for barrier_type in ["di", "do"]:
            price = barrier_option_bs(
                option_type="call", barrier_type=barrier_type, barrier=80.0, **params
            )
            assert price >= 0

        # Put options
        for barrier_type in ["ui", "uo"]:
            price = barrier_option_bs(
                option_type="put", barrier_type=barrier_type, barrier=120.0, **params
            )
            assert price >= 0

        for barrier_type in ["di", "do"]:
            price = barrier_option_bs(
                option_type="put", barrier_type=barrier_type, barrier=80.0, **params
            )
            assert price >= 0

    def test_knock_out_reduces_value(self):
        """Test that knock-out options are cheaper than vanilla."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "rate": 0.05,
            "volatility": 0.25,
        }

        vanilla_call = float(
            black_scholes_price(
                params["spot"],
                params["strike"],
                params["maturity"],
                params["rate"],
                params["volatility"],
                "call",
            )
        )

        uo_call = barrier_option_bs(
            option_type="call", barrier_type="uo", barrier=120.0, **params
        )
        do_call = barrier_option_bs(
            option_type="call", barrier_type="do", barrier=80.0, **params
        )

        assert uo_call < vanilla_call
        assert do_call < vanilla_call
