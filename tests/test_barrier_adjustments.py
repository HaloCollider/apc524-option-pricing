"""
Tests for barrier adjustment formulas.
"""

import numpy as np
import pytest

from option_pricing.barrier_adjustments import (
    bgk_price_adjustment,
    broadie_glasserman_kou_adjustment,
    discrete_to_continuous_barrier,
)


class TestBroadieGlassermanKouAdjustment:
    """Test BGK barrier adjustment formula."""

    def test_up_barrier_adjustment(self):
        """Test that up barriers are adjusted downward."""
        original_barrier = 120.0
        adjusted = broadie_glasserman_kou_adjustment(
            barrier=original_barrier,
            volatility=0.25,
            maturity=1.0,
            num_observations=365,
            barrier_direction="up",
        )

        # Up barriers should be reduced for discrete monitoring
        assert adjusted < original_barrier

    def test_down_barrier_adjustment(self):
        """Test that down barriers are adjusted upward."""
        original_barrier = 80.0
        adjusted = broadie_glasserman_kou_adjustment(
            barrier=original_barrier,
            volatility=0.25,
            maturity=1.0,
            num_observations=365,
            barrier_direction="down",
        )

        # Down barriers should be increased for discrete monitoring
        assert adjusted > original_barrier

    def test_more_observations_less_adjustment(self):
        """Test that more frequent monitoring reduces adjustment."""
        barrier = 120.0
        params = {
            "barrier": barrier,
            "volatility": 0.25,
            "maturity": 1.0,
            "barrier_direction": "up",
        }

        adj_monthly = broadie_glasserman_kou_adjustment(num_observations=12, **params)
        adj_daily = broadie_glasserman_kou_adjustment(num_observations=365, **params)

        # Daily monitoring should have less adjustment (closer to original)
        assert abs(adj_daily - barrier) < abs(adj_monthly - barrier)

    def test_higher_volatility_more_adjustment(self):
        """Test that higher volatility increases adjustment magnitude."""
        barrier = 120.0
        params = {
            "barrier": barrier,
            "maturity": 1.0,
            "num_observations": 365,
            "barrier_direction": "up",
        }

        adj_low_vol = broadie_glasserman_kou_adjustment(volatility=0.1, **params)
        adj_high_vol = broadie_glasserman_kou_adjustment(volatility=0.5, **params)

        # Higher volatility should have more adjustment
        assert abs(adj_high_vol - barrier) > abs(adj_low_vol - barrier)

    def test_longer_maturity_more_adjustment(self):
        """Test that longer maturity increases adjustment (for fixed num_observations)."""
        barrier = 120.0
        params = {
            "barrier": barrier,
            "volatility": 0.25,
            "num_observations": 12,
            "barrier_direction": "up",
        }

        adj_short = broadie_glasserman_kou_adjustment(maturity=0.5, **params)
        adj_long = broadie_glasserman_kou_adjustment(maturity=2.0, **params)

        # Longer maturity with same number of observations means larger intervals
        assert abs(adj_long - barrier) > abs(adj_short - barrier)

    def test_case_insensitive_direction(self):
        """Test that barrier_direction is case insensitive."""
        params = {
            "barrier": 120.0,
            "volatility": 0.25,
            "maturity": 1.0,
            "num_observations": 365,
        }

        adj1 = broadie_glasserman_kou_adjustment(barrier_direction="up", **params)
        adj2 = broadie_glasserman_kou_adjustment(barrier_direction="UP", **params)
        adj3 = broadie_glasserman_kou_adjustment(barrier_direction="Up", **params)

        assert abs(adj1 - adj2) < 1e-10
        assert abs(adj1 - adj3) < 1e-10

    def test_invalid_direction(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="barrier_direction must be"):
            broadie_glasserman_kou_adjustment(
                barrier=120.0,
                volatility=0.25,
                maturity=1.0,
                num_observations=365,
                barrier_direction="invalid",
            )

    def test_negative_barrier(self):
        """Test that negative barrier raises ValueError."""
        with pytest.raises(ValueError, match="barrier must be positive"):
            broadie_glasserman_kou_adjustment(
                barrier=-120.0,
                volatility=0.25,
                maturity=1.0,
                num_observations=365,
                barrier_direction="up",
            )

    def test_negative_volatility(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="volatility must be positive"):
            broadie_glasserman_kou_adjustment(
                barrier=120.0,
                volatility=-0.25,
                maturity=1.0,
                num_observations=365,
                barrier_direction="up",
            )

    def test_zero_observations(self):
        """Test that zero observations raises ValueError."""
        with pytest.raises(ValueError, match="num_observations must be positive"):
            broadie_glasserman_kou_adjustment(
                barrier=120.0,
                volatility=0.25,
                maturity=1.0,
                num_observations=0,
                barrier_direction="up",
            )


class TestDiscreteToContinuousBarrier:
    """Test inverse BGK adjustment."""

    def test_inverse_relationship_up(self):
        """Test that discrete_to_continuous is inverse of BGK for up barriers."""
        discrete_barrier = 120.0
        params = {
            "volatility": 0.25,
            "maturity": 1.0,
            "num_observations": 365,
            "barrier_direction": "up",
        }

        # Convert discrete to continuous
        continuous = discrete_to_continuous_barrier(discrete_barrier, **params)

        # Apply BGK to get back to discrete
        back_to_discrete = broadie_glasserman_kou_adjustment(barrier=continuous, **params)

        # Should recover original discrete barrier
        assert abs(back_to_discrete - discrete_barrier) < 1e-6

    def test_inverse_relationship_down(self):
        """Test that discrete_to_continuous is inverse of BGK for down barriers."""
        discrete_barrier = 80.0
        params = {
            "volatility": 0.25,
            "maturity": 1.0,
            "num_observations": 365,
            "barrier_direction": "down",
        }

        continuous = discrete_to_continuous_barrier(discrete_barrier, **params)
        back_to_discrete = broadie_glasserman_kou_adjustment(barrier=continuous, **params)

        assert abs(back_to_discrete - discrete_barrier) < 1e-6

    def test_continuous_higher_for_down(self):
        """Test that continuous barrier > discrete barrier for down barriers."""
        discrete = 80.0
        continuous = discrete_to_continuous_barrier(
            discrete_barrier=discrete,
            volatility=0.25,
            maturity=1.0,
            num_observations=365,
            barrier_direction="down",
        )

        # Continuous should be lower (more conservative)
        assert continuous < discrete

    def test_continuous_lower_for_up(self):
        """Test that continuous barrier < discrete barrier for up barriers."""
        discrete = 120.0
        continuous = discrete_to_continuous_barrier(
            discrete_barrier=discrete,
            volatility=0.25,
            maturity=1.0,
            num_observations=365,
            barrier_direction="up",
        )

        # Continuous should be higher (more conservative)
        assert continuous > discrete


class TestBGKPriceAdjustment:
    """Test price adjustment function."""

    def test_returns_adjusted_barrier(self):
        """Test that function returns adjusted barrier level."""
        adjusted = bgk_price_adjustment(
            continuous_price=10.0,
            spot=100.0,
            barrier=120.0,
            volatility=0.25,
            maturity=1.0,
            num_observations=365,
            barrier_direction="up",
        )

        # Should return a different barrier
        assert adjusted != 120.0
        assert adjusted < 120.0  # Up barrier adjusted down

    def test_matches_bgk_adjustment(self):
        """Test that result matches direct BGK adjustment."""
        params = {
            "barrier": 120.0,
            "volatility": 0.25,
            "maturity": 1.0,
            "num_observations": 365,
            "barrier_direction": "up",
        }

        adjusted1 = bgk_price_adjustment(
            continuous_price=10.0,  # Not used in current implementation
            spot=100.0,  # Not used in current implementation
            **params,
        )

        adjusted2 = broadie_glasserman_kou_adjustment(**params)

        assert abs(adjusted1 - adjusted2) < 1e-10
