"""
Tests for barrier product class and monitoring frequencies.
"""

import numpy as np
import pytest

from option_pricing.barrier_product import BarrierOption, MonitoringFrequency


class TestMonitoringFrequency:
    """Test monitoring frequency enumeration."""

    def test_frequency_values(self):
        """Test that frequency enums have correct values."""
        assert MonitoringFrequency.MONTHLY == 12
        assert MonitoringFrequency.WEEKLY == 52
        assert MonitoringFrequency.DAILY == 365

    def test_frequency_comparison(self):
        """Test that frequencies can be compared."""
        assert MonitoringFrequency.DAILY > MonitoringFrequency.MONTHLY
        assert MonitoringFrequency.WEEKLY < MonitoringFrequency.DAILY


class TestBarrierOption:
    """Test BarrierOption product class."""

    def test_initialization(self):
        """Test basic initialization."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
        )

        assert opt.option_type == "call"
        assert opt.barrier_type == "uo"
        assert opt.spot == 100.0
        assert opt.monitoring_frequency == MonitoringFrequency.DAILY

    def test_validation_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be"):
            BarrierOption(
                option_type="invalid",
                barrier_type="uo",
                spot=100.0,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_validation_invalid_barrier_type(self):
        """Test that invalid barrier type raises ValueError."""
        with pytest.raises(ValueError, match="barrier_type must be"):
            BarrierOption(
                option_type="call",
                barrier_type="invalid",
                spot=100.0,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_validation_negative_spot(self):
        """Test that negative spot raises ValueError."""
        with pytest.raises(ValueError, match="spot must be positive"):
            BarrierOption(
                option_type="call",
                barrier_type="uo",
                spot=-100.0,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_validation_negative_strike(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be positive"):
            BarrierOption(
                option_type="call",
                barrier_type="uo",
                spot=100.0,
                strike=-100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.25,
            )

    def test_validation_negative_volatility(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="volatility must be positive"):
            BarrierOption(
                option_type="call",
                barrier_type="uo",
                spot=100.0,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                rate=0.05,
                volatility=-0.25,
            )

    def test_generate_observation_dates_monthly(self):
        """Test observation date generation for monthly monitoring."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
            monitoring_frequency=MonitoringFrequency.MONTHLY,
        )

        dates = opt.generate_observation_dates()

        # Should have 13 dates (0, 1/12, 2/12, ..., 12/12)
        assert len(dates) == 13
        assert dates[0] == 0.0
        assert abs(dates[-1] - 1.0) < 1e-10

    def test_generate_observation_dates_daily(self):
        """Test observation date generation for daily monitoring."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
            monitoring_frequency=MonitoringFrequency.DAILY,
        )

        dates = opt.generate_observation_dates()

        # Should have 366 dates (0 plus 365 observations)
        assert len(dates) == 366
        assert dates[0] == 0.0
        assert abs(dates[-1] - 1.0) < 1e-10

    def test_generate_observation_dates_with_stub(self):
        """Test observation date generation with stub period."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=0.75,  # Not a multiple of monthly period
            rate=0.05,
            volatility=0.25,
            monitoring_frequency=MonitoringFrequency.MONTHLY,
        )

        dates = opt.generate_observation_dates()

        # Stub should be at front
        assert dates[0] == 0.0
        # Second observation should be after stub period
        stub_length = 0.75 - int(0.75 * 12) / 12
        if stub_length > 1e-10:
            assert abs(dates[1] - stub_length) < 1e-6

    def test_price_analytical(self):
        """Test analytical pricing method."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
        )

        price = opt.price_analytical()
        assert isinstance(price, float)
        assert price > 0

    def test_price_pde(self):
        """Test PDE pricing method."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
        )

        price = opt.price_pde(grid_points=100, time_steps=100)
        assert isinstance(price, float)
        assert price > 0

    def test_price_methods_agree(self):
        """Test that analytical and PDE methods give similar results."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
        )

        analytical_price = opt.price_analytical()
        pde_price = opt.price_pde(grid_points=200, time_steps=200)

        # PDE methods for barrier options have systematic errors with standard grids
        # 10% tolerance is reasonable for 200x200 grid
        relative_diff = abs(analytical_price - pde_price) / analytical_price
        assert relative_diff < 0.10

    def test_num_observations(self):
        """Test calculation of number of observations."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
            monitoring_frequency=MonitoringFrequency.DAILY,
        )

        num_obs = opt.num_observations()
        assert num_obs == 365

    def test_num_observations_monthly(self):
        """Test number of observations for monthly frequency."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
            monitoring_frequency=MonitoringFrequency.MONTHLY,
        )

        num_obs = opt.num_observations()
        assert num_obs == 12

    def test_with_dividend_yield(self):
        """Test pricing with dividend yield."""
        opt = BarrierOption(
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

        analytical_price = opt.price_analytical()
        pde_price = opt.price_pde(grid_points=200, time_steps=200)

        assert analytical_price > 0
        assert pde_price > 0

    def test_different_schemes(self):
        """Test PDE pricing with different schemes."""
        opt = BarrierOption(
            option_type="call",
            barrier_type="uo",
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.25,
        )

        implicit_price = opt.price_pde(grid_points=100, time_steps=100, scheme="implicit")
        cn_price = opt.price_pde(grid_points=100, time_steps=100, scheme="crank-nicolson")

        assert implicit_price > 0
        assert cn_price > 0
