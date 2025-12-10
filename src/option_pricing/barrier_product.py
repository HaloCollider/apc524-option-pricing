"""
Barrier option product class with discrete monitoring frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .barrier_analytic import barrier_option_bs
from .pde_solver import solve_barrier_pde


class MonitoringFrequency(IntEnum):
    """
    Enumeration of barrier monitoring frequencies.

    Attributes
    ----------
    MONTHLY : int
        12 observations per year.
    WEEKLY : int
        52 observations per year.
    DAILY : int
        365 observations per year.
    """

    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365


@dataclass
class BarrierOption:
    """
    Specification for a barrier option with discrete monitoring.

    This class encapsulates all parameters needed to price a barrier option
    and provides methods to price using both analytical formulas (continuous
    monitoring) and PDE methods (can approximate discrete monitoring).

    Parameters
    ----------
    option_type : str
        Either "call" or "put".
    barrier_type : str
        Two-character barrier type: "ui", "uo", "di", "do".
    spot : float
        Current underlying asset price.
    strike : float
        Strike price.
    barrier : float
        Barrier level.
    maturity : float
        Time to expiration in years.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility.
    dividend_yield : float, default=0.0
        Continuous dividend yield.
    monitoring_frequency : MonitoringFrequency, default=MonitoringFrequency.DAILY
        Frequency at which the barrier is monitored.

    Examples
    --------
    >>> opt = BarrierOption(
    ...     option_type="call",
    ...     barrier_type="uo",
    ...     spot=100.0,
    ...     strike=100.0,
    ...     barrier=120.0,
    ...     maturity=1.0,
    ...     rate=0.05,
    ...     volatility=0.25,
    ... )
    >>> opt.price_analytical()
    7.964401294924583
    """

    option_type: str
    barrier_type: str
    spot: float
    strike: float
    barrier: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    monitoring_frequency: MonitoringFrequency = MonitoringFrequency.DAILY

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        if self.option_type.lower() not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{self.option_type}'")

        if self.barrier_type.lower() not in ("ui", "uo", "di", "do"):
            raise ValueError(
                f"barrier_type must be 'ui', 'uo', 'di', or 'do', got '{self.barrier_type}'"
            )

        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")

        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")

        if self.barrier <= 0:
            raise ValueError(f"barrier must be positive, got {self.barrier}")

        if self.maturity <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")

        if self.volatility <= 0:
            raise ValueError(f"volatility must be positive, got {self.volatility}")

    def generate_observation_dates(self) -> NDArray[np.float64]:
        """
        Generate discrete monitoring dates with stub period at the front.

        The stub period (if any) is placed near t=0, not near expiry, as specified
        in the project requirements.

        Returns
        -------
        numpy.ndarray
            Array of observation times from 0 to maturity, including both endpoints.

        Examples
        --------
        >>> opt = BarrierOption("call", "uo", 100, 100, 120, 1.0, 0.05, 0.25,
        ...                     monitoring_frequency=MonitoringFrequency.MONTHLY)
        >>> dates = opt.generate_observation_dates()
        >>> len(dates)
        13
        """
        num_observations = int(self.monitoring_frequency)
        period = 1.0 / num_observations  # Time between observations

        # Calculate how many full periods fit in maturity
        num_full_periods = int(self.maturity / period)
        stub_length = self.maturity - num_full_periods * period

        if stub_length < 1e-10:  # No stub needed
            observation_times = np.linspace(0, self.maturity, num_full_periods + 1)
        else:
            # Place stub at front
            # First observation after stub
            stub_time = stub_length
            # Subsequent observations at regular intervals
            regular_times = stub_time + np.arange(num_full_periods + 1) * period
            observation_times = np.concatenate(([0.0], [stub_time], regular_times[1:]))

        return observation_times

    def price_analytical(self) -> float:
        """
        Price the barrier option using Black-Scholes closed-form formula.

        This assumes continuous monitoring of the barrier.

        Returns
        -------
        float
            Present value of the barrier option.

        Examples
        --------
        >>> opt = BarrierOption("call", "uo", 100, 100, 120, 1.0, 0.05, 0.25)
        >>> opt.price_analytical()
        7.964401294924583
        """
        return barrier_option_bs(
            option_type=self.option_type,
            barrier_type=self.barrier_type,
            spot=self.spot,
            strike=self.strike,
            barrier=self.barrier,
            maturity=self.maturity,
            rate=self.rate,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield,
        )

    def price_pde(
        self,
        grid_points: int = 100,
        time_steps: int = 100,
        scheme: str = "crank-nicolson",
    ) -> float:
        """
        Price the barrier option using PDE finite difference methods.

        The grid is automatically aligned to the barrier level for improved accuracy.

        Parameters
        ----------
        grid_points : int, default=100
            Number of spatial grid points.
        time_steps : int, default=100
            Number of time steps.
        scheme : {"implicit", "crank-nicolson"}, default="crank-nicolson"
            Finite difference scheme to use.

        Returns
        -------
        float
            Present value of the barrier option.

        Examples
        --------
        >>> opt = BarrierOption("call", "uo", 100, 100, 120, 1.0, 0.05, 0.25)
        >>> opt.price_pde(grid_points=200, time_steps=200)
        7.964401294924583
        """
        return solve_barrier_pde(
            option_type=self.option_type,
            barrier_type=self.barrier_type,
            spot=self.spot,
            strike=self.strike,
            barrier=self.barrier,
            maturity=self.maturity,
            rate=self.rate,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield,
            grid_points=grid_points,
            time_steps=time_steps,
            scheme=scheme,
        )

    def num_observations(self) -> int:
        """
        Calculate the total number of barrier observations.

        Returns
        -------
        int
            Number of times the barrier is checked during the option's life.

        Examples
        --------
        >>> opt = BarrierOption("call", "uo", 100, 100, 120, 1.0, 0.05, 0.25,
        ...                     monitoring_frequency=MonitoringFrequency.DAILY)
        >>> opt.num_observations()
        365
        """
        return int(self.maturity * self.monitoring_frequency)
