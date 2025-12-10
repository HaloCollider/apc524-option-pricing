from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ..pricing import (
    OptionType,
    black_scholes_digital_delta,
    black_scholes_digital_price,
    black_scholes_greeks,
    black_scholes_price,
    simulate_gbm_paths,
)

ProductType = Literal["european", "digital"]


@dataclass(slots=True)
class BsMcSimpleDeltaHedger:
    """Delta-hedging engine under Black-Scholes via Monte Carlo.

    The class simulates geometric Brownian motion paths for a single
    underlying using ``sim_vol`` and performs delta hedging using
    Black-Scholes prices and deltas evaluated with ``mark_vol``.

    The implementation supports plain European and cash-or-nothing
    digital calls and puts. Hedging is self-financing with a bank
    account accruing at the continuously compounded risk-free rate.
    """

    product_type: ProductType
    payoff_type: OptionType
    strike: float
    expiry: float
    rate: float
    dividend_yield: float
    spot: float
    sim_vol: float
    mark_vol: float
    n_steps: int
    seed: int | None = None
    _rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.product_type not in ("european", "digital"):
            raise ValueError("product_type must be 'european' or 'digital'.")
        if self.payoff_type not in ("call", "put"):
            raise ValueError("payoff_type must be 'call' or 'put'.")
        if self.expiry <= 0:
            raise ValueError("expiry must be positive.")
        if self.spot <= 0:
            raise ValueError("spot must be positive.")
        if self.strike <= 0:
            raise ValueError("strike must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.sim_vol < 0 or self.mark_vol <= 0:
            raise ValueError("volatilities must be non-negative (sim) and positive (mark).")

        # dataclasses with ``slots=True`` require that all attributes are
        # declared as fields. ``_rng`` is therefore part of the dataclass
        # and is initialised here to keep the random generator
        # configuration encapsulated.
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    # --- Internal helpers -------------------------------------------------

    def _option_price_and_delta(self, spot: float, time_to_maturity: float) -> tuple[float, float]:
        if time_to_maturity <= 0:
            # At maturity the option value equals the payoff and delta is
            # either 0 or undefined; the hedging logic never calls this
            # function with non-positive time-to-maturity.
            raise ValueError("time_to_maturity must be positive.")

        if self.product_type == "european":
            price = float(
                black_scholes_price(
                    spot=spot,
                    strike=self.strike,
                    maturity=time_to_maturity,
                    rate=self.rate,
                    volatility=self.mark_vol,
                    option_type=self.payoff_type,
                    dividend_yield=self.dividend_yield,
                )
            )
            greeks = black_scholes_greeks(
                spot=spot,
                strike=self.strike,
                maturity=time_to_maturity,
                rate=self.rate,
                volatility=self.mark_vol,
                option_type=self.payoff_type,
                dividend_yield=self.dividend_yield,
            )
            delta = float(greeks[..., 0])
        else:
            price = float(
                black_scholes_digital_price(
                    spot=spot,
                    strike=self.strike,
                    maturity=time_to_maturity,
                    rate=self.rate,
                    volatility=self.mark_vol,
                    option_type=self.payoff_type,
                    dividend_yield=self.dividend_yield,
                )
            )
            delta = float(
                black_scholes_digital_delta(
                    spot=spot,
                    strike=self.strike,
                    maturity=time_to_maturity,
                    rate=self.rate,
                    volatility=self.mark_vol,
                    option_type=self.payoff_type,
                    dividend_yield=self.dividend_yield,
                )
            )

        return price, delta

    def _payoff(self, spot_terminal: float) -> float:
        if self.product_type == "european":
            if self.payoff_type == "call":
                return max(spot_terminal - self.strike, 0.0)
            return max(self.strike - spot_terminal, 0.0)

        # Digital cash-or-nothing payoff of 1.0
        if self.payoff_type == "call":
            return 1.0 if spot_terminal > self.strike else 0.0
        return 1.0 if spot_terminal < self.strike else 0.0

    # --- Public API -------------------------------------------------------

    def hedge(self, stats_calcs: Sequence[object] | object, hedge_freq: int, n_paths: int) -> None:
        """Run the delta-hedging simulation and update statistics.

        Parameters
        ----------
        stats_calcs : object or sequence of objects
            One or more statistics calculators exposing an ``add_sample``
            method that accepts a single scalar P&L value.
        hedge_freq : int
            Number of hedge rebalancings per year (e.g. 252 for daily,
            52 for weekly). The time discretization used for simulation
            is refined so that hedge dates lie on the simulation grid.
        n_paths : int
            Number of Monte Carlo scenarios to simulate.
        """

        if hedge_freq <= 0:
            raise ValueError("hedge_freq must be positive.")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")

        if isinstance(stats_calcs, (list, tuple)):
            calculators: list[object] = list(stats_calcs)
        else:
            calculators = [stats_calcs]

        # Choose a time grid that is compatible with both the requested
        # simulation resolution and the hedging frequency by using the
        # least common multiple of the per-year step counts.
        steps_per_year = math.lcm(self.n_steps, hedge_freq)
        total_steps = int(round(self.expiry * steps_per_year))
        if total_steps <= 0:
            raise ValueError("Total number of timesteps must be positive.")

        dt = self.expiry / total_steps
        hedge_step = steps_per_year // hedge_freq
        # Number of hedge intervals over the whole life of the option.
        n_hedge_intervals = int(round(self.expiry * hedge_freq))

        hedge_indices = [k * hedge_step for k in range(n_hedge_intervals + 1)]
        if hedge_indices[-1] != total_steps:
            # This should not happen with the construction above, but we
            # guard against silent off-by-one errors.
            raise RuntimeError("Inconsistent hedge grid and simulation grid alignment.")

        paths = simulate_gbm_paths(
            spot=self.spot,
            maturity=self.expiry,
            rate=self.rate,
            volatility=self.sim_vol,
            steps=total_steps,
            paths=n_paths,
            rng=self._rng,
        )

        for path in paths:
            s0 = float(path[0])
            price0, delta = self._option_price_and_delta(s0, self.expiry)

            # Short option + replicating hedge portfolio has zero initial
            # wealth. We track the wealth of the short position and then
            # convert to the long-option P&L at maturity.
            bank = price0 - delta * s0

            for k in range(n_hedge_intervals):
                i0 = hedge_indices[k]
                i1 = hedge_indices[k + 1]
                s_end = float(path[i1])
                dt_interval = (i1 - i0) * dt

                # Bank account accrues interest over the interval.
                if dt_interval > 0.0 and self.rate != 0.0:
                    bank *= math.exp(self.rate * dt_interval)

                # At the final interval we stop rebalancing and only
                # carry the existing hedge to maturity.
                if k == n_hedge_intervals - 1:
                    break

                t_next = i1 * dt
                time_to_maturity = self.expiry - t_next
                if time_to_maturity <= 0.0:
                    break

                _, new_delta = self._option_price_and_delta(s_end, time_to_maturity)
                trade = new_delta - delta
                bank -= trade * s_end
                delta = new_delta

            s_terminal = float(path[-1])
            payoff = self._payoff(s_terminal)
            hedge_value = delta * s_terminal + bank
            # Wealth of the *short* option plus hedge after paying the payoff.
            wealth_short = hedge_value - payoff
            # P&L for the long option + hedge is the opposite.
            pnl_long = -wealth_short

            for calc in calculators:
                add_sample = getattr(calc, "add_sample", None)
                if add_sample is None:
                    raise TypeError("Statistics calculator must expose an 'add_sample' method.")
                add_sample(pnl_long)
