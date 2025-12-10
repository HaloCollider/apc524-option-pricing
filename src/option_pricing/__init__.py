"""
Public API for the option_pricing package.
"""

from .pricing import (
    black_scholes_greeks,
    black_scholes_price,
    monte_carlo_digital_price,
    monte_carlo_european_price,
    simulate_gbm_paths,
    standard_normal_cdf,
    standard_normal_pdf,
)

from .barrier_analytic import barrier_option_bs
from .barrier_product import BarrierOption, MonitoringFrequency
from .barrier_adjustments import (
    broadie_glasserman_kou_adjustment,
    discrete_to_continuous_barrier,
)
from .pde_solver import Pde1DSolver, solve_barrier_pde
from .convergence import convergence_study, plot_convergence

__all__ = [
    # Vanilla options
    "black_scholes_greeks",
    "black_scholes_price",
    "monte_carlo_digital_price",
    "monte_carlo_european_price",
    "simulate_gbm_paths",
    "standard_normal_cdf",
    "standard_normal_pdf",
    # Barrier options - Analytical
    "barrier_option_bs",
    # Barrier options - Products
    "BarrierOption",
    "MonitoringFrequency",
    # Barrier options - Adjustments
    "broadie_glasserman_kou_adjustment",
    "discrete_to_continuous_barrier",
    # Barrier options - PDE
    "Pde1DSolver",
    "solve_barrier_pde",
    # Analysis tools
    "convergence_study",
    "plot_convergence",
]
