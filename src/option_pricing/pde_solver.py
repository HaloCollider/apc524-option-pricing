"""
One-dimensional PDE solver for barrier option pricing using finite difference methods.

Implements fully implicit and Crank-Nicolson schemes with configurable grid alignment.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class Pde1DSolver:
    """
    One-dimensional finite difference solver for barrier options.

    This solver implements both fully implicit and Crank-Nicolson schemes
    for pricing barrier options. The grid can be aligned to the barrier level
    for improved accuracy near the barrier boundary.

    Parameters
    ----------
    spot : float
        Current underlying asset price.
    barrier : float
        Barrier level (used for grid alignment if requested).
    maturity : float
        Time to expiration in years.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annualized volatility.
    dividend_yield : float, default=0.0
        Continuous dividend yield.
    grid_points : int, default=100
        Number of spatial grid points.
    time_steps : int, default=100
        Number of time steps.
    spot_mult : float, default=5.0
        Grid spans from barrier/spot_mult to barrier*spot_mult.

    Attributes
    ----------
    S_grid : ndarray
        Spatial grid of underlying prices.
    dt : float
        Time step size.
    dS : ndarray
        Spatial step sizes (variable grid).
    """

    def __init__(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        grid_points: int = 100,
        time_steps: int = 100,
        spot_mult: float = 5.0,
    ):
        self.spot = spot
        self.barrier = barrier
        self.maturity = maturity
        self.rate = rate
        self.volatility = volatility
        self.dividend_yield = dividend_yield
        self.grid_points = grid_points
        self.time_steps = time_steps
        self.spot_mult = spot_mult

        # Grid alignment: default is to center on barrier
        self.alignment_level = barrier

        # Time discretization
        self.dt = maturity / time_steps

        # Initialize spatial grid (will be set by set_alignment)
        self.S_grid: NDArray[np.float64] = np.array([])
        self.dS: NDArray[np.float64] = np.array([])
        self._setup_grid()

    def set_alignment(self, level: float | None = None) -> None:
        """
        Set the grid alignment level.

        Parameters
        ----------
        level : float or None
            Price level to center the grid on. If None, uses barrier level.
        """
        if level is not None:
            self.alignment_level = level
        else:
            self.alignment_level = self.barrier
        self._setup_grid()

    def _setup_grid(self) -> None:
        """Create spatial grid aligned to the alignment level."""
        S_min = self.alignment_level / self.spot_mult
        S_max = self.alignment_level * self.spot_mult

        # Uniform grid in log-space for better resolution near barrier
        log_S = np.linspace(np.log(S_min), np.log(S_max), self.grid_points)
        self.S_grid = np.exp(log_S)

        # Compute variable spatial steps
        self.dS = np.diff(self.S_grid)

    def solve_implicit(
        self,
        option_type: str,
        strike: float,
        barrier_type: str,
    ) -> float:
        """
        Price a barrier option using the fully implicit scheme.

        Parameters
        ----------
        option_type : str
            Either "call" or "put".
        strike : float
            Strike price.
        barrier_type : str
            Two-character barrier type: "ui", "uo", "di", "do".

        Returns
        -------
        float
            Present value of the barrier option.
        """
        return self._solve(option_type, strike, barrier_type, scheme="implicit")

    def solve_crank_nicolson(
        self,
        option_type: str,
        strike: float,
        barrier_type: str,
    ) -> float:
        """
        Price a barrier option using the Crank-Nicolson scheme.

        Parameters
        ----------
        option_type : str
            Either "call" or "put".
        strike : float
            Strike price.
        barrier_type : str
            Two-character barrier type: "ui", "uo", "di", "do".

        Returns
        -------
        float
            Present value of the barrier option.
        """
        return self._solve(option_type, strike, barrier_type, scheme="crank-nicolson")

    def _solve(
        self,
        option_type: str,
        strike: float,
        barrier_type: str,
        scheme: Literal["implicit", "crank-nicolson"],
    ) -> float:
        """Internal solver implementing both schemes."""
        opt_type = option_type.lower()
        barr_type = barrier_type.lower()

        # For knock-in options, use in-out parity
        if "i" in barr_type:
            # Price knock-in using: knock-in + knock-out = vanilla
            from option_pricing.pricing import black_scholes_price
            
            # Price the vanilla option
            vanilla_price = black_scholes_price(
                spot=self.spot,
                strike=strike,
                maturity=self.maturity,
                rate=self.rate,
                volatility=self.volatility,
                dividend_yield=self.dividend_yield,
                option_type=opt_type,
            )
            
            # Convert knock-in to knock-out and price it
            knock_out_type = barr_type.replace("i", "o")
            knock_out_price = self._solve(opt_type, strike, knock_out_type, scheme)
            
            # Return knock-in price via parity
            return vanilla_price - knock_out_price

        # Initialize value at maturity (terminal condition)
        if opt_type == "call":
            V = np.maximum(self.S_grid - strike, 0.0)
        else:
            V = np.maximum(strike - self.S_grid, 0.0)

        # Apply barrier boundary conditions
        barrier_idx = self._find_barrier_index(barr_type)
        V = self._apply_barrier_condition(V, barr_type, barrier_idx)

        # Build finite difference matrices
        N = len(self.S_grid)
        alpha, beta, gamma = self._build_coefficients()

        # Construct tridiagonal matrices and solve backwards in time
        if scheme == "implicit":
            # Fully implicit: V^{n} = A^{-1} V^{n+1}
            A = self._build_implicit_matrix(alpha, beta, gamma)
            for step in range(self.time_steps):
                current_time = self.maturity - (step + 1) * self.dt
                V = self._time_step_implicit(V, A, barr_type, barrier_idx, opt_type, strike, current_time)
        else:  # crank-nicolson
            # Crank-Nicolson: (I - 0.5 dt L) V^{n} = (I + 0.5 dt L) V^{n+1}
            A_lhs = self._build_cn_lhs_matrix(alpha, beta, gamma)
            A_rhs = self._build_cn_rhs_matrix(alpha, beta, gamma)
            for step in range(self.time_steps):
                current_time = self.maturity - (step + 1) * self.dt
                V = self._time_step_crank_nicolson(
                    V, A_lhs, A_rhs, barr_type, barrier_idx, opt_type, strike, current_time
                )

        # Interpolate to spot price
        return float(np.interp(self.spot, self.S_grid, V))

    def _build_coefficients(self) -> tuple[NDArray, NDArray, NDArray]:
        """Build PDE coefficients for the discretization."""
        N = len(self.S_grid)
        sigma2 = self.volatility**2
        r = self.rate
        q = self.dividend_yield

        alpha = np.zeros(N)
        beta = np.zeros(N)
        gamma = np.zeros(N)

        for i in range(1, N - 1):
            S_i = self.S_grid[i]
            dS_forward = self.S_grid[i + 1] - self.S_grid[i]
            dS_backward = self.S_grid[i] - self.S_grid[i - 1]
            dS_central = 0.5 * (dS_forward + dS_backward)

            # Coefficients for second derivative
            coeff_second = 0.5 * sigma2 * S_i**2
            alpha[i] = coeff_second / (dS_backward * dS_central)
            gamma[i] = coeff_second / (dS_forward * dS_central)

            # Coefficient for first derivative
            coeff_first = (r - q) * S_i
            alpha[i] -= coeff_first / (2 * dS_central)
            gamma[i] += coeff_first / (2 * dS_central)

            # Diagonal term
            beta[i] = -alpha[i] - gamma[i] - r

        return alpha, beta, gamma

    def _build_implicit_matrix(
        self, alpha: NDArray, beta: NDArray, gamma: NDArray
    ) -> NDArray:
        """Build matrix for fully implicit scheme: (I - dt L)."""
        N = len(self.S_grid)
        diag_main = 1.0 - self.dt * beta[1:-1]
        diag_lower = -self.dt * alpha[2:-1]
        diag_upper = -self.dt * gamma[1:-2]

        A = diags(
            [diag_lower, diag_main, diag_upper],
            offsets=[-1, 0, 1],
            shape=(N - 2, N - 2),
            format="csr",
        )
        return A

    def _build_cn_lhs_matrix(
        self, alpha: NDArray, beta: NDArray, gamma: NDArray
    ) -> NDArray:
        """Build LHS matrix for Crank-Nicolson: (I - 0.5 dt L)."""
        N = len(self.S_grid)
        diag_main = 1.0 - 0.5 * self.dt * beta[1:-1]
        diag_lower = -0.5 * self.dt * alpha[2:-1]
        diag_upper = -0.5 * self.dt * gamma[1:-2]

        A = diags(
            [diag_lower, diag_main, diag_upper],
            offsets=[-1, 0, 1],
            shape=(N - 2, N - 2),
            format="csr",
        )
        return A

    def _build_cn_rhs_matrix(
        self, alpha: NDArray, beta: NDArray, gamma: NDArray
    ) -> NDArray:
        """Build RHS matrix for Crank-Nicolson: (I + 0.5 dt L)."""
        N = len(self.S_grid)
        diag_main = 1.0 + 0.5 * self.dt * beta[1:-1]
        diag_lower = 0.5 * self.dt * alpha[2:-1]
        diag_upper = 0.5 * self.dt * gamma[1:-2]

        A = diags(
            [diag_lower, diag_main, diag_upper],
            offsets=[-1, 0, 1],
            shape=(N - 2, N - 2),
            format="csr",
        )
        return A

    def _time_step_implicit(
        self,
        V: NDArray,
        A: NDArray,
        barrier_type: str,
        barrier_idx: int,
        option_type: str,
        strike: float,
        current_time: float,
    ) -> NDArray:
        """Perform one implicit time step."""
        N = len(V)
        rhs = V[1:-1].copy()

        # Boundary conditions
        rhs[0] -= (-self.dt * self._build_coefficients()[0][1]) * V[0]
        rhs[-1] -= (-self.dt * self._build_coefficients()[2][-2]) * V[-1]

        # Solve
        V_interior = spsolve(A, rhs)
        V_new = np.concatenate(([V[0]], V_interior, [V[-1]]))

        # Update boundaries first
        V_new = self._update_boundaries(V_new, option_type, strike, barrier_type, barrier_idx, current_time)

        # Apply barrier condition (this should override boundary updates if needed)
        V_new = self._apply_barrier_condition(V_new, barrier_type, barrier_idx)

        return V_new

    def _time_step_crank_nicolson(
        self,
        V: NDArray,
        A_lhs: NDArray,
        A_rhs: NDArray,
        barrier_type: str,
        barrier_idx: int,
        option_type: str,
        strike: float,
        current_time: float,
    ) -> NDArray:
        """Perform one Crank-Nicolson time step."""
        N = len(V)
        rhs = A_rhs @ V[1:-1]

        # Boundary contributions
        alpha, beta, gamma = self._build_coefficients()
        rhs[0] -= (0.5 * self.dt * alpha[1]) * V[0]
        rhs[-1] -= (0.5 * self.dt * gamma[-2]) * V[-1]

        # Solve
        V_interior = spsolve(A_lhs, rhs)
        V_new = np.concatenate(([V[0]], V_interior, [V[-1]]))

        # Update boundaries first
        V_new = self._update_boundaries(V_new, option_type, strike, barrier_type, barrier_idx, current_time)

        # Apply barrier condition (this should override boundary updates if needed)
        V_new = self._apply_barrier_condition(V_new, barrier_type, barrier_idx)

        return V_new

    def _find_barrier_index(self, barrier_type: str) -> int:
        """Find the grid index closest to the barrier."""
        return int(np.argmin(np.abs(self.S_grid - self.barrier)))

    def _apply_barrier_condition(
        self, V: NDArray, barrier_type: str, barrier_idx: int
    ) -> NDArray:
        """Apply knock-out boundary condition at barrier."""
        V_new = V.copy()
        if barrier_type in ("do",):
            # Down-and-out: set values at and below barrier to zero
            V_new[: barrier_idx + 1] = 0.0
        elif barrier_type in ("uo",):
            # Up-and-out: set values at and above barrier to zero
            V_new[barrier_idx:] = 0.0
        # Knock-in options: no special barrier condition during backward solve
        # They are handled via in-out parity
        return V_new

    def _update_boundaries(
        self, V: NDArray, option_type: str, strike: float, barrier_type: str, barrier_idx: int, current_time: float
    ) -> NDArray:
        """Update boundary conditions at S_min and S_max."""
        V_new = V.copy()
        
        # Discount factor from current time to maturity
        discount = np.exp(-self.rate * current_time)

        # Lower boundary (S → 0)
        # Don't override if the barrier is at the lower boundary for down knock-out
        if barrier_type == "do" and barrier_idx <= 1:
            # Barrier is at lower boundary, keep it at zero
            pass
        else:
            if option_type == "call":
                V_new[0] = 0.0
            else:  # put
                # Put value at S=0 is PV(K) = K * e^{-r*tau}
                V_new[0] = strike * discount

        # Upper boundary (S → ∞)
        # Don't override if the barrier is at the upper boundary for up knock-out
        if barrier_type == "uo" and barrier_idx >= len(V) - 2:
            # Barrier is at upper boundary, keep it at zero
            pass
        else:
            if option_type == "call":
                # Call value at S=infinity is S - PV(K) = S - K * e^{-r*tau}
                V_new[-1] = self.S_grid[-1] - strike * discount
            else:  # put
                V_new[-1] = 0.0

        return V_new


def solve_barrier_pde(
    option_type: str,
    barrier_type: str,
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    grid_points: int = 100,
    time_steps: int = 100,
    scheme: Literal["implicit", "crank-nicolson"] = "crank-nicolson",
) -> float:
    """
    Price a barrier option using finite difference PDE methods.

    Convenience function that creates a solver and prices the option.

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
    >>> solve_barrier_pde("call", "uo", 100, 100, 120, 1.0, 0.05, 0.25,
    ...                   grid_points=200, time_steps=200)
    7.964401294924583
    """
    solver = Pde1DSolver(
        spot=spot,
        barrier=barrier,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        grid_points=grid_points,
        time_steps=time_steps,
    )

    if scheme == "implicit":
        return solver.solve_implicit(option_type, strike, barrier_type)
    else:
        return solver.solve_crank_nicolson(option_type, strike, barrier_type)
