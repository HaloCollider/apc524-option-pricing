from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MeanVarCalculator:
    """Incremental mean and variance calculator.

    Parameters
    ----------
    n_vars : int, default=1
        Number of variables tracked in parallel. Each call to :meth:`add_sample`
        must provide a scalar or a 1D array of length ``n_vars``.
    """

    def __init__(self, n_vars: int = 1) -> None:
        if n_vars <= 0:
            raise ValueError("n_vars must be positive.")
        self._n_vars = int(n_vars)
        self._count: int = 0
        self._mean: NDArray[np.float64] | None = None
        self._m2: NDArray[np.float64] | None = None

    @property
    def n_vars(self) -> int:  # pragma: no cover - trivial accessor
        return self._n_vars

    @property
    def count(self) -> int:  # pragma: no cover - trivial accessor
        return self._count

    def add_sample(self, values: ArrayLike) -> None:
        """Add a new observation.

        Parameters
        ----------
        values : array_like
            Scalar or 1D array of length ``n_vars``.
        """

        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size != self._n_vars:
            raise ValueError("values must have length equal to n_vars.")

        if self._mean is None:
            self._mean = np.array(arr, dtype=np.float64)
            self._m2 = np.zeros_like(self._mean)
            self._count = 1
            return

        self._count += 1
        delta = arr - self._mean
        self._mean = self._mean + delta / self._count
        self._m2 = self._m2 + delta * (arr - self._mean)

    def results(self) -> dict[str, NDArray[np.float64]]:
        """Return mean, variance, and standard deviation.

        Returns
        -------
        dict
            Dictionary with keys ``"mean"``, ``"variance"``, and ``"std"``.
        """

        if self._mean is None or self._m2 is None or self._count == 0:
            zeros = np.zeros(self._n_vars, dtype=np.float64)
            return {"mean": zeros, "variance": zeros, "std": zeros}

        if self._count > 1:
            var = self._m2 / (self._count - 1)
        else:
            var = np.zeros_like(self._mean)
        std = np.sqrt(var)
        return {"mean": self._mean.copy(), "variance": var, "std": std}


class HistogramCalculator:
    """Empirical histogram calculator for one or more variables.

    Parameters
    ----------
    bin_edges : array_like
        Monotonically increasing sequence of bin boundaries, including both
        left and right edges.
    n_vars : int, default=1
        Number of variables tracked in parallel. Each call to :meth:`add_sample`
        must provide a scalar or 1D array of length ``n_vars``.
    """

    def __init__(self, bin_edges: ArrayLike, n_vars: int = 1) -> None:
        edges = np.asarray(bin_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must be a 1D array with at least two elements.")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges must be strictly increasing.")
        if n_vars <= 0:
            raise ValueError("n_vars must be positive.")

        self.bin_edges: NDArray[np.float64] = edges
        self._n_vars = int(n_vars)
        n_bins = edges.size - 1
        self._counts: NDArray[np.int64] = np.zeros((n_bins, self._n_vars), dtype=np.int64)

    @property
    def n_bins(self) -> int:  # pragma: no cover - trivial accessor
        return self._counts.shape[0]

    @property
    def n_vars(self) -> int:  # pragma: no cover - trivial accessor
        return self._n_vars

    @property
    def counts(self) -> NDArray[np.int64]:  # pragma: no cover - trivial accessor
        return self._counts

    def add_sample(self, values: ArrayLike) -> None:
        """Add a new observation and update bin counts.

        Parameters
        ----------
        values : array_like
            Scalar or 1D array of length ``n_vars``.
        """

        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size != self._n_vars:
            raise ValueError("values must have length equal to n_vars.")

        # Compute bin indices using the same convention as numpy.histogram:
        # bins are [left, right), with the last bin including the right edge.
        idx = np.searchsorted(self.bin_edges, arr, side="right") - 1
        # Clamp to the valid range to keep extreme values in the edge bins.
        idx = np.clip(idx, 0, self.n_bins - 1)

        for j, bin_index in enumerate(idx):
            self._counts[int(bin_index), j] += 1

    def results(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return histogram counts and bin edges.

        Returns
        -------
        counts : numpy.ndarray
            2D array of shape ``(n_bins, n_vars)`` containing bin counts.
        bin_edges : numpy.ndarray
            1D array of bin boundaries.
        """

        return self._counts.copy(), self.bin_edges.copy()
