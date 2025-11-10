.. option-pricing documentation master file

Option Pricing Library
======================

An educational Python library implementing analytical Black-Scholes-Merton valuations and Monte Carlo simulation approaches for European and digital options.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api


Getting Started
---------------

Install dependencies and run tests with ``uv``::

   uv sync --all-groups
   uv run pytest -q

API Overview
------------

The core public functions are:

* ``black_scholes_price`` – Analytical European call/put option pricing.
* ``black_scholes_greeks`` – Delta, gamma, vega, theta, rho in one pass.
* ``simulate_gbm_paths`` – Geometric Brownian motion path simulation.
* ``monte_carlo_european_price`` – Monte Carlo estimator for vanilla European options.
* ``monte_carlo_digital_price`` – Monte Carlo estimator for digital cash-or-nothing options.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
