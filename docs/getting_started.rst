===============
Getting Started
===============

Overview
========

This project implements a modular framework for:

* Risk factor simulation under both risk–neutral and historical measures.
* Pricing vanilla and exotic payoffs on simulated scenarios.
* Collateral and exposure aggregation (EE, EPE, PFE, EEPE).
* XVA computation (CVA, DVA, FVA, MVA, CollVA).
* Backtesting, benchmarking and validation.

Installation
============

.. code-block:: bash

   git clone https://github.com/your_org/xva_engine.git
   cd xva_engine
   pip install -e .

   # build docs
   cd docs
   make html

Minimal Example
===============

The file :mod:`xva_engine.orchestration.workflows` contains a
minimal vertical slice that:

1. Simulates a single GBM equity risk factor.
2. Prices a European call option by Monte Carlo.
3. Computes Expected Exposure (EE) and 95% PFE.

Example usage:

.. code-block:: python

   from xva_engine.orchestration.workflows import run_single_equity_option_demo

   params = {
       "spot": 100.0,
       "rate": 0.02,
       "sigma": 0.20,
       "maturity": 1.0,
       "strike": 100.0,
       "n_scenarios": 10000,
       "n_steps": 50,
       "t_max": 1.0,
   }

   result = run_single_equity_option_demo(params)

   print("EE at maturity:", result.ee[-1, 0])
   print("PFE(95%) at maturity:", result.pfe_95[-1, 0])

The next sections describe the architecture and APIs in detail.
