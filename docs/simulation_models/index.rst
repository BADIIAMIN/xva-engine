Simulation Models
=================

This section documents the stochastic risk-factor simulation framework implemented
under ``xva_engine/simulation``.

Important naming clarification
------------------------------
The folder ``xva_engine/market_data/simulators`` is reserved for generating *synthetic*
market data objects (fixtures/dummy datasets), not time-dependent Monte Carlo trajectories.

Monte Carlo scenario generation is implemented under:

- ``xva_engine/simulation``

.. toctree::
   :maxdepth: 2

   conventions
   ir_ultimate_base_curve
   api_ir
