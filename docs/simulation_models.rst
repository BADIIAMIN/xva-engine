Simulation Models
=================

This section documents the stochastic risk-factor models used to generate
Monte Carlo trajectories (scenarios) for XVA / CCR / PFE.

The repository distinguishes:
- *Market data simulators* under ``xva_engine/market_data/simulators``:
  produce synthetic market data snapshots (fixtures/dummies), not trajectories.
- *Scenario generators* under ``xva_engine/simulation``:
  generate time-dependent trajectories from calibrated stochastic models.

.. toctree::
   :maxdepth: 2

   simulation_models/conventions
   simulation_models/ir_ultimate_base_curve
   simulation_models/troubleshooting
