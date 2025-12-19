Conventions
===========

Repository structure (simulation layer)
---------------------------------------
The simulation layer is organized as follows:

- ``xva_engine/simulation/driver.py``:
  orchestration and driving logic (time grids, scenario runs, calling generators).
- ``xva_engine/simulation/schemes.py``:
  numerical schemes / stepping utilities used by processes.
- ``xva_engine/simulation/generators``:
  scenario generators (engine-facing wrappers producing scenario cubes).
- ``xva_engine/simulation/risk_factors``:
  mathematical processes and calibration components, by asset class.

Time representation
-------------------
- Simulation time is expressed in *year fractions* (typically ACT/365).
- Curve pillars are stored in *days* and converted to years when needed.

Units
-----
- Rates are in *rate units* (not basis points).
- Basis points conversion: ``1 bp = 1e-4``.

Scenario cube layout
--------------------
Scenario cubes are produced as NumPy arrays with shape:

- ``rates[path_index, time_index, pillar_index]``

Correlation
-----------
Cross-pillar dependency is represented by a correlation matrix ``corr``.
Correlated standard normals are generated using Cholesky factorization.

Responsibilities
----------------
- *Risk-factor processes* implement state evolution and transformations.
- *Generators* combine deterministic mean functions, calibrated parameters,
  and the process to produce scenario cubes consumable by the engine.
