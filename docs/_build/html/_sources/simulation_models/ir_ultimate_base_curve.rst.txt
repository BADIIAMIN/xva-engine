IR Ultimate Base Curve Model
============================

Module coverage
---------------
The following modules are implemented for interest-rate trajectory simulation:

- Generator (engine-facing):
  ``xva_engine.simulation.generators.ir_ultimate_base_curve_generator``

- IR risk-factor process (mathematics):
  ``xva_engine.simulation.risk_factors.ir.ultimate_base_curve_process``

- Deterministic mean function builder:
  ``xva_engine.simulation.risk_factors.ir.mean_function``

- Historical calibration utilities:
  ``xva_engine.simulation.risk_factors.ir.calibration_historical``

- Calibration orchestration (if used):
  ``xva_engine.simulation.risk_factors.ir.calibrators``

Purpose and scope
-----------------
This model generates Monte Carlo trajectories for the *ultimate base interest-rate curve*
represented as continuous zero rates at fixed maturity pillars.

The design goal is to provide:
- deterministic mean consistency with the initial discount curve,
- per-pillar stochastic dynamics with cross-pillar correlation,
- a clean separation between process mathematics and scenario generation.

Architecture placement
----------------------
This is a trajectory model and belongs under ``xva_engine/simulation`` (not under
``xva_engine/market_data/simulators``).

Model definition
----------------

Pillars
~~~~~~~
Let ``k = 1..K`` index curve maturity pillars. Pillars are stored as ``M_k`` (in days)
and converted to year fractions ``m_k = M_k / 365``.

OU drivers (per pillar)
~~~~~~~~~~~~~~~~~~~~~~~
For each pillar ``k`` define an OU driver:

.. math::

   dX_k(t) = -\lambda_k X_k(t)\,dt + \sigma_k\,dW_k(t),
   \qquad X_k(0)=0.

Cross-pillar correlation
~~~~~~~~~~~~~~~~~~~~~~~~
Brownian motions are correlated across pillars:

.. math::

   dW_i(t)\,dW_j(t) = \rho_{ij}\,dt.

In implementation, a Cholesky factor ``L`` of ``rho`` is used to correlate
standard normal increments.

Deterministic mean function (forward-forward)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A deterministic mean function ``g_k(t)`` is built from the initial discount curve ``DF(0,t)``
using a forward-forward rate:

.. math::

   f_k(t) = -\frac{1}{m_k}\ln\left(\frac{DF(0,t+m_k)}{DF(0,t)}\right),
   \qquad g_k(t) = \max(f_k(t), \delta).

Shifted exponential transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The simulated zero rate is obtained via:

.. math::

   Y_k(t) = (g_k(t) + s_k)\exp\left(X_k(t) - \tfrac12 v_k^2(t)\right) - s_k,

where ``s_k`` is a per-pillar shift (in rate units) and:

.. math::

   v_k^2(t) = \mathrm{Var}[X_k(t)]
           = \sigma_k^2 \frac{1-e^{-2\lambda_k t}}{2\lambda_k}.

This ensures mean consistency:

.. math::

   \mathbb{E}[Y_k(t)] = g_k(t).

Discretization (exact OU)
-------------------------
The OU drivers are evolved using the exact discretization over a time step ``Δt``:

.. math::

   X_k(t+\Delta t) = e^{-\lambda_k\Delta t}X_k(t)
   + \sqrt{ \sigma_k^2 \frac{1-e^{-2\lambda_k\Delta t}}{2\lambda_k} }\,Z_k,

where ``Z`` is a correlated standard normal vector.

Calibration (implemented to date)
---------------------------------
A minimal historical calibration is implemented to estimate:

- cross-pillar correlation matrix ``rho`` from shifted log-returns,
- per-pillar volatility ``sigma_k`` from shifted log-return variance,
- mean reversion ``lambda_k`` is currently taken as configured (scalar expanded across pillars),
- shift ``s_k`` is currently configured (scalar or per pillar).

Shifted log-returns (horizon h days)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With a shift ``s`` ensuring positivity:

.. math::

   r_k(t) = \ln\left(\frac{Y_k(t+h)+s}{Y_k(t)+s}\right).

Empirical correlation of ``r_k`` provides ``rho`` and variance provides a mapping to ``sigma_k``.

Scenario generation workflow
----------------------------
End-to-end scenario generation proceeds as:

1. Build deterministic mean function ``g(t,k)`` from ``DF(0,t)``.
2. Obtain parameters ``(rho, sigma_k, lambda_k, s_k)`` from calibration/config.
3. Simulate OU drivers ``X(t,k)`` with correlated increments.
4. Transform to zero rates ``Y(t,k)`` via shifted exponential mapping.
5. Output scenarios as cube ``rates[path, time, pillar]``.

Outputs
-------
- ``rates``: NumPy array with shape ``(n_paths, n_times, n_pillars)``
- optional ``driver`` (if enabled): same shape, holding ``X`` for diagnostics.

Recommended validation checks
-----------------------------
- Mean consistency: verify empirical mean across paths satisfies
  ``avg_paths(Y_k(t)) ≈ g_k(t)``.
- Positivity under shift: ``Y_k(t) + s_k > 0`` for all ``t`` and ``k`` (required if using log-returns).
- SPD correlation: verify Cholesky of ``rho`` succeeds (or apply SPD repair).
- Long-horizon variance diagnostic: compare achieved variance of rates versus target implied by parameters.

Planned enhancements
--------------------
- Full shift selection algorithm (per your model documentation) enforcing stability/positivity conditions.
- Market-implied calibration to swaption vol targets.
- Benchmarking models (e.g. Hull-White 1F, PCA/Gaussian term-structure model).
