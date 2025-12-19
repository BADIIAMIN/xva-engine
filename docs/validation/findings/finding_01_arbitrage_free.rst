Finding 1: Absence of Pathwise Arbitrage-Free Term-Structure Dynamics
====================================================================

Finding statement
-----------------
The Interest Rate (IR) simulation model does **not enforce pathwise arbitrage-free**
relationships across maturities. Each zero-rate maturity pillar is simulated as a
distinct stochastic process, with cross-tenor dependence introduced exclusively via
correlation between latent drivers.

As a result, discount-factor multiplicative identities and forward-rate reconstruction
conditions are not guaranteed to hold along individual Monte Carlo paths, particularly
at long horizons or under elevated volatility regimes.

This limitation may lead to economically implausible curve shapes and dynamic
inconsistencies in simulated interest-rate term structures.

Model mechanics leading to the finding
--------------------------------------
In the implemented framework, each maturity pillar is simulated through a shifted
exponential Vasicek-type construction:

.. math::

   Y_k(t)
   =
   (g_k(t)+s_k)\exp\!\left(X_k(t) - \tfrac12 v_k^2(t)\right) - s_k,

with per-pillar OU drivers (correlated across pillars). The model does not impose a
short-rate representation, an instantaneous forward-rate model, nor HJM drift
restrictions, hence does not guarantee joint curve consistency along paths.

Theory corroboration
--------------------
In arbitrage-free short-rate or HJM frameworks, discount factors satisfy the pathwise
multiplicative identity:

.. math::

   DF(t,t+T) = DF(t,t+u)\cdot DF(t+u,t+T), \qquad \forall\ 0<u<T.

Because the current model infers discount factors at time :math:`t` from the simulated curve
at :math:`t`, and discount factors at :math:`t+u` from an independently simulated curve at
:math:`t+u`, no structural constraint enforces this identity, and violations are expected.

Practical implications for PFE
------------------------------
This limitation may result in:

- static inconsistencies (e.g. non-monotone discount factors at a snapshot),
- dynamic inconsistencies (long-horizon discount factors not matching compounded shorter intervals),
- potential bias for long-dated portfolios where pathwise discounting matters.

Validation tests and diagnostics
--------------------------------

Test 1A — Static discount-factor monotonicity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Objective:
  Verify absence of static arbitrage at each simulation time.

Definition:
  For maturity pillars :math:`M_k` (year fractions),

.. math::

   DF(t_i,t_i+M_k) := \exp(-Y_k(t_i)\,M_k),

and the condition:

.. math::

   DF(t_i,t_i+M_{k+1}) \le DF(t_i,t_i+M_k), \quad \forall k.

Outputs:
  Violation frequency, max breach severity, time×pillar heatmap.

Test 1B — Forward-rate reconstruction sanity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Objective:
  Ensure implied forwards between pillars remain stable and plausible.

Definition for :math:`M_i<M_j`:

.. math::

   F_{i,j}(t)
   =
   \frac{Y_j(t)M_j - Y_i(t)M_i}{M_j - M_i}.

Outputs:
  Distributional diagnostics and curvature/kink metrics.

Test 1C — Pathwise DF multiplicative wedge (core test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Objective:
  Quantify failure of pathwise multiplicative discounting identity.

Definition:

.. math::

   \mathrm{Wedge}(t,u,T)
   =
   \log DF(t,t+T) - \log DF(t,t+u) - \log DF(t+u,t+T).

Outputs:
  Wedge distribution, percentile bands, exceedance rates.

Illustrations
-------------
Recommended figures produced by the demo script:

- DF monotonicity violation heatmaps
- kink index bands (median and 5/95%)
- wedge histograms (long maturity)
- wedge magnitude vs maturity (e.g. p95(|wedge|))

How to run
----------
From repo root:

.. code-block:: bash

   python -m xva_engine.scripts.run_ir_validations_and_plots_demo

Outputs are saved under:

.. code-block:: text

   outputs/finding1/
