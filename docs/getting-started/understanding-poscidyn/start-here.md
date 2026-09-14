# Multistart and synthetic sweeps

Frequency sweeps in Poscidyn are built from independent time integrations,
then assembled into selected paths. This is different from numerical
continuation, where each solution is explicitly initialized from its immediate
predecessor.

## Why use multistart?

At one drive frequency, nonlinear systems may admit several stable responses
with different basins of attraction. `LinearResponse` generates a reproducible
set of initial displacement and velocity candidates for each requested
frequency. `TimeIntegration` integrates them in a JAX batch, then applies the
configured response measure.

This improves the chance of finding different attracting responses, but it does
not prove exhaustive branch discovery. Coverage depends on the number and range
of initial conditions.

## Why select a synthetic path?

Experiments often reveal one response as frequency rises or falls. Once many
candidates have been computed independently, `NearestNeighbour` selects a
forward or backward path by preferring nearby response amplitude and phase and
discouraging unnecessary seed changes.

The result can be useful for experiment-like plots and parameter studies. It is
not a continuation method, a stability computation, or a replacement for
bifurcation analysis. Use the [nearest-neighbour guide](artificial-sweeps/nearest-neighbour.md)
for the selection rule and [limitations](../limitations.md) for the caveats.

## What to tune first

1. Validate one representative case with `solver.time_response(...)`.
2. Set enough retained samples for the response features you need.
3. Increase `n_init_cond` and adjust `linear_response_factor` to explore
   candidate coverage.
4. Compare forward and backward paths and check `result.stats`.

This is the technical background behind the [first frequency sweep](../../../quickstart/frequency-sweep.md).
