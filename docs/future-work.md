# Future work

Poscidyn is still under active development. This page describes possible enhancements and longer-term directions; it is not a release commitment.

## Planned enhancements

- **Additional solvers**
  Add shooting, collocation, and possibly standard continuation methods while preserving efficient batching. A hybrid strategy could segment a sweep and use continuation within smaller batches to reduce memory use.

- **Oscillator models**
  Expand the library with canonical systems such as Van der Pol, Lorenz, and Rayleigh oscillators. Improve the component interface so users can define custom dynamical systems without having to understand solver internals.

- **Parametric excitation**
  Complete end-to-end solver support for the existing `ParametricExcitation` class.

- **Visualization tools**
  Add plotting utilities for phase-space trajectories, frequency-response curves, and time-domain responses.

- **Sweep methods**
  Develop physically motivated synthetic sweep strategies that better emulate experimental frequency sweeps.

## Long-term ideas

- **Hybrid approaches**
  Combine multistart batching with localized continuation, for example by segmenting sweeps and performing parallel continuations.

## Contributing

See [Extending Poscidyn](usage/extending-poscidyn.md) for current component interfaces and the local documentation build workflow.
