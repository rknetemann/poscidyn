# Is Poscidyn for me?

Poscidyn is for researchers and engineers who model nonlinear oscillators in
Python and want efficient time-domain simulation or experiment-like
frequency-response curves. It is especially useful when many related
simulations make JAX batching worthwhile.

It is a good fit when you:

- work with modal oscillator models and harmonic excitation;
- need trajectories, response amplitudes, phases, or scalar response measures;
- want to explore sensitivity to initial conditions and possible coexisting
  attracting responses;
- value a compact Python workflow that can run on JAX-supported CPU or GPU
  backends.

It is not yet the right primary tool when you need certified branch following,
bifurcation detection, stability/Floquet analysis, non-smooth hybrid dynamics,
or a broad library of excitation types. Those are active design directions, not
supported promises. See [status and roadmap](future-work.md).

Start with the [first frequency sweep](quickstart/frequency-sweep.md) if this
matches your use case. Read [limitations](getting-started/limitations.md)
before using a selected synthetic path as a scientific result.
