# The Poscidyn workflow

Poscidyn separates a nonlinear-dynamics study into components with distinct
responsibilities. This keeps the everyday workflow small while leaving room to
change one numerical decision at a time.

```text
Oscillator + excitation + response measure
                    │
                    ▼
          TimeIntegration solver
             │              │
             ▼              ▼
      time_response     frequency_sweep
                                │
                                ▼
                  multistart candidates
                                │
                                ▼
                 synthetic branch selection
```

## 1. Describe the system

`NonlinearOscillator` represents modal equations with linear damping and
stiffness plus quadratic and cubic stiffness tensors. An excitation supplies
the external term. The supported guide path uses
`DirectHarmonicExcitation`, a single harmonic drive with modal force vector
`f_d`, optional scaling factors `lambdas`, and an `omega` for time responses.

The oscillator and excitation belong to the solver when it is constructed.
They are not passed again on each solve call.

## 2. Choose the output you need

Call `solver.time_response(x0, v0)` to retrieve sampled displacement and
velocity trajectories from one initial state. Use it to inspect transients,
phase portraits, or a single steady regime.

Call `solver.frequency_sweep(omegas)` to explore a range of drive frequencies.
For each frequency Poscidyn integrates multiple initial states, applies a
response measure to the retained trajectory, then asks the synthetic-sweep
strategy to select forward and backward paths.

## 3. Treat selected paths as synthetic sweeps

The nearest-neighbour strategy chooses the candidate that changes least from
the preceding selected response. This is useful when you want an
experiment-like curve from independently batched simulations. It is not
classical continuation, a stability calculation, or a guarantee that every
coexisting solution has been discovered.

That distinction matters most near jumps, bifurcations, narrow basins of
attraction, and strongly nonlinear resonances. Increase and validate the
multistart sampling before interpreting such features. The detailed
[limitations](../getting-started/limitations.md) describe the physical and
computational trade-offs.

## 4. Read results deliberately

`FrequencySweep` has `frequency`, `forward`, `backward`, and `stats` fields.
Each branch contains a `modal` response and a weighted `total` response. For
demodulation, amplitude, phase, and response-frequency information are
available; scalar measures expose their value as `amplitude` for a uniform
plotting interface.

See [interpreting sweep results](../guides/interpreting-results.md) for
concrete shapes and checks.
