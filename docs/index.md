# Poscidyn

**Fast, batched simulation of nonlinear oscillator dynamics in Python.**

Poscidyn is a JAX-based toolkit for nonlinear time responses and frequency
sweeps. Its current workflow combines direct time integration, many initial
conditions, and an artificial sweep-selection method to make branch-like
frequency-response curves practical on accelerator-oriented hardware.

It is an alpha-stage scientific package. The supported public workflow is the
**solver-instance API**: construct a model, excitation, and `TimeIntegration`
solver, then call `solver.time_response(...)` or
`solver.frequency_sweep(...)`.

## Start here

1. [Install Poscidyn](getting-started/installation.md).
2. Run the [first frequency sweep](quickstart/frequency-sweep.md) to produce a
   response curve.
3. Read [the workflow](concepts/workflow.md) before tuning multistart or
   interpreting branches.

## What is available today?

- nonlinear modal oscillator models with quadratic and cubic stiffness terms;
- direct harmonic excitation;
- time-domain responses and batched frequency sweeps with `TimeIntegration`;
- linear-response multistart, nearest-neighbour artificial sweeps, and
  demodulated, RMS, minimum, or maximum response measures.

## A deliberately narrow first example

```python
import jax.numpy as jnp
import poscidyn

oscillator = poscidyn.NonlinearOscillator(
    omega_0=jnp.array([1.0]),
    Q=jnp.array([80.0]),
    a=jnp.zeros((1, 1, 1)),
    b=jnp.array([[[[0.2]]]]),
)
excitation = poscidyn.DirectHarmonicExcitation(f_d=jnp.array([0.002]))
solver = poscidyn.TimeIntegration(
    oscillator=oscillator,
    excitation=excitation,
    response_measure=poscidyn.Demodulation(),
    n_time_steps=64,
    multistart=poscidyn.LinearResponse(n_init_cond=8),
)
result = solver.frequency_sweep(jnp.linspace(0.8, 1.2, 100))
```

The quickstart explains each line and shows how to plot `result`.

## Use Poscidyn with care

The selected curves are synthetic approximations to quasi-static frequency
sweeps, not a replacement for continuation or a proof that every solution
branch has been found. Batch size also has a direct memory cost. Read
[accuracy, performance, and limitations](getting-started/limitations.md)
before relying on results for a scientific conclusion.

## Looking ahead

Poscidyn is intentionally designed to grow. Collocation, shooting,
continuation/hybrid methods, hybrid dynamics, and new excitation families are
important directions, but they are not part of the supported API yet. See
[status and roadmap](future-work.md) for the boundary between available work,
experiments, and plans.
