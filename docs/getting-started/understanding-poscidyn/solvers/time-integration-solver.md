[<-- Go back to solvers](../start-here.md#solvers)

# Time integration solver

`TimeIntegrationSolver` is the default solver in Poscidyn. It computes responses by integrating the equations of motion in time and then extracting the steady-state part of the trajectory.

## Core idea

The solver rewrites the oscillator as a first-order system

$$
\dot{\mathbf{y}}(t) = \mathbf{f}(\mathbf{y}(t), t),
\qquad
\mathbf{y}(0) = \mathbf{y}_0,
$$

with state \(\mathbf{y} = [\mathbf{x}, \mathbf{v}]\), where \(\mathbf{x}\) contains modal displacements and \(\mathbf{v}\) modal velocities.

Poscidyn then:

1. estimates a transient duration using `model.t_steady_state(...)`,
2. multiplies that estimate by `t_steady_state_factor`,
3. integrates beyond the transient,
4. retains the final 10 drive periods,
5. evaluates the chosen response measure on that retained window.

## Numerical method

The implementation uses:

- `diffrax.Tsit5()` as the ODE integrator,
- adaptive step-size control through `diffrax.PIDController`,
- tolerances `rtol` and `atol`,
- a hard `max_steps` limit per solve.

For frequency sweeps, trajectories with non-finite states are marked as unsuccessful and excluded from the final sweep statistics.

## Sampling

`n_time_steps` controls how densely the retained response window is sampled.

- If you set it explicitly, that value is used directly.
- If you leave it as `None`, Poscidyn estimates a suitable value from the highest expected frequency component.
- Inside traced or JIT-compiled workflows, `n_time_steps` must be set explicitly.

For `time_response`, you can also pass `only_save_steady_state=True` to save only the final steady-state portion instead of the full transient.

## Parameters

- `rtol`, `atol`: relative and absolute tolerances for the adaptive integrator.
- `n_time_steps`: number of saved samples per retained period block.
- `max_steps`: maximum number of internal solver steps allowed per trajectory.
- `t_steady_state_factor`: safety factor applied to the model's steady-state time estimate.
- `throw`: passed to Diffrax. If `True`, integration failures raise immediately.
- `verbose`: reserved on the class but currently not used in the implementation.
