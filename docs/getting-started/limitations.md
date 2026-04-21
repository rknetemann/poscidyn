Poscidyn has several limitations. Please review them before using the library for your application.

## Physics limitations

### Transient response estimation

Time-integration solvers estimate the duration of the transient response using an approximation derived from a **linear single-degree-of-freedom (SDOF)** system. This estimate may therefore be inaccurate for **nonlinear multi-degree-of-freedom (MDOF)** systems. In practice, the estimate works well for many parameter combinations and system sizes. If you suspect that this estimate is too low, `TimeIntegration` allows setting the `t_steady_state_factor`.

For the `Nonlinear` model, the estimated time to steady state used for time integration is

$$
\tau_d =
-2Q
\ln\!\left(
\frac{d\sqrt{1-\frac{1}{4Q^2}}}{\omega_d}
\right)
$$

**Reference**

[1] B. Balachandran and E. B. Magrab, *Vibrations*, Second edition (Cengage Learning, Australia, 2009).


### Synthetic frequency sweeps

Synthetic frequency sweeps implemented in Poscidyn do not fully reproduce the physics of real quasi-static experiments. Instead of simulating a continuous frequency sweep, the solver evaluates multiple initial conditions independently in parallel and applies a nearest-neighbour selection to approximate experiments where only one solution is observed per frequency. Consequently, the simulated sweep may differ from a true quasi-static experiment.

### Multistart initial-condition range

The built-in multistart method generates a grid of initial conditions to explore stable solution branches. When using `LinearResponse`, the estimated range of initial conditions may be inaccurate because nonlinearities can significantly increase the oscillation amplitude.

In such cases, manual tuning of the parameter `linear_response_factor` in `LinearResponse` may be required.

---

## Computational limitations

### Batched execution using `jax.vmap`

The term *parallel* in this package refers to **batched execution** using `jax.vmap`, not embarrassingly parallel execution. `vmap` fuses multiple function evaluations into a single compiled kernel.

Because the computation is executed as a batched operation, intermediate arrays are also batched. As a result, memory usage scales approximately **linearly with the batch size**.

Increasing the multistart parameter `n_init_cond` therefore increases memory usage and may eventually exceed the available CPU or GPU memory.

### Compilation overhead

Poscidyn relies heavily on Just-In-Time (JIT) compilation provided by `jax.jit`. Before a simulation can be executed, JAX traces the computation graph and compiles it into an optimized kernel. This compilation step can take a noticeable amount of time, depending on the complexity of the model and the size of the computational graph.

As a result, the first execution of a simulation may be significantly slower due to this compilation overhead. Subsequent executions with the same function structure and input shapes are typically much faster because the compiled kernel can be reused.

Because of this compilation requirement, the package is generally less efficient for single, isolated simulations. The performance benefits of JAX are most pronounced when running many simulations using `jax.vmap`.