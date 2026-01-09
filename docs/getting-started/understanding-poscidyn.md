To get started with **Poscidyn**, there are only a few core concepts to understand. Poscidyn provides classes for selecting **oscillator models** (for example the [Duffing oscillator](https://en.wikipedia.org/wiki/Duffing_equation) or the [Van der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)), **excitation models** (for example single-tone or multi-tone excitation), and several advanced components such as **solvers** (e.g. time integration or shooting-method solvers), **sweep methods** (used to imitate continuation), and **multistart methods** (used to explore different initial conditions).

These advanced classes may seem abstract at first. Therefore, it is important to first understand how Poscidyn operates conceptually, and how it compares to established continuation and bifurcation analysis software such as [AUTO](https://sourceforge.net/projects/auto-07p/), [MATCONT](https://sourceforge.net/projects/matcont/), and [COCO](https://sourceforge.net/projects/cocotools/).

## Numerical frequency sweeping

Frequency sweeps performed in an experimental setting typically involve slowly increasing the excitation frequency while allowing the system to reach steady state at each step. Once steady state is reached, the maximum oscillation amplitude is measured. It is crucial that this process is quasi-static: if the frequency step is too large, the system may jump to a different solution branch.

In computational dynamics, this procedure is known as **continuation**. Software packages such as AUTO, MATCONT, and COCO are specifically designed for this purpose and have become cornerstone tools for analyzing bifurcations, limit cycles, frequency response curves, and many related phenomena.

A major drawback of continuation methods is that they rely on *sequential* computations. Periodic orbits must be solved one after another in order to follow a solution branch, meaning that each computation depends on the previous one. This inherently limits parallelization and can result in long computation times.

Poscidyn instead adopts a **multistart approach**, where steady-state periodic orbits are computed from a large set of initial conditions. While this approach requires significantly more computational resources, it can be much faster in practice because all simulations can be executed in parallel.

This multistart strategy typically yields *all* solution branches simultaneously, meaning multiple solutions may exist for a single excitation frequency. In contrast, experimental frequency sweeps usually observe only a single branch. To bridge this gap, Poscidyn introduces **artificial sweep methods**. These methods synthetically reconstruct continuation-like behavior by selecting and ordering solutions from the high-dimensional multistart solution space.

## Limitations

Be sure to consult the [Limitations](../limitations) page, as these kinds of synthetic sweep methods come with important caveats and do not fully replicate true continuation algorithms.
