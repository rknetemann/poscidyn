Poscidyn is primarily intended for users who need to simulate **experimentally realistic time responses and frequency sweeps** with minimal implementation overhead. The package is designed for applications where **ease of use and computational speed** are central requirements.

Rather than serving as a general-purpose continuation or bifurcation analysis framework, Poscidyn focuses on simplifying common workflows in nonlinear oscillator dynamics. It provides ready-to-use abstractions for frequently used oscillator models, excitation types, solvers, and sweep strategies, allowing users to focus on analysis rather than numerical boilerplate.

By handling many of the practical challenges associated with these simulations—such as steady-state detection, sweep construction, and batching—Poscidyn enables rapid experimentation and large-scale studies with only a small amount of user code.

If Poscidyn does not include a specific model or excitation type required for your application, the package is designed to be extensible. The [Extending Poscidyn](../usage/extending-poscidyn) section of the documentation explains how existing components can be adapted or expanded to fit custom use cases.

## Where to go next?

If this package aligns with your needs, the getting-started section walks through the initial steps of using Poscidyn. Begin with the [Installation](../getting-started/installation) page to install the package and verify platform compatibility.
