Poscidyn is still under active development and has not yet reached its full potential. This page outlines the planned enhancements and long-term vision for the package.

## Planned Enhancements

- **Remove memory limitation**
  Currently Poscidyn requires quite some memory to perform sweeps, but cutting the work into smaller pieces allows for a much smaller memory footprint. 
  
- **Oscillator models**  
  Extension of the built-in model library to include additional canonical systems such as the Van der Pol, Lorenz, and Rayleigh oscillators.

- **Parametric excitation**  
  Support for parametric excitation mechanisms, enabling the study of parametrically driven systems.

- **Solvers**  
  Integration of additional solvers, including shooting methods and potentially standard continuation techniques, while preserving Poscidyn’s ability to batch computations efficiently.

- **Visualization tools**  
  Built-in plotting and visualization utilities for common analyses, such as phase-space trajectories, frequency response curves, and time-domain responses.

- **Sweep methods**  
  Development of more physically motivated synthetic sweep strategies to better emulate experimental frequency sweep behavior for non-continuation methods.

## Long-Term Ideas

- **Hybrid approaches**  
  Combination of multistart batching with localized continuation methods, for example by segmenting sweeps and performing parallel continuations.

## How to Contribute

Contribution guidelines and development workflows will be documented in a future release.
