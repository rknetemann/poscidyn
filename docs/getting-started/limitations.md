Poscidyn has limitations, please make sure to read them before using this library for your application:

- Difference between parallel and embarassingly parallel and how for example increasing the multistart search space could seriously increase simulation time.

- Time integration solvers estimate the steady-state time, but are based on approximations and may require tuning of the time to steady state parameter to correctly calculate the maximum amplitude at different frequencies.

- Synthetically frequency sweeps do not reflect real world physics, and may therefor be inaccurate. 

- Synthetic frequency sweeps currently do also not account for other complex solution branches.

- The built-in multistart method generates a grid of initial conditions based on the linear response amplitude of the system. As nonlinearities can significantly increase the amplitude of oscillation, this search space might be inaccurate, and could require manual tuning.


