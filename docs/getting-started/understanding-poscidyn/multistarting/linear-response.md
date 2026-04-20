[<-- Go back to multistarting](../start-here.md#multistarting)

# Linear response multistart

For each combination of drive frequency and drive amplitude, Poscidyn generates a set of random initial conditions. The search space for these initial conditions is estimated from the linear response of each mode. Assuming linear resonance, the maximum modal displacement and velocity can be approximated as 

$$
\begin{aligned}
x_{\max,i} &= \frac{f_i Q_i}{\omega_{0,i}^2}, \\
v_{\max,i} &= \frac{f_i Q_i}{\omega_{0,i}}.
\end{aligned}
$$

This defines an approximate operating range

$$
\begin{aligned}
x_{0,i} &\in [-x_{\max,i},\, x_{\max,i}], \\
v_{0,i} &\in [-v_{\max,i},\, v_{\max,i}],
\end{aligned}
$$

from which Poscidyn draws $\texttt{n_init_cond}$ random initial conditions. By sampling the initial state space in this way, the package increases the probability of capturing the relevant stable attractors at each excitation condition. Increasing $\texttt{n_init_cond}$ thus increases the odds of finding all the stable attractors, and will be a key simulation hyperparameter. 