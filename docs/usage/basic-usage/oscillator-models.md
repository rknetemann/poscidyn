Let's start by choosing an oscillator model. Poscidyn currently only has one models built in. But new models are easily added and will be added in the near future. The full list of models is/will be available on the [Oscillator models](#) page.

!!! note
    
    Did you extend Poscidyn using your own oscillator? Please contribute to the project by doing a pull request. Have a look at the [Contributing](#) page.

### Single DOF Linear Oscillator

To keep things simple we will setup a simple linear harmonic oscillator first. Each model requires their own set of parameters to be initialized. In the case of a the harmonic oscillator that are the quality factor $Q$ and resonance frequency $\omega_0$:

$$
\ddot x
  + \frac{\omega_0}{Q}\, \dot x
  + \omega_0^{2}\, x
  \;=\; f
        \cos\!\left(\omega t\right).
$$

Everything related to the external forcing/ input ($f$ and $\omega$) we will define later.

We begin by importing the necessary libraries:
```python
import numpy as np
import poscidyn
```

We then define the parameters of our system:
```python
n_modes = 1
Q = np.array([100])
omega_0 = np.array([1.0])
```

!!! note
    It may seem confusing at first swhy we are using arrays here, that will become clear once we use this same system for multiple degrees of freedom.

The last thing to do is initialize the model:

```python

model = poscidyn.NonlinearOscillator(n_modes=n_modes, Q=Q, omega_0=omega_0)
```

That's it! Though the process of setting up a model, differs a little bit per model. So make sure to have a look at the documentation of each model to get started using it quickly.

### Single DOF Duffing Oscillator

To keep things simple we will setup a simple linear harmonic oscillator first. Each model requires their own set of parameters to be initialized. In the case of a the harmonic oscillator that are the quality factor $Q$, resonance frequency $\omega_0$ and cubic stiffness $b$:

$$
\ddot x
  + \frac{\omega_0}{Q}\, \dot x
  + \omega_0^{2}\, x + b\,x^3
  \;=\; f
        \cos\!\left(\omega t\right).
$$


We again define the parameters of our system:
```python
n_modes = 1
Q = np.array([100])
omega_0 = np.array([1.0])
b = np.zeros((n_modes, n_modes, n_modes, n_modes))
b[0,0,0,0] = 0.1
```

We initialize the model:

```python

model = poscidyn.NonlinearOscillator(n_modes=n_modes, Q=Q, omega_0=omega_0)
```

!!! note
    
    Why did we use np.zeros((n_modes, n_modes, n_modes, n_modes)) here for b? Have a look at the page for [NonlinearOscillator](#) to see why this is the case.


### Multiple DOF Symmetry-breaking Oscillator

For demonstration purposes, let's also see how we can use the same model for multiple degrees of freedom simulations.

$$
\begin{aligned}
  \ddot q_1 &+ \frac{ \omega_{0,1}}{ Q_1} \dot q_1 +  \omega^2_{0,1} q_1 + a^{(1)}_{12} q_1 q_2 + b^{(1)}_{111} q^3_1 = f_1 \cos( \omega t), \\[6pt]
  \ddot q_2 &+ \frac{ \omega_{0,2}}{ Q_2} \dot q_2 +  \omega^2_{0,2} q_2 + a^{(2)}_{11} q^2_1 = f_2 \cos( \omega t) .
\end{aligned}
$$

```python
Q = np.array([80.0, 40.0])
omega_0 = np.array([1.0, 2.0])
a = np.zeros((2, 2, 2))
b = np.zeros((2, 2, 2, 2))
a[0,0,1] = 0.16
a[1,0,0] = 0.08
b[0, 0, 0, 0] = 1
```

The last thing to do is initialize the model:

```python

model = poscidyn.NonlinearOscillator(omega_0=omega_0, Q=Q,a=a, b=b)
```

