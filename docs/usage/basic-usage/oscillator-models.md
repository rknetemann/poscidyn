Let's start by choosing an oscillator model. Poscidyn has multiple built in oscillator models. The full list of models is available on the [Oscillator models](#) page.

!!! note
    
    Did you extend Poscidyn using your own oscillator, please contribute to the project by doing a pull request. Have a look at the [Contributing](#) page.

### Single DOF Harmonic Oscillator

To keep things simple we will setup a simple linear harmonic oscillator first. Each model requires their own set of parameters to be initialized. In the case of a the harmonic oscillator that are the damping ratio $\zeta$, resonance frequency $\omega_0$ and cubic  stiffness $\gamma$:

$$
\ddot x
  + 2\zeta \omega_0\, \dot x
  + \omega_0^{2}\, x
  \;=\; f
        \cos\!\left(\omega t\right).
$$

We begin by importing the necessary libraries:
```python
import numpy as np
import poscidyn
```

We then define the parameters of our system:
```python
zeta = np.array([0.01])
omega_0 = np.array([1.0])
driving_frequency = np.linspace(0.5, 2.0, 500)
driving_amplitude = np.linspace(0.1, 1.0, 10)
```

!!! note
    It may seem confusing at first swhy we are using arrays here, that will become clear once we use this same system for multiple degrees of freedom.

The last thing to do is initialize the model:

```python

model = poscidyn.HarmonicOscillator(zeta=zeta, omega_0=omega_0)
```

That's it! Though the process of setting up a model, differs a little bit per model. So make sure to have a look at the documentation of each model to get started using it quickly.


### Multiple DOF Harmonic Oscillator

For demonstration purposes, let's also see how we can use the same model for multiple degrees of freedom simulations.

$$
\begin{aligned}
\ddot x_1
&+ 2\zeta_1 \omega_1\, \dot x_1
+ \omega_1^{2}\, x_1
+ c_{12}\,(\dot x_1 - \dot x_2)
+ k_{12}\,(x_1 - x_2)
= f \cos(\omega t), \\[6pt]
\ddot x_2
&+ 2\zeta_2 \omega_2\, \dot x_2
+ \omega_2^{2}\, x_2
+ c_{12}\,(\dot x_2 - \dot x_1)
+ k_{12}\,(x_2 - x_1)
= 0 .
\end{aligned}
$$

We begin by importing the necessary libraries:
```python
import numpy as np
import poscidyn
```

We then define the parameters of our system:
```python
zeta = np.array([0.01])
omega_0 = np.array([1.0])
driving_frequency = np.linspace(0.5, 2.0, 500)
driving_amplitude = np.linspace(0.1, 1.0, 10)
```

The last thing to do is initialize the model:

```python

model = poscidyn.HarmonicOscillator(zeta=zeta, omega_0=omega_0)
```

