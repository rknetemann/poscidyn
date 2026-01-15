Let's start by choosing an oscillator model. Poscidyn has multiple built in oscillator models. The full list of models is available on the [Oscillator models](#) page.

!!! note
    
    Did you extend Poscidyn using your own oscillator, please contribute to the project by doing a pull request. Have a look at the [Contributing](#) page.

Each model requires their own set of parameters to be initialized. In the case of a duffing oscillator defined in terms of Quality-factor Q, resonance frequency $\omega_0$ and cubic  stiffness $\gamma$:

$$
\ddot x
  + \frac{\omega_0}{Q}\, \dot x
  + \omega_0^{2}\, x
  + \gamma\, x^{3} 
  \;=\; f\,
        \cos\!\left(\omega\,t\right).
$$


```python
import numpy as np
import poscidyn

Q = np.array([Q_1_val])
omega_0 = np.array([1.0])
alpha = np.zeros((1,1,1))
gamma = np.zeros((1,1,1,1))
gamma[0,0,0,0] = 1.0

model = poscidyn.NonlinearOscillator(Q=Q, omega_0=omega_0, alpha=alpha, gamma=gamma)
```