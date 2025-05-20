from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

alpha = 1 # 1/day
beta = 0.2 # 1/wolves/day
delta = 0.5 # 1/rabbits/day
gamma = 0.2 # 1/day

def diffeq(t,pop):
  x,y = pop
  return [alpha*x-beta*x*y,
          delta*x*y-gamma*y]

sol = solve_ivp(diffeq, 
                t_span = [0,40],
                y0=[1,5],
                t_eval=np.linspace(0,40,100))

plt.plot(sol.t, sol.y.T,'-')
plt.legend(['Rabbits','Wolves'])
plt.xlabel('Time [days]')
plt.ylabel('Population [#]')
plt.show()