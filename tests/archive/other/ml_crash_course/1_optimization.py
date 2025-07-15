from matplotlib import pyplot as plt
import jax.numpy as jnp
from test_jax import grad

def E(y,spring,length,mass_g):
    """
    Calculate the energy.
    
    y : array containing vertical coordinates, evaluated at x = 0, 1, 2, ...
    spring : spring constant
    length : equilibrium length
    mass_g : mass times g for an individual mass
    """
    
    spring_energy = jnp.sum( 0.5*spring * ( jnp.sqrt((y[1:]-y[:-1])**2 + 1) - length )**2 )
    
    # add contributions from beginning and end (fixed boundaries!):
    spring_energy += 0.5*spring * ( jnp.sqrt(y[0]**2 + 1) - length )**2
    spring_energy += 0.5*spring * ( jnp.sqrt(y[-1]**2 + 1) - length )**2
    
    gravitational_energy = mass_g * jnp.sum(y)
    
    return spring_energy + gravitational_energy
        
# Produce the function that calculates the gradient of E
# with respect to the y argument (argument number 0):

grad_E = grad(E, argnums = 0)

# Physics parameters:
N = 20 # number of masses in the chain
spring = 5.0 # spring constant
length = 0.8 # springs are stretched even when chain is straight!
mass_g = 0.1 # m*g for each mass

# Initialization:
y = jnp.zeros(N) # initially, all masses at y=0

# Now do several steps of gradient descent:

# Gradient descent steps:
nsteps = 60 # number of gradient descent steps
eta = 0.2 # size of each gradient descent step

zero = jnp.array([0.0]) # needed for plotting

for step in range(nsteps):
    y -= eta * grad_E(y, spring, length, mass_g)
    plt.plot(jnp.concatenate([zero,y,zero]), color="orange") # plot with boundaries appended

plt.title("Chain under gravity")
plt.show()


### OPTAX ###
import optax


# Initialization:
y = jnp.zeros(N) # initially, all masses at y=0

# the gradient step size is called "learning rate"
# in machine learning language:
learning_rate = 0.1

# get the optimizer:
optimizer = optax.adam( learning_rate )
# initialize the 'state' of the optimizer, by
# telling it about the initial values:
opt_state = optimizer.init( y )

# Gradient descent steps:
nsteps = 60 # number of gradient descent steps

zero = jnp.array([0.0])

for step in range(nsteps):
    # get gradients:
    grads = grad_E(y, spring, length, mass_g)
    
    # update y using the gradients:
    updates, opt_state = optimizer.update( grads, opt_state)
    y = optax.apply_updates( y, updates )
    
    plt.plot(jnp.concatenate([zero,y,zero]), color="orange") # plot with boundaries appended

plt.title("Chain under gravity")
plt.show()