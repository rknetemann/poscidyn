from matplotlib import pyplot as plt

import jax.numpy as jnp
from jax import grad, vmap, jit, value_and_grad
import jax
import jax.random

import optax

def NN(x, params):
    """10.5281/zenodo.7808834
    Standard multilayer perception "MLP" with params['weights'] and params['biases'],
    applied to input vector x. Activation tanh applied to all
    layers except last.
    
    Returns activation vector of the output layer.
    """
    num_layers=len(params['weights'])
    for layer_idx, (w, b) in enumerate(zip(params['weights'],params['biases'])):
        x = jnp.matmul(w,x) + b
        if layer_idx < num_layers-1:
            x = jnp.tanh( x )
    return x

def NN_init_params(key, num_neurons_layers):
    """
    Given a jax random key and a list of the neuron numbers
    in the layers of a network (simple fully connected network,
    i.e. 'multi-layer perceptron'), return a dictionary
    with the weights initialized randomly and biases set to zero.
    
    Returns: params, with params['weights'] a list of matrices and
    params['biases'] a list of vectors.
    """
    params = {}
    params['weights'] = []
    params['biases'] = []
    
    for lower_layer, higher_layer in zip( num_neurons_layers[:-1], num_neurons_layers[1:] ):
        key,subkey = jax.random.split( key )
        params['weights'].append( jax.random.normal( subkey,
                                        [higher_layer,lower_layer] ) /  
                                 jnp.sqrt( lower_layer ) )
        
    for num_neurons in num_neurons_layers[1:]:
        params['biases'].append( jnp.zeros( num_neurons) )
    
    return params


NN_batch = vmap( NN, in_axes = [0,None],
             out_axes = 0)


# batch-averaged mean-square-error deviation between x and y_target:
def mse_loss_batch(x, y_target, params):
    return jnp.sum( ( NN_batch(x,params) - y_target )**2 ) / jnp.shape(x)[0]


# take gradient with respect to params (argument number '2' out of 0,1,2)
# also return value of loss:
mse_loss_batch_val_grad = value_and_grad(mse_loss_batch, argnums=2)
mse_loss_batch_val_grad = jax.jit( mse_loss_batch_val_grad  )



# our "true" function that we want to fit using
# the neural network:
def F(q):
    return jnp.exp( - q**2 ) * jnp.sin( 5 * q )

# again, initialize network randomly:

num_hidden_1 = 30 # number of neurons in hidden layer 1
num_hidden_2 = 20

key = jax.random.key( 45 )
subkey, key = jax.random.split(key)
params = NN_init_params( subkey, [1, num_hidden_1, num_hidden_2, 1] )

learning_rate = 1e-2

# get the optimizer:
optimizer = optax.adam( learning_rate )
# initialize the 'state' of the optimizer, by
# telling it about the initial values:
opt_state = optimizer.init( params )

# training parameters
num_training_batches = 10000
batchsize = 32
x_range = [-3.0, 3.0]
losses=[]

# run this cell multiple times to continue training!

for idx_batch in range(num_training_batches):
    # get training batch, by evaluating F at
    # random locations:
    subkey,key = jax.random.split( key )
    x = jax.random.uniform( subkey, [batchsize, 1], minval = x_range[0], maxval = x_range[1])
    y_target = F( x ) # the true values
    
    # get loss and its gradient with respect to network parameters:
    loss, grads = mse_loss_batch_val_grad( x, y_target, params )

    # update the network parameters:
    updates, opt_state = optimizer.update( grads, opt_state )
    params = optax.apply_updates( params, updates )

    # add the loss to the list:
    losses.append(loss)

plt.plot( losses )
plt.yscale( "log" )
plt.title("Mean-square error loss vs training batch")
plt.show()

x = jnp.linspace( -4.0, 4.0, 200 )[:,None]
net_output = NN_batch( x, params )

plt.plot( x, net_output )
plt.plot( x, F( x ))
plt.title("Network (blue) vs true function (orange)")
plt.show()