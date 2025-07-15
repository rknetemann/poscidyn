from matplotlib import pyplot as plt

import jax.numpy as jnp
from jax import grad, vmap, jit, value_and_grad
import jax
import jax.random

import optax

from functools import partial

import tqdm

def runge_kutta_step_with_params(X,params,t,rhs,dt):
    """
    One step of the standard Runge-Kutta fourth order.
    This assumes rhs can be called like rhs(X,params,t)
    """
    k1=rhs(X,params,t)
    k2=rhs(X+0.5*dt*k1,params,t+0.5*dt)
    k3=rhs(X+0.5*dt*k2,params,t+0.5*dt)
    k4=rhs(X+dt*k3,params,t+dt)
    return( (dt/6.)*(k1+2*k2+2*k3+k4) )

# we need this partial statement to tell jax
# that the 'rhs' argument is not an array but something
# else!
@partial(jax.jit,static_argnames=['rhs'])
def runge_kutta_solve(X0,rhs,ts,params):
    """
    Solve the differential equation dX/dt=rhs(X,params,t), for all (equally spaced) times in ts,
    with initial value X(t=0)=X0.
    
    Here X0 can have ANY shape as an array, and rhs(X,params,t) must return the same shape.
    
    This is convenient in case you want to, say, do many trajectories in parallel,
    or reserve one index for the particle number, etc. You do not need to assume
    X to be a 1d array, as you would have to do for some other Runge-Kutta implementations.
    It is also nice to split the positions and velocities, e.g. X[0,:] for the positions
    of all particles, and X[1,:] for the velocities.
    
    Returns: Xs,ts
    
    where the shape of the solution Xs is the same as that of X, except we add another dimension
    at the end, of size 'nsteps'. 'ts' just is a 1d array denoting the time steps for
    the solution.
    
    Plotting the solution may work like this, if X was a 1d array:
    
    plt.plot(ts,Xs[5]) # Note that Xs[5] is the same as Xs[5,:]
    
    ...or like this, if e.g. X[1,:] were all the velocities of all particles:
    
    plt.plot(ts,Xs[1,3]) # plotting velocity of particle number 3: Xs[1,3] is the same as Xs[1,3,:]
    
    (code by Florian Marquardt 2020, 2024)
    """
    dt=ts[1]-ts[0]

    def loop_body(x,t):
        x+=runge_kutta_step_with_params(x,params,t,rhs,dt)
        return x,x
    
    _,Xs=jax.lax.scan(loop_body,X0,xs=ts)
    return Xs,ts

# batched parameters:
parallel_param_runge_kutta_solve = vmap(runge_kutta_solve,in_axes=[None,None,None,0],
                                                   out_axes=0)

# batched initial conditions:
parallel_runge_kutta_solve = vmap(runge_kutta_solve,in_axes=[0,None,None,None],
                                                   out_axes=0)

def duffing_rhs(z,rhs_params,t):
    # params = [omega_start, domega_dt , gamma, epsilon, force, omega0]
    return ( (-1j*(rhs_params[5]-(rhs_params[0]+rhs_params[1]*t)) - 0.5*rhs_params[2])*z 
            -1j*rhs_params[3]*jnp.abs(z)**2 * z + 1j*rhs_params[4] )

def solve_duffing(z0, omega0, gamma, epsilon, force, omega_start, omega_stop, t_end, nsteps):
    ts=jnp.linspace(0.,t_end, nsteps)
    domega_dt = (omega_stop - omega_start)/t_end 
    rhs_params=jnp.array([omega_start, domega_dt , gamma, epsilon, force, omega0])
    omegas=omega_start + domega_dt * ts
    return *runge_kutta_solve(z0, duffing_rhs, ts, rhs_params),omegas

# batch-processing version, where omega0,gamma,epsilon can vary across
# the samples of the batch:
solve_duffing_parameter_batch = vmap(solve_duffing,
                                    in_axes=[None,0,0,0,None,None,None,None,None])

# batch-processing version, which can be used to compute a force sweep:
solve_duffing_force_batch = vmap(solve_duffing,
                                 in_axes=[None,None,None,None,0,None,None,None,None],
                                out_axes=0)

def random_parameter_vectors( key, batchsize , ranges ):
    """
    Produce several vectors of length batchsize, with values randomly
    uniformly distributed within the respective values range [min_val,max_val].
    ranges is a list of such value ranges.
    
    Returns: list of random vectors.
    """
    subkeys = jax.random.split( key, len(ranges) )
    return [ jax.random.uniform( subkey, [ batchsize ], 
                                      minval = value_range[0], maxval = value_range[1] )
            for subkey, value_range in zip(subkeys, ranges) ]

def duffing_produce_training_batch( key, batchsize, ranges , num_frequency_bins,
                                  force=1.0, omega_start=-4.0, omega_end=+4.0,
                                  t_end=200.0, n_steps=400):
    """
    Produce a Duffing model training batch.
    Random values for omega0, gamma, and epsilon are generated uniformly
    in the ranges given in the list 'ranges' (a list of [min_val,max_val] entries).
    
    Returns:
    x, y_target
    
    where
    
    x is of shape [batchsize, num_frequency_bins] an]d represents the response curves
    y_target is of shape [batchsize, 3] and gives (omega0,gamma,epsilon) for each sample
    """
    omega0s, gammas, epsilons = random_parameter_vectors( key, batchsize, ranges )
    zs,_,_ = solve_duffing_parameter_batch( 0.0+0.0j, omega0s, gammas, epsilons, 
                                 force, omega_start, omega_end, t_end, n_steps )
    x = jax.image.resize( jnp.abs(zs), [ batchsize, num_frequency_bins ] , "linear")
    return x, jnp.stack([omega0s,gammas,epsilons],1)
    

def produce_force_sweep_image(key, omega0,epsilon,gamma, npixels, 
                              omega_range, low_force, high_force, 
                              t_end, nsteps, noise_strength):
    zs,ts,omegas = solve_duffing_force_batch( 0.0+0.01j, omega0, gamma, 
                      epsilon, jnp.linspace(low_force,high_force,npixels), -omega_range, +omega_range,
                      t_end, nsteps )
    # downscale image (less data for the network to process):
    resized_img = jax.image.resize(jnp.abs(zs), (npixels,npixels), "cubic")
    # add noise:
    resized_img+= noise_strength * jax.random.normal(key, jnp.shape(resized_img) )
    
    # also produce images containing the frequency values
    # and the force values:
    freq_img = jnp.repeat( jnp.linspace(-omega_range, 
                            +omega_range, npixels)[None,:], npixels, axis=0 )
    force_img = jnp.repeat( jnp.linspace(low_force, 
                            high_force, npixels)[:,None], npixels, axis=1 )
    # stack all three images together, so that
    # the resulting axis 0 will represent the three channels:
    return jnp.stack( [resized_img, freq_img, force_img], axis=0 )

# a batched version,
# also compiled for speedup.
# for jit, static_argnums=[3,8] says that arguments index 4 and 9, 
# "npixels" and "nsteps" are static parameters
# and not jax arrays. If these change,
# a recompilation will be triggered automatically.
produce_force_sweep_image_batch = jax.jit( vmap( produce_force_sweep_image, 
                                       in_axes=[0,0,0,0,None,None,None,None,None,None,None], out_axes=0),
                                         static_argnums=[4,9])


# the function that puts everything together
# and will be called during the training loop:
def get_duffing_image_batch(key, batchsize, ranges, 
                            npixels, omega_range, 
                            low_force, high_force, t_end, nsteps,
                            noise_strength):
    subkey1, subkey2 = jax.random.split( key )
    omega0s, gammas, epsilons = random_parameter_vectors( subkey1, batchsize, ranges )
    
    # need to get many random keys, one for each
    # noisy sample in the batch:
    subkeys = jax.random.split( subkey2, batchsize )
    output = produce_force_sweep_image_batch( subkeys, omega0s,epsilons,gammas, 
                                        npixels, omega_range, 
                                        low_force, high_force, 
                                        t_end, nsteps, noise_strength )
    return ( output, jnp.stack([omega0s,gammas,epsilons], axis=1) )

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

def NN(x, params):
    """
    Standard MLP with params['weights'] and params['biases'],
    applied to input vector x. Activation tanh applied to all
    layers except last.
    """
    num_layers=len(params['weights'])
    for layer_idx, (w, b) in enumerate(zip(params['weights'],params['biases'])):
        x = jnp.matmul(w,x) + b
        if layer_idx < num_layers-1:
            x = jnp.tanh( x )
    return x

# produce a batched version (where x has a batch dimension 0)
NN_batch = vmap(NN, in_axes=[0,None], out_axes=0)

# batch-averaged mean-square-error deviation between network output and y_target:
def mse_loss_batch(x, y_target, params):
    return jnp.sum( ( NN_batch(x,params) - y_target )**2 ) / jnp.shape(x)[0]

# take gradient with respect to params (argument number '2' out of 0,1,2)
# also return value of loss. Apply jit to compile it:
mse_loss_batch_val_grad = jax.jit( value_and_grad(mse_loss_batch, argnums=2) )

def conv2d( img, kernel, scalefactor ):
    """
    Shortcut for jax convolution of kernel with
    image 'img', with padding 'VALID' and possibly scaling down by
    a factor (which should ideally divide the img size).
    """
    return jax.lax.conv_general_dilated(
    img, kernel, [scalefactor,scalefactor], "VALID")
    
def NN_conv( x, params ):
    """
    Apply several convolutional layers and then
    switch to fully connected. The structure here
    is fixed.
    
    params must contain params['conv_kernels'],
    params['conv_biases'] for the convolutional
    part and params['dense'] containing weights
    and biases for the dense part.
    """
    # first add a fake batch dimension, because we
    # will deal with batching separately later
    # using vmap, but jax.lax.conv needs a batch
    # dimension (here of size 1):
    x = x[None,...]
    
    # apply several 2d convolutions and downscalings by
    # factor of 2:
    for kernel, bias in zip(params['conv_kernels'],params['conv_biases']):
        x = conv2d( x, kernel, 2 ) + bias[None,:,None,None]
        x = jnp.tanh( x )
        
    # now switch to dense network, "flattening"
    # the image and its channels into a single vector
    x = jnp.reshape( x, [-1] )
    
    # finally, apply the usual dense network,
    # with its weights and biases provided
    # inside params['dense']:
    return NN( x, params['dense'] )

# produce a batched version (where x has a batch dimension 0)
NN_conv_batch = vmap(NN_conv, in_axes=[0,None], out_axes=0)

# batch-averaged mean-square-error deviation between network output and y_target:
def mse_loss_conv_batch(x, y_target, params):
    return jnp.sum( ( NN_conv_batch(x,params) - y_target )**2 ) / jnp.shape(x)[0]

# take gradient with respect to params (argument number '2' out of 0,1,2)
# also return value of loss. Apply jit to compile it:
mse_loss_conv_batch_val_grad = jax.jit( value_and_grad(mse_loss_conv_batch, argnums=2) )

def NN_conv_init_params( key, pixel_rows, pixel_cols,
                        num_channels, kernel_sizes,
                        final_dense_layers ):
    """
    Randomly initialize the parameters needed for
    a convolutional+dense neural network.
    
    key: jax random key
    pixel_rows, pixel_cols: dimensions of input image
       (needed to properly set up the dense part!)
    num_channels: list of channels for the conv. part,
       starting with the input image channels
    kernel_sizes: listing the sizes of the kernels,
       a list of length one less than num_channels
    final_dense_layers: list of numbers of neurons in
       the final few dense layers (excluding the
       first dense layer, which results from flattening
       the convolutional output, whose neuron number is
       computed automatically from the image dim.
       given above)
       
    Returns params dictionary with entries
    'conv_kernels', 'conv_biases', and 'dense'.
    """
    params = {}
    params['conv_kernels']=[]
    params['conv_biases']=[]
    
    for lower_channels, higher_channels, kernel_size in zip( num_channels[:-1], num_channels[1:], kernel_sizes ):
        key,subkey = jax.random.split( key )
        params['conv_kernels'].append( jax.random.normal( subkey,
                                        [higher_channels,lower_channels,kernel_size,kernel_size] ) /  
                                 jnp.sqrt( lower_channels ) )
        # keep track of image shape during these convolutions and
        # downscalings! (this would change if you use 'SAME' instead of 'VALID')
        pixel_rows = ( pixel_rows - (kernel_size-1) + 1 ) // 2
        pixel_cols = ( pixel_cols - (kernel_size-1) +1 ) // 2

    for channels in num_channels[1:]:
        params['conv_biases'].append( jnp.zeros( channels ) )

    # now we switch to the dense network!
    # need to calculate the size of the input vector
    # to the fully connected (dense) network:
    neuron_num_input = num_channels[-1] * pixel_rows * pixel_cols
    key,subkey = jax.random.split( key )
    params['dense'] = NN_init_params( subkey, [neuron_num_input] + final_dense_layers )
    
    return params


num_physics_parameters = 3 # output dimension
npixels = 50 # size of images will be npixels * npixels
learning_rate = 1e-3

key = jax.random.key( 42 )

subkey, key = jax.random.split(key)
params = NN_conv_init_params( subkey, 
                pixel_rows=npixels, pixel_cols=npixels,
                num_channels=[3,30,30,30,20,10], 
                kernel_sizes=[3,3,3,3,3],
                final_dense_layers=[50,num_physics_parameters] )

optimizer = optax.adam( learning_rate )
opt_state = optimizer.init( params )

# training parameters
num_training_batches = 100
batchsize = 16

# parameters for the training batches:
value_ranges = [[-1.,1.],[0.5,1.5],[0.0,0.15]] # omega0, gamma, epsilon
noise_strength = 0.1
omega_range = 4.0 # sweep -omega_range..+omega_range
t_end = 200.0 # duration of frequency sweep
nsteps = 400 # time steps for solving dynamics in sweep
low_force = 0.1 # force sweep range
high_force = 2.0

losses=[]

# run this cell multiple times to continue training!

print("Training started, this may take a while...")

for idx_batch in range(num_training_batches):
    # get training batch:
    subkey,key = jax.random.split( key )
    x, y_target = get_duffing_image_batch( subkey,
                                batchsize, value_ranges, npixels,
                                omega_range, low_force, high_force,
                                t_end, nsteps,
                                noise_strength)    
    
    # get loss and its gradient with respect to network parameters:
    loss, grads = mse_loss_conv_batch_val_grad( x, y_target, params )

    # update the network parameters:
    updates, opt_state = optimizer.update( grads, opt_state)
    params = optax.apply_updates( params, updates )

    # add the loss to the list:
    losses.append(loss)

plt.plot( losses )
plt.yscale( "log" )
plt.title("Mean-square error loss vs training batch")
plt.show()

