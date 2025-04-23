import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from normal_form import normal_form, random_parameter_vectors
import jax

if __name__ == "__main__":
    key = jax.random.key( 44 )
    
    batchsize = 10
    ranges = [[-1., 1.], [0.5, 1.5], [0.0, 0.15]]
    result = random_parameter_vectors( key, batchsize, ranges )
