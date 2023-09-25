"""
File: constitutive_law.py
Author: David Dalton
Description: Various setups before training, e.g.constitutive law
"""

####################################################
## Package imports
####################################################

import jax.numpy as jnp
from jax import device_put
import os
import sys

####################################################
## Transformation functions for stabilising strain 
## energy - see Section 2.4.3 of paper for details
####################################################

Jmin = 0.001  # minimum value of J which we allow
Jtrans = 0.05 # point at which transformation function kicks in

Jdiff = Jtrans - Jmin

beta_1 = 1. / Jdiff
beta_2 = jnp.log(Jdiff) - Jtrans/Jdiff
beta_3 = Jmin

exp_fn = lambda J: jnp.exp(beta_1*J + beta_2) + beta_3
J_transformation_fn = lambda J: jnp.where(J > Jtrans, J, exp_fn(J))

I1_trans = 10 # point at which transformation function kicks in

tanh_fn = lambda I1: jnp.tanh(I1 - I1_trans) + I1_trans
I1_trans_fn = lambda I1: jnp.where(I1 < I1_trans, I1, tanh_fn(I1))

####################################################
## Define constitutive law 
####################################################

def NeoHookean(params, F, J, fibres=None):
    """NeoHookean strain energy density function

    Inputs:
    -----------
    params: jnp.array
       Material parameter vector
    F     : jnp.array
       Deformation gradient
    J     : jnp.array
       Determinant of F
    fibres: None
       Not used here as we assume isotropic material

    Returns:
    ----------
    sed: float
       Strain energy density
    """

    # extract material parameters
    E, nu = params

    # convert to Lame parameters
    mu, lambda_ = (E/(2*(1 + nu))), (E*nu/((1 + nu)*(1 - 2*nu)))

    # right Cauchy-Green tensor
    C = jnp.matmul(F.T, F)

    # first invariant
    Ic = jnp.trace(C)

    # transform to stop very large values
    Ic = I1_trans_fn(Ic)

    lnJ = jnp.log(J)

    sed = (mu/2.)*(Ic - 3.) - mu*lnJ + (lambda_/2.)*(lnJ**2)

    return sed

constitutive_law = NeoHookean

####################################################
## Define function to enforce essential boundary conditions
####################################################

abs_path= os.path.join(sys.path[0])

interior_points = device_put(jnp.load(f'{abs_path}/interior-points.npy')*1.)
disp_add = device_put(jnp.load(f'{abs_path}/disp-add.npy')*1.)

boundary_adjust_fn = lambda U: (U*interior_points) + disp_add

####################################################
## Define external forces (boundary and traction forces)
####################################################

final_pressure_loading = None
body_force             = None

####################################################
## Define material parameter boundaries
####################################################

params_lb = jnp.array([1.,  0.1])
params_ub = jnp.array([25., 0.4])

####################################################
## Sample parameters on log scale or uniform scale 
####################################################

log_sampling = True

####################################################
## Set number of parameters to sample at each epoch 
####################################################

epoch_size = 200
















