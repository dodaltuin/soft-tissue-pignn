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

Jmin   = 0.001 # minimum value of J which we allow
Jtrans = 0.05  # point at which transformation function kicks in

Jdiff = Jtrans - Jmin

beta_1 = 1. / Jdiff
beta_2 = jnp.log(Jdiff) - Jtrans/Jdiff
beta_3 = Jmin

exp_fn              = lambda J: jnp.exp(beta_1*J + beta_2) + beta_3
J_transformation_fn = lambda J: jnp.where(J > Jtrans, J, exp_fn(J))

I1_trans    = 10 # point at which transformation function kicks in
tanh_fn     = lambda I1: jnp.tanh(I1 - I1_trans) + I1_trans
I1_trans_fn = lambda I1: jnp.where(I1 < I1_trans, I1, tanh_fn(I1))

I4f_trans    = 8 # point at which transformation function kicks in
tanh_fn      = lambda I4f: jnp.tanh(I4f - I4f_trans) + I4f_trans
I4f_trans_fn = lambda I4f: jnp.where(I4f < I4f_trans, I4f, tanh_fn(I4f))


kappa = 25.
def HO_trans_iso(params, F, J, f0):

    a, b, af, bf, = params

    C = jnp.power(J, -2./3.) * jnp.matmul(F.T, F)

    I1 = jnp.trace(C)
    I4f = jnp.matmul(jnp.matmul(f0.T, C), f0)

    I1 = I1_trans_fn(I1)
    I4f = I4f_trans_fn(I4f)

    Wani = (a/b) * (jnp.exp(b * (I1 - 3.)) - 1.) + (af/bf)*(jnp.exp(bf*(I4f - 1.)**2) - 1)
    Winc = kappa * (J**2 - 1. - 2.*jnp.log(J))

    return Wani + Winc

constitutive_law = HO_trans_iso

####################################################
## Define function to enforce essential boundary conditions
####################################################

abs_path= os.path.join(sys.path[0])

interior_points = device_put(jnp.load(f'{abs_path}/interior-points.npy')*1.).reshape(-1,1)

boundary_adjust_fn = lambda U: U*interior_points


####################################################
## Define external forces (boundary and traction forces)
####################################################

final_pressure_loading = 1.06658
body_force             = None

####################################################
## Define material parameter boundaries
####################################################

params_lb = jnp.array([.1,  1.,  1.5,  1.])
params_ub = jnp.array([.26, 4.2, 5.18, 4.46])

####################################################
## Sample parameters on log scale or uniform scale
####################################################

log_sampling = True

####################################################
## Set number of parameters to sample at each epoch
####################################################

epoch_size = 200


