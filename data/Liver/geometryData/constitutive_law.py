#
import jax.numpy as jnp
from jax import device_put
import os
import sys

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

def NeoHookean(params, F, J, fibres=None):

    lambda_, mu = params

    C = jnp.matmul(F.T, F)
    Ic = jnp.trace(C)
    Ic = I1_trans_fn(Ic)

    lnJ = jnp.log(J)

    return (mu/2.)*(Ic - 3.) - mu*lnJ + (lambda_/2.)*(lnJ**2)


constitutive_law = NeoHookean

####################################################
## Define function to enforce essential boundary conditions
####################################################

abs_path= os.path.join(sys.path[0])

interior_points = device_put(jnp.load(f'{abs_path}/interior-points.npy')*1.)

boundary_adjust_fn = lambda U: U*interior_points


####################################################
## Define external forces (boundary and traction forces)
####################################################

final_pressure_loading = None
g      = 980
body_force             = jnp.array([0.]*(2) + [-g])

####################################################
## Define material parameter boundaries
####################################################

params_lb = jnp.array([35000.]*2)
params_ub = jnp.array([100000.]*2)

epoch_size = 200

log_sampling = True






