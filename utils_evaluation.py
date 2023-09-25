"""
File: utils_evaluation.py
Author: David Dalton
Description: Utility functions for evaluating emulator prediction accuracy
"""

import jax
import jax.numpy as jnp
import numpy as np
import shutil
import os

from utils_potential_energy import compute_def_gradient

def compute_I1(F):
    """Compute first invariant of the deformation gradient F"""
    return jnp.trace(jnp.matmul(F.T, F))

compute_I1_vmap = jax.vmap(compute_I1)

def compute_F_J_I1(displacement, ref_geom_data):
    """Compute deformation gradient F, J=det(F) and first invariant I1"""

    # coords in current configuration after displacement
    cur_coords = ref_geom_data.ref_coords + displacement

    F, J = compute_def_gradient(ref_geom_data.elements, ref_geom_data.ref_coords, cur_coords, ref_geom_data.Jtransform)

    I1 = compute_I1_vmap(F)

    return F, J, I1

compute_F_J_I1_vmap = jax.vmap(compute_F_J_I1, in_axes = [0] + [None])

def err_u(u, uhat=0.):
    """RMSE of nodal displacement predictions"""
    return (((u - uhat)**2).sum(-1))**.5

def rel_err_u(u, uhat):
    """Relative error of displacement predictions"""
    return err_u(u, uhat)/err_u(u)

# matrix norm
Frobenius_norm = lambda X: (X**2).sum([-1, -2])**.5

def err_F(F, Fhat):
    """Error in prediction of deformation gradient"""
    return 100.*Frobenius_norm(F-Fhat)/Frobenius_norm(F)

def err_norm(true, pred):
    return jnp.abs((true-pred)/true)*100.

def find_quantiles(arr, label, logging):
    quantiles = jnp.percentile(arr, jnp.array([0., 2.5, 25., 50., 75., 97.5, 100.]))
    logging.info(f'{label} quantiles: 0%:{quantiles[0]:.2e}, 2.5%:{quantiles[1]:.2e}, 25%:{quantiles[2]:.2e}, 50%:{quantiles[3]:.2e}, 75%:{quantiles[4]:.2e}, 97.5%:{quantiles[5]:.2e}, 100%:{quantiles[-1]:.2e}')
    return quantiles

def print_error_statistics(true_arrs, pred_arrs, save_dir, logging):
    """Prints prediction error statistics to console"""

    Utrue, PEtrue, Ftrue, Jtrue, I1true = true_arrs
    Upred, PEpred, Fpred, Jpred, I1pred = pred_arrs

    u_norms  = err_u(Utrue)
    u_errors = err_u(Utrue, Upred)
    PE_errors = err_norm(PEtrue, PEpred)
    F_errors = err_F(Ftrue, Fpred)
    J_errors = err_norm(Jtrue, Jpred)
    I1_errors = err_norm(I1true, I1pred)

    logging.info('Prediction error quantiles:')
    find_quantiles(u_norms,   'u norm', logging)
    find_quantiles(u_errors,  'err_u ', logging)
    find_quantiles(PE_errors, 'err_PE', logging)
    find_quantiles(F_errors,  'err_F ', logging)
    find_quantiles(J_errors,  'err_J ', logging)
    find_quantiles(I1_errors, 'err_I1', logging)

def find_quantile_indices(loss_array):
    """find the simulations from the test set on which the emulator achieved the median and worst prediction error respectively"""

    # median and worst loss values
    loss_quantiles = [50, 100]
    loss_quantile_values = np.percentile(loss_array, loss_quantiles)

    index_values = []

    # find the corresponding indices in the data set
    for loss_quantile_i in loss_quantile_values:
        index = np.argmin(np.abs(loss_array - loss_quantile_i))
        index_values.append(index)

    return index_values
