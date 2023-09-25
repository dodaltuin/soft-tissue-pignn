"""
File: utils.py
Author: David Dalton
Description: Utility functions for initialising PI-GNN emulators and saving results
"""

import os

import pathlib
import pickle

from jax import random
import jax.numpy as jnp
import flax
from flax.core import freeze, unfreeze

from typing import Sequence

import models

def create_config_dict(K: int, n_epochs: int, lr: float, output_dim: int, local_embed_dim: int, mlp_features: Sequence[int], rng_seed: int):
    """Creates dictionary of configuration details for the GNN emulator"""

    return {'K': K,
            'n_train_epochs': n_epochs,
            'learning_rate': lr,
            'output_dim': output_dim,
            'local_embedding_dim': local_embed_dim,
            'mlp_features': mlp_features,
            'rng_seed': rng_seed
            }

def create_savedir(data_path: str, K: int, n_epochs: int, lr: float, dir_label: str, local_embedding_dim: int, mlp_width: int, mlp_depth: int, rng_seed: int):
    """Create directory where emulation results are saved

    The emulator's configuration details are written to the directory name for ease of reference
    """

    save_dir = f'emulationResults/{data_path}/pge_K{K}_Ntrain{n_epochs}_lr{lr:.1e}_M{local_embedding_dim}_FCNN_W{mlp_width}_FCNN_D{mlp_depth}_RNG{rng_seed}{dir_label}/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    # also create a directory to save log files
    log_dir = f'{save_dir}/logFiles'
    if not os.path.isdir(log_dir): os.makedirs(log_dir)

    return save_dir


def load_trained_params(params_save_dir, params_filename="trainedNetworkParams.pkl"):
    """Load trained parameters of GNN emulator from params_save_dir"""

    params_filename_full = params_save_dir + params_filename
    if not os.path.isfile(params_filename_full):
        raise FileNotFoundError(f'No file at: {params_filename_full}')

    with pathlib.Path(params_filename_full).open('rb') as fp:
        params_load = pickle.load(fp)

    return params_load


def gen_zero_params_gnn(model, params_randL):
    node_decode_first_mlp_index = model.K*2 + 3
    node_decode_last_mlp_index = node_decode_first_mlp_index + model.output_dim[0]

    mlp_depth = len(model.mlp_features)
    index = node_decode_first_mlp_index

    final_weights_layer_decoder_rand = params_randL['params'][f'FlaxMLP_{index}'][f'Dense_{mlp_depth}']['kernel']
    final_weights_layer_decoder_zero = jnp.zeros_like(final_weights_layer_decoder_rand)

    params_zero = unfreeze(params_randL)

    for index in range(node_decode_first_mlp_index, node_decode_last_mlp_index):
        params_zero['params'][f'FlaxMLP_{index}'][f'Dense_{mlp_depth}']['kernel'] = final_weights_layer_decoder_zero

    return freeze(params_zero)


def initialise_network_params(data_generator, ref_geom, model, trained_params_dir: str, rng_seed: int):
    """Initialise the parameters of the GNN emulator

    If initialising from scratch, use the ".init" method from Flax

    If initialising from earlier training results, simply read these parameters
    from trained_params_dir
    """

    if trained_params_dir == "None":
        key = random.PRNGKey(rng_seed)
        theta_init, _ = data_generator.get_data(0)
        V_init = ref_geom._node_features
        E_init = ref_geom._edge_features

        params = model.init(key, V_init, E_init, theta_init)
        return params
    else:
        trained_params = load_trained_params(trained_params_dir)
        return trained_params


def init_emulator_full(config_dict: dict, data_generator, trained_params_dir: str, ref_geom):
    """Initialise GNN emulator (varying geometry data)

    Initialises GNN architecture and trainable paramters for prediction of varying LV geom data

    If trained_params_dir is "None", the parameters are initialised randomly
    If trained_params_dir is a directory path, pre-trained parameters are read from there
    """

    # initialise GNN architecture based on configuration details
    emulator =  models.PrimalGraphEmulator(mlp_features=config_dict['mlp_features'],
                                           latent_size=[config_dict['local_embedding_dim']],
                                           K = config_dict['K'],
                                           receivers = ref_geom._receivers,
                                           senders = ref_geom._senders,
                                           n_total_nodes= ref_geom._n_total_nodes,
                                           output_dim= [config_dict['output_dim']],
                                           real_node_indices = ref_geom._real_node_indices,
                                           boundary_adjust_fn = ref_geom.boundary_adjust_fn)

    # initialise trainable emulator parameters (either randomly or read from trained_params_dir)
    params = initialise_network_params(data_generator, ref_geom, emulator, trained_params_dir, config_dict['rng_seed'])

    return emulator, params


def init_emulator_decoder(model, params: dict, data_generator, ref_geom, results_save_dir: str, trained_params_dir: str):
    """Initialises fixed-geom emulator network parameters

    Initialises the parameters of a GNN emulator for fixed LV geometry data,
    where the pre-trained parameters come from a varying geometry emulator

    In this case, the message passing stage for the fixed geometry must
    be pre-compututed before the emulator (DeepGraphEmulatorFixedGeom) is
    initialised
    """

    # extract the node features (V), edge features (E), global parameters (theta)
    # and global shape embedding coefficients (z_global)
    #V, E, theta, z_global, _, _ = data_loader.return_index_0()
    theta_init = None
    V_init = ref_geom._node_features
    E_init = ref_geom._edge_features
    z_global_init = None

    # sow_latents=True returns the local learned reprentation from the message
    # passing stage or each node in the geometry
    latent_nodal_values = model.apply(params, V_init, E_init, theta_init, sow_latents=True)

    # Flax names the internal MLPs of the GNN 'FlaxMLP_i', where i ranges over the
    # number of MLPs in the network, and is numbered in order of initialisation
    # There are 2 (the two encode MLPs) + K*2 (two MLPs for each message passing step)
    # + 1 (the theta encode MLP) + D decoder MLPs = 3 + K*2 + D MLPs in the DeepGraphEmulator
    # GNN architecture. We want the index of the theta encoder MLP for use in the fixed
    # geometry emulator, which was the (2*K + 3)th MLP to be initialised. Therefore its
    # index is given as follows (Python uses 0-based indexing)
    theta_encode_mlp_index = model.K*2 + 2

    # extract theta encoder MLP params from params dictionary
    theta_encode_mlp_params = params['params'][f'FlaxMLP_{theta_encode_mlp_index}']
    theta_encode_mlp_params_dict = {'params': theta_encode_mlp_params.unfreeze()}
    theta_encode_mlp_params_dict = flax.core.frozen_dict.freeze(theta_encode_mlp_params_dict)

    # define a function to map theta to z_theta given value for theta encode MLP parameters
    theta_encode_mlp = models.FlaxMLP(model.mlp_features + model.latent_size, True)
    theta_encode_mlp_fn = lambda x: theta_encode_mlp.apply(theta_encode_mlp_params_dict, x)

    # initialise the fixed geometry (fg) emulator
    model_fg = models.PrimalGraphEmulatorDecoder(mlp_features = model.mlp_features,
                                                 output_dim= [ref_geom._output_dim],
                                                 n_real_nodes= ref_geom._n_real_nodes,
                                                 latent_nodal_values= latent_nodal_values,
                                                 theta_encode_mlp_fn=theta_encode_mlp_fn,
                                                 boundary_adjust_fn = ref_geom.boundary_adjust_fn)

    # index for first decoder MLP is one after the theta_encode_mlp
    node_decode_first_mlp_index = theta_encode_mlp_index + 1
    node_decode_last_mlp_index = node_decode_first_mlp_index + model.output_dim[0]

    # extract parameters for "D" node decode MLPs to a list
    decoder_mlps_params_list = [params['params'][f'FlaxMLP_{index}'] for index in range(node_decode_first_mlp_index, node_decode_last_mlp_index)]

    # convert parameters list to a frozen_dict suitable for use with Flax
    decoder_mlps_params_dict = {'params': {f'FlaxMLP_{i}':params_i for i, params_i in enumerate(decoder_mlps_params_list)}}
    params_fg = flax.core.frozen_dict.freeze(decoder_mlps_params_dict)

    return model_fg, params_fg


def initialise_emulator(emulator_config_dict, data_generator, results_save_dir, ref_geom_data, fix_message_layers=False, trained_params_dir="None"):
    """Initialises GNN emulator

    Initialises GNN emulator and parameters, given specifications given in emulator_config_dict

    If fixed_message_layers=False, returns instance of models.PrimalGraphEmulator
    If fixed_message_layers=True, returns instance of models.PrimalGraphEmulatorDecoder
    """

    # check input conflict
    if fix_message_layers:
        assert trained_params_dir != "None", \
               'If fixing message passing layers, must initialise from pre-trained network parameters'

    # initialise varying geometry emulator (models.PrimalraphEmulator) and parameters
    emulator, params = init_emulator_full(emulator_config_dict, data_generator, trained_params_dir, ref_geom_data)
    emulator_pred_fn = lambda p, theta_norm: emulator.apply(p, ref_geom_data._node_features, ref_geom_data._edge_features, theta_norm)

    # if fixing message passing layers, use the above model and parameters to initialise the decoder stage of the emulator by pre-computing the message passing stage of the emulation architecture
    if fix_message_layers:
        emulator, params = init_emulator_decoder(emulator, params, data_generator, ref_geom_data, results_save_dir, trained_params_dir)
        emulator_pred_fn = lambda p, theta_norm: emulator.apply(p, theta_norm)

    return emulator_pred_fn, params, emulator

