"""
File: utils_data.py
Author: David Dalton
Description: data loading utility classes for model and simulation data
"""

import numpy as np
import jax.numpy as jnp
from jax import device_put, random

from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

import os
import sys

from typing import Sequence

import utils_potential_energy as utils_pe

class ExternalForces:
    """Class for storing any external forces that act on the geometry"""

    def __init__(self, data_path: str):

        geometry_data_dir = f'data/{data_path}/topologyData'

        sys.path.insert(0, geometry_data_dir)
        from constitutive_law import final_pressure_loading, body_force

        self.body_force    = body_force             # jnp array (vector)
        self.surface_force = final_pressure_loading # float


class DataGenerator:
    """Class for generating input theta data points at which the PI-GNN emulator is trained"""

    def __init__(self, data_path: str, lhs_seed: int = 101132, sampler_seed: int = 42, shuffle_seed: int = 420):
        """
        Parameters
         ----------
        data_path: str
               Name of the subdirectory within "/data" where the data is stored
        *_seed: int
               Random seeds for data sampling/shuffling
        """

        geometry_data_dir = f'data/{data_path}/geometryData'
        stats_dir         = f'data/{data_path}/normalisationStatistics'

        sys.path.insert(0, geometry_data_dir)
        from constitutive_law import params_lb, params_ub, epoch_size, log_sampling

        # can optionally specify to sample theta inputs on log scale between upper and lower bounds
        if log_sampling:
            self.transform_fn = jnp.log
            self.transform_inv = jnp.exp
        else:
            identity_fn = lambda x: x
            self.transform_fn  = identity_fn
            self.transform_inv = identity_fn

        self.params_lb = self.transform_fn(params_lb)
        self.params_ub = self.transform_fn(params_ub)

        # initialse sampler and shuffler random seeds
        self.lhs_seed = lhs_seed
        self.sampler_key = random.PRNGKey(sampler_seed)
        self.shuffle_key = random.PRNGKey(shuffle_seed)

        self.n_params = len(self.params_lb)

        # array of data point indices that can be iterated over during each epoch
        self.epoch_size = epoch_size
        self.epoch_indices = jnp.arange(self.epoch_size)

        # generate input data points using LHS sample
        self.generate_lhs_points(stats_dir)

    def generate_lhs_points(self, stats_dir):
        """Generate input theta points using Latin HyperCube sampling"""

        sampler = LatinHypercube(d=len(self.params_lb), seed=self.lhs_seed)

        hypercube_samples = sampler.random(n=self.epoch_size)

        theta = self.transform_inv(qmc_scale(hypercube_samples, self.params_lb, self.params_ub))

        self.theta = device_put(theta)

        self.theta_mean = self.theta.mean(0)
        self.theta_std = self.theta.std(0)

        np.save(f'{stats_dir}/theta-mean.npy', self.theta_mean)
        np.save(f'{stats_dir}/theta-std.npy', self.theta_std)

        self.theta_norm = (self.theta - self.theta_mean) / self.theta_std

    def shuffle_epoch_indices(self):
        """Shuffles the order in which the dataset is cycled through

        This is called at the start of each training epoch to randomise the order in which the input data points are seen
        """

        self.shuffle_key, key = random.split(self.shuffle_key)
        self.epoch_indices = random.choice(key, self.epoch_indices, shape=(self.epoch_size,), replace=False)

    def resample_input_points(self):
        """Resample the input points over which the emulator is trained"""

        # reset sampler key
        self.sampler_key, key = random.split(self.sampler_key)

        # uniform random sampling between the lower and upper bounds
        samples = random.uniform(key, shape=(self.epoch_size, self.n_params), minval=self.params_lb, maxval=self.params_ub)

        # save results in original space and normalised results
        self.theta = device_put(self.transform_inv(samples))
        self.theta_norm = device_put((self.theta - self.theta_mean) / self.theta_std)

    def get_data(self, data_idx):
        """Returns input global graph values for specified data point (normalised and unnormalised)"""

        return self.theta_norm[data_idx], self.theta[data_idx]


class ReferenceGeometry:
    """Class for storing geometric/topological data for reference body"""

    def __init__(self, data_path: str, logging=None):
        """
        Parameters
        ----------
        data_path: str
            Name of the subdirectory within "/data" where the data is stored
        logging
            logging module to write information to
        """

        full_data_path = f'data/{data_path}'
        if not os.path.isdir(full_data_path):
            raise NotADirectoryError(f'No directory at: {full_data_path}')

        geometry_data_dir = f'{full_data_path}/geometryData'
        stats_dir         = f'{full_data_path}/normalisationStatistics'

        # load geometry coords in reference configuration
        self.ref_coords = device_put(jnp.load(f'{geometry_data_dir}/reference-coords.npy'))
        self._n_real_nodes, self._output_dim = self.ref_coords.shape

        sys.path.insert(0, geometry_data_dir)
        from constitutive_law import constitutive_law, J_transformation_fn, boundary_adjust_fn

        # constitutive law for the material
        self.constitutive_law = constitutive_law

        # function to transform J to prevent negative values
        self.Jtransform = J_transformation_fn

        self.boundary_adjust_fn = boundary_adjust_fn

        # load mesh finite elements
        self.elements = device_put(jnp.load(f'{geometry_data_dir}/elements.npy'))

        # compute volumes of the finite elements
        self.element_vols = utils_pe.compute_vol(self.elements, self.ref_coords).reshape(-1,1)

        # load mesh topology
        sparse_topology =  jnp.load(f'{geometry_data_dir}/sparse-topology.npy').astype(jnp.int32)
        self._senders = sparse_topology[:,0]
        self._receivers = sparse_topology[:,-1]

        # node features
        self._node_features = device_put(jnp.load(f'{geometry_data_dir}/augmented-node-features.npy').astype('float32'))

        # edge features
        self._edge_features = device_put(jnp.load(f'{geometry_data_dir}/edge-features.npy').astype(np.float32))

        # fibre field (for non-isotropic materials), defined element-wise
        fibre_field_file =   f'{geometry_data_dir}/fibre-field-elementwise.npy'
        if os.path.exists(fibre_field_file):
            self._fibre_field = device_put(np.load(fibre_field_file))
        else:
            # for isotropic materials the fibre field is set to None
            self._fibre_field = [None]*self.elements.shape[0]

        # for models involving a surface force, the below file indicates the nodes and surface facets which lie on the surface where the force is applied
        pressure_surface_indicator_file = f'{geometry_data_dir}/pressure-surface-facets.npy'
        if os.path.exists(pressure_surface_indicator_file):

            pressure_surface_facets = jnp.array((jnp.load(pressure_surface_indicator_file)))

            # compute areas and normal vectors of each triangular facet on the pressure surface ($\partial \Omega^\sigma$ in Eq. 1 of the manuscript)
            area_facets, N_facets = utils_pe.compute_area_N(pressure_surface_facets, self.ref_coords, self.elements)

            # save results
            self.tri_surface_area_normals    = device_put(area_facets.reshape(-1,1) * N_facets)
            self.tri_surface_element_indices = device_put(pressure_surface_facets[:,0])
            self.tri_surface_node_indices    = device_put(pressure_surface_facets[:,1:])

        # label each node as layer 0 (real nodes), layer 1 (first layer of virtual nodes), etc
        node_layer_labels = np.load(f'{geometry_data_dir}/node-layer-labels.npy')
        self._real_node_indices = (node_layer_labels == 0)

        self._n_total_nodes = self._node_features.shape[0]
        self._n_edges= self._edge_features.shape[0]
        self._output_dim = self.ref_coords.shape[-1]

        if logging is not None:
            logging.info(f'{data_path} geometry summary:')
            logging.info(f'n_real_nodes  : {self._n_real_nodes}')
            logging.info(f'n_total_nodes : {self._n_total_nodes}')
            logging.info(f'n_edges       : {self._n_edges}')
            logging.info(f'elements.shape: {self.elements.shape}')
            logging.info(f'edges.shape   : {self._edge_features.shape}')
            logging.info(f'nodes.shape   : {self._node_features.shape}')
            if os.path.exists(pressure_surface_indicator_file):
                logging.info(f'tri_surface_area_normals.shape   : {self.tri_surface_area_normals.shape}\n')


class DataLoader:
    """Data loader for graph-formatted input-output simulation data with common, fixed topology"""

    def __init__(self, data_path: str, data_type: str):
        """
        Parameters
        ----------
        data_path: str
            Name of the subdirectory within "/data" where the data is stored
        data_type: str
            The type of data to be loaded, one of "train", "validation", "test"
        """

        full_data_path = f'data/{data_path}'
        if not os.path.isdir(full_data_path):
            raise NotADirectoryError(f'No directory at: {full_data_path}')

        sim_data_dir      = f'{full_data_path}/simulationData/{data_type}'
        stats_dir         = f'{full_data_path}/normalisationStatistics'
        geometry_data_dir = f'{full_data_path}/geometryData'

        sys.path.insert(0, geometry_data_dir)
        from constitutive_law import log_sampling

        # simulated displacements
        self._displacement = jnp.load(f'{sim_data_dir}/displacement.npy')

        # total potential energy values returned by the simulator
        self._pe_values = np.load(f'{sim_data_dir}/pe-values.npy')

        # global graph parameters for each simulation (generally material parameters)
        self._theta = device_put(jnp.load(f'{sim_data_dir}/theta.npy'))

        # normalisation statistics for theta
        self.theta_mean = jnp.load(f'{stats_dir}/theta-mean.npy')
        self.theta_std = jnp.load(f'{stats_dir}/theta-std.npy')

        # normalised theta values for input to GNN
        theta_norm = (((self._theta)) - self.theta_mean ) / self.theta_std
        self._theta_norm = device_put(theta_norm)

        # array of data point indices that can be iterated over during each epoch
        self._data_size = self._theta.shape[0]
        self._epoch_indices = np.arange(self._data_size)

    def get_graph(self, data_idx):
        """Returns input/output values for specified data point and places on GPU"""

        return device_put(self._theta_norm[data_idx:(data_idx+1)]), device_put(self._displacement[data_idx]), device_put(self._theta[data_idx]), self._pe_values[data_idx]

    def get_data(self, data_idx):
        """Returns input global graph values for specified data point (normalised and unnormalised)"""

        return self._theta_norm[data_idx], self._theta[data_idx]

    def return_index_0(self):
        """Returns input/output data ([V, E, theta]/ U) for first data point and places on GPU

        This method is used when initialising the parameters of the varying geometry GNN emulator
        """

        return device_put(self._nodes[0]), device_put(self._edges[0]), device_put(self._theta_vals[0:1]), device_put(self._shape_coeffs[0]), device_put(self._displacement[0]), device_put(self._theta[0])

    def shuffle_epoch_indices(self, seed_idx):
        """Shuffles the order in which the dataset is cycled through"""

        np.random.seed(seed_idx)
        self._epoch_indices = np.random.choice(self._data_size, self._data_size, False)

