"""
File: data_process_utils.py
Author: David Dalton
Description: Utility functions to process raw simulation data
"""

import os

import numpy as np
from numpy import newaxis

import os
import shutil

import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_debug_nans', True)
from jax import vmap
import jax.numpy as jnp

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from functools import partial
from typing import Sequence

from absl import logging

# some strings printed to console to seperate different results sections
SECTION_SEPERATOR = '##############################'
SECTION_SEPERATOR2 = SECTION_SEPERATOR * 2

# we normalise shape coefficients with respect to the variance on the below indexed column
REFERENCE_SHAPE_COEFF_COLUMN = 1

####################################
## Augmented Graph Topology Generation
####################################

def calc_n_nodes_per_layer(n_real_nodes: int, root_leaf_factor: int, min_root_nodes: int) -> Sequence[int]:
    """Calculates number of nodes in each layer of the augmented graph

    Parameters:
    -----------
    n_real_nodes: int
        The number of real nodes in the original graph representation
    root_leaf_factor: int
        The number of leaves to collapse into one root node (on average)
    min_root_nodes: int
        The minimum number of allowable root nodes in a single layer

    Returns:
    ----------
    n_nodes_per_layer: Sequence[int]
        A list of integers giving the number of nodes in each layer of the augmented graph.
        The 0'th index is the number of real nodes. The remainder gives the number in each
        of the virtual node layers, which is referred to as the vector $\mathbf{n}$ in
        Algorithm 1 of the manuscript. Each value is found by recursively integer dividing
        the previous value by "root_leaf_factor", starting with the value "n_real_nodes".
        This continues until the resulting value is less than the threshold specified by
        "min_root_nodes", when the routine stops
    """

    # initialise list to hold count of number of nodes in each node layer
    nodes_count_per_layer = []
    nodes_count_per_layer.append(n_real_nodes)

    i = 1
    while True:

        # number of nodes in layer i is the number of nodes from the previous layer
        # (integer) divided by "root_leaf_factor"
        nodes_count_layer_i = nodes_count_per_layer[i-1] // root_leaf_factor

        # exit if we have reached the specificed min number of virtual nodes allowed in a layer
        if nodes_count_layer_i < min_root_nodes:
            break

        nodes_count_per_layer.append(nodes_count_layer_i)

        i += 1

    return nodes_count_per_layer


def intra_layer_topology(positions: np.array, nn_count: int, shift_value: int) -> np.array:
    """Calculates connectivity between a single layer of virtual nodes

    Parameters:
    -----------
    positions: np.array
        An array giving the euclidean coordiantes of the virtual nodes to be connected
    nn_count: int
        The number of nearest neigbours to consider when forming node connections
    shift_value: int
        Amount by which (local) node indices need to be adjusted to align with
        their indices from the global graph structure

    Returns:
    ----------
    intra_layer_topology: np.array
        np.array representing the topology betweent the virtual node whose coordinates are
        given in positions. The array consists of two columns, the first being the sender
        node indices, and the second the receiver nodes. Note how these indices are shifted
        by "shift_value" so that they align with the global graph strucutre. The connections
        are found by connecting each node to its "nn_count" neighbours among all other nodes
    """

    # find "nn_count" nearest neighbours of each point in "positions"
    nodes_count = positions.shape[0]
    nbrs = NearestNeighbors(n_neighbors=(nn_count+1), algorithm='ball_tree').fit(positions)

    # find the indices of these nearest neighbours
    _, indices = nbrs.kneighbors(positions)

    # sender indices are simply 0, 1, ..., nodes_count
    senders = np.repeat(np.arange(nodes_count), nn_count).reshape(-1,1)

    # receiver indices come from the KNN results above
    receivers = indices[:,1:].flatten().reshape(-1,1)

    # indices of both senders and receivers are shifted to make sure they make sense
    # with respect to the entire augmented graph
    senders += shift_value
    receivers += shift_value

    return np.concatenate((senders, receivers), axis=1)


def intra_layer_topology_final(positions: np.array, shift_value: int) -> np.array:
    """Calculates complete connectivity between the last layer of virtual nodes

    Parameters:
    -----------
    positions: np.array
        An array giving the euclidean coordiantes of the virtual nodes to be connected
    shift_value: int
        Amount by which (local) node indices need to be adjusted to align with
        their indices from the global graph structure

    Returns:
    ----------
    intra_layer_topology: np.array
        np.array representing the topology between the final layer of virtual nodes,
        whose coordinates are given in positions, in the same format as outputted by
        the function "intra_layer_topology" above. In the last layer of nodes however,
        the nodes are fully connected
    """

    # create a directed edge between each pair of nodes
    n_nodes= positions.shape[0]
    topology = np.array([[i,j] for i in range(n_nodes) for j in range(n_nodes) if i != j])

    # the indices of topology are shifted so that they
    # make sense with respect to the entire augmented graph
    return topology + shift_value

def inter_layer_topology(leafnode_indices: Sequence[int], rootnode_indices: Sequence[int]) -> np.array:
    """Calculates connectivity from leaf to root nodes

    Parameters:
    -----------
    leafnode_indices: Sequence[int]
        Indices of the leaf nodes
    rootnode_indices: Sequence[int]
        Indices of the root nodes

    Returns:
    ----------
    intra_layer_topology: np.array
        np.array with two columns representing the connections from leaf to root nodes.
        The first column is simply the indices of the leaf nodes, "leafnode_indices", and
        the second column gives the indices of the root nodes, "rootnode_indices"
    """

    # the leaf nodes become senders
    senders = leafnode_indices

    # the root nodes the receivers
    receivers = rootnode_indices

    return np.concatenate((senders, receivers), axis=1)

def calculate_virtual_nodes_and_edges(real_nodes: np.array,
                                      real_topology: np.array,
                                      n_nearest_nbrs: int,
                                      n_nodes_per_layer: Sequence[int],
                                      n_leaf_nodes: int,
                                      min_root_nodes: int):
    """Generate Augmented Topology (Algorithm 1 from the manuscript)

    Parameters:
    -----------
    Are as explained in "generate_augmented_topology()" below

    Returns:
    ----------
    tuple
       The first value of which is the augmented graph topology used for GNN
       message passing for all graphs under considieration. Some other variables
       are returned to assist in generating the virtual nodes for different graphs
       of the class of problem under consideration
    """

    logging.info(SECTION_SEPERATOR)
    logging.info('Calling calculate_virtual_nodes_and_edges()')

    # initialise list to keep track of the root node indices for each leaf node
    # across all virtual node layers
    kmeans_labels_list = []

    # calculate number of nodes in each virtual node layer
    n_real_nodes = real_nodes.shape[0]
    if n_nodes_per_layer is None:
        n_nodes_per_layer = calc_n_nodes_per_layer(n_real_nodes, n_leaf_nodes, min_root_nodes)

    n_total_nodes = sum(n_nodes_per_layer)
    n_node_layers = len(n_nodes_per_layer)
    n_virtual_node_layers = n_node_layers - 1
    logging.info(f'n_virtual_node_layers: {n_virtual_node_layers}')
    logging.info(f'n_nodes_per_layer: {n_nodes_per_layer}')

    # boolean array which labels each real and virtual node with its respective node layer index, 0, ... n_node_layers-1
    node_layer_labels_nested = [[i]*count_i for i, count_i in enumerate(n_nodes_per_layer)]
    node_layer_labels = np.array([item for sublist in node_layer_labels_nested for item in sublist])

    # arbitrary (0-based) indexing of all nodes (real and virtual)
    node_indices = np.arange(n_total_nodes).reshape(-1,1)

    # dimensionality of the data (2d for beam data, 3d for LV data)
    n_dim = real_nodes.shape[1]

    # initialise all nodes (real and augmented, or $V \cup \tilde(V)$)
    all_nodes = np.zeros((n_total_nodes, n_dim), dtype=np.float32)
    all_nodes[node_layer_labels == 0] = real_nodes

    # initialise augmented topology ($\tilde(E)$) as np array
    augmented_topology = real_topology.copy()

    for layer_num in range(1, n_node_layers):

        logging.info(f'Processing virtual node layer {layer_num}')

        # the leaf nodes of virtual node layer "layer_num" are the nodes from the previous layer "layer_num-1"
        leaf_node_indices = node_indices[node_layer_labels == (layer_num - 1)]
        leaf_nodes = all_nodes[node_layer_labels == (layer_num - 1)]

        # cluster 'leaf_nodes' into 'nodes_count_per_layer[layer_num]' clusters
        kmeans = KMeans(n_clusters = n_nodes_per_layer[layer_num], random_state=0).fit(leaf_nodes)

        # the cluster centers obtained via k-means become the root nodes of this layer
        root_nodes = kmeans.cluster_centers_

        # write these coords to the array holding all node coords, at the correct rows
        all_nodes[node_layer_labels == layer_num] = root_nodes

        # save list which labels each leaf node the index of its root node found using kmeans above
        kmeans_labels_list.append(kmeans.labels_)

        # need to shift all node indices so that they make sense with respect to the entire augmented graph
        shift_value = np.sum(node_layer_labels < (layer_num))
        root_node_indices = (kmeans.labels_ + shift_value).reshape(-1,1)

        # calculate connectivty between the leaf and root nodes
        leaf_root_topology = inter_layer_topology(leaf_node_indices, root_node_indices)

        # calculate connectivity between the root nodes using "n_nearest_nbrs"
        if layer_num < n_virtual_node_layers:
            root_root_topology = intra_layer_topology(root_nodes, n_nearest_nbrs, shift_value)
        else:
            logging.info('Generating fully connected final virtal node layer\n')
            root_root_topology = intra_layer_topology_final(root_nodes, shift_value)

        # concatenate new node connections onto existing topology
        for new_topology in [leaf_root_topology, root_root_topology]:
            augmented_topology = np.concatenate((augmented_topology, new_topology), axis=0)

    # for each directed edge e_{ij}, we require that there is a twin edge e_{ji}, to ensure
    # conservation of momentum in the message passing stage of the GNN. The below makes
    # sure that this is the case
    reverse_topology = np.concatenate((augmented_topology[:,1:2], augmented_topology[:,0:1]), axis=1)
    augmented_topology = np.concatenate((augmented_topology, reverse_topology), axis=0)
    # remove duplicates
    augmented_topology = np.unique(augmented_topology, axis=0)

    return augmented_topology, kmeans_labels_list, node_layer_labels, all_nodes


def generate_augmented_topology(data_dir: str,
                                existing_topology_dir: str,
                                n_nearest_nbrs: int,
                                n_nodes_per_layer: Sequence[int],
                                n_leaves: int,
                                min_root_nodes: int):
    """Generate augmented topology on a representative geometry and process results

    Parameters:
    -----------
    Are as explained in "generate_augmented_topology()" below

    data_dir: str
       The directory where the raw simulation data is stored

    existing_topology_dir: str
       If using the topology data not stored in /data_dir, but instead existing
       augmented topology data already computed, set the path here

    n_nearest_nbrs: int
       Number of nearest neighbours to consider

    n_nodes_per_layer: Sequence[int] (OPTIONAL)
       As described in "calc_n_nodes_per_layer" above

    If n_nodes_per_layer is None, the below two inputs are used to calculate
    n_nodes_per_layer automatically:

    n_leaves: int
        The number of leaves to collapse into one root node (on average)

    min_root_nodes: int
        The minimum number of allowable root nodes in a single layer


    Returns:
    ----------
    Nothing is returned, instead the augmented topology for a fixed representative
    geometry is computed as in Algorithm 1 of the manuscript, with all results processed
    and saved for use for generating the augmented graph representation of all other
    simulation inputs.
    """


    logging.info(SECTION_SEPERATOR2)
    logging.info('Calling generate_augmented_graph()')

    # copy augmented topology data from "existing_topology_dir" if it exists
    if existing_topology_dir != "None":

        if not os.path.isdir(existing_topology_dir):
            raise NotADirectoryError(f'No directory at: {existing_topology_dir}')

        new_topology_dir = f'{data_dir}/topologyData'
        logging.info(f'Copying "{existing_topology_dir}" to  "{new_topology_dir}"')
        shutil.copytree(existing_topology_dir, new_topology_dir, dirs_exist_ok=False)

        return

    logging.info(f'n_leaves: {n_leaves}')
    logging.info(f'n_nearest_nbrs: {n_nearest_nbrs}')
    logging.info(SECTION_SEPERATOR2 + '\n')

    # load real node coordinates of representative graph
    ref_coords = np.load(f'{data_dir}/geometryData/reference-coords.npy')
    logging.info(f'ref_coords.shape: {ref_coords.shape}')

    # load the topology of these nodes
    real_topology = np.load(f'{data_dir}/geometryData/real-node-topology.npy').astype(np.int32)
    logging.info(f'real_topology.shape: {real_topology.shape}\n')

    # calculate the virtual nodes and edges of the augmented graph using Algorithm 2 of the manuscript
    results_tuple = calculate_virtual_nodes_and_edges(ref_coords, real_topology, n_nearest_nbrs, n_nodes_per_layer, n_leaves, min_root_nodes)

    # split out results tuple into its individual arrays
    augmented_topology, kmeans_labels_list, node_layer_labels, ref_augmented_coords = results_tuple

    # Only half of the directed edges in the augmented graph need to be assigned an edge feature vector,
    # because we enforce symmetry in the messages passed on pairs of directed edges. See Algorithm 3 of the manuscript
    # The edges we assign features vectors to are those in the "sparse_topology" defined below
    sparse_topology_indices = augmented_topology[:,0] > augmented_topology[:,1]
    sparse_topology = augmented_topology[sparse_topology_indices]

    logging.info(f'augmented_topology.shape    : {augmented_topology.shape}')
    logging.info(f'sparse_topology.shape       : {sparse_topology.shape}')
    logging.info(f'node_layer_labels.shape     : {node_layer_labels.shape}')
    logging.info(f'np.unique(node_layer_labels): {np.unique(node_layer_labels)}')
    logging.info(f'rep_augmented_coords.shape  : {ref_augmented_coords.shape}')
    logging.info(f'len(kmeans_labels_list)     : {len(kmeans_labels_list)}')

    # convert kmeans_labels_list to np array for saving
    if len(kmeans_labels_list) > 1:
        kmeans_labels_list = np.array(kmeans_labels_list, dtype=object)
    else:
        kmeans_labels_list = np.array(kmeans_labels_list, dtype=np.int32)

    logging.info(f'Saving generate_augmented_graph() results\n')
    np.save(f'{data_dir}/geometryData/augmented-topology.npy', augmented_topology)
    np.save(f'{data_dir}/geometryData/sparse-topology.npy', sparse_topology)
    np.save(f'{data_dir}/geometryData/reference-augmented-coords.npy', ref_augmented_coords)
    np.save(f'{data_dir}/geometryData/kmeans-labels-list.npy', kmeans_labels_list)
    np.save(f'{data_dir}/geometryData/node-layer-labels.npy', node_layer_labels)


####################################
## Augmented Node Generation
####################################

def aggregate_leaf_nodes(leaf_nodes: jnp.array, labels_list: Sequence[int], n_root_nodes: int) -> jnp.array:
    """Computes augmented nodes for all graphs in given datasets

    Parameters:
    -----------
    leaf_nodes: jnp.array
        Array giving the coords / features of the leaf node
    labels_list: Sequence[int]
        List which labels each leaf node in "leaf_nodes" to the corresponding root
        node to which it is grouped
    n_root_nodes: int
        Number of root nodes that "leaf_nodes" are being clustered to

    Returns:
    ----------
    root_nodes: jnp.array
       Returns the coords (features) of the each root node, found by
       averaging the coords (features) of their corresponding leaf nodes
    """

    # root node co-ordinates / features are the mean of all their respective leaf nodes
    sum_values = jax.ops.segment_sum(leaf_nodes, labels_list, n_root_nodes)

    # cardinalities of each root node cluster
    norm_values = jax.ops.segment_sum(np.ones((leaf_nodes.shape[0],1)), labels_list, n_root_nodes)

    # divide sum by cardinality to get average value
    root_nodes = sum_values / norm_values

    return root_nodes


def compute_augmented_nodes(real_nodes: jnp.array, real_node_features: jnp.array, node_layer_labels: Sequence[int], kmeans_labels_list: Sequence[Sequence[int]]):
    """Computes augmented node coords and features for given input graph

    Parameters:
    -----------
    real_nodes: jnp.array
        Array giving the coords of the real nodes in the graph
    real_node_features: jnp.array
        Array giving the feature vectors of the real nodes in the graph
    node_layer_labels: Sequence[int] >= 0
        Labels each node (real and virtual) with the node layer it is part of,
        starting with 0 for real nodes, 1 for first layer of virtual nodes, and
        so on
    kmeans_labels_list: Sequence[Sequence[int]]
        List of lists, where each element $l$ is a list giving the indices of the cluster
        centre to which the leaf nodes from step $l$ of Algorithm 1 of the manuscript were
        assigned

    Returns:
    ----------
    augmented_node_features: jnp.array
       Returns the augmented node features, found by recursively averaging the features
       from the previous layer, with respect to the common node topology as implied by "kmeans_labels"
    """

    n_dim1 = real_nodes.shape[-1]
    n_dim2 = real_node_features.shape[-1]

    # initialise arrays to hold augmented node coords and features
    augmented_nodes = jnp.zeros((node_layer_labels.shape[0], n_dim1), dtype=jnp.float32)
    augmented_node_features = jnp.zeros((node_layer_labels.shape[0], n_dim2), dtype=jnp.float32)

    # 0th layer of augmented nodes are simply the real nodes
    augmented_nodes = augmented_nodes.at[node_layer_labels==0].set(real_nodes)
    augmented_node_features = augmented_node_features.at[node_layer_labels==0].set(real_node_features)

    # loop over each layer of virtual nodes
    for i, labels_list_i in enumerate(kmeans_labels_list):

        leaf_nodes_i = augmented_nodes[node_layer_labels==i]
        leaf_node_features_i = augmented_node_features[node_layer_labels==i]

        # number of root nodes in this layer
        n_root_nodes = np.sum(node_layer_labels == (i+1))

        # aggregate leaf node coords to find root node coords
        root_nodes = aggregate_leaf_nodes(leaf_nodes_i, labels_list_i, n_root_nodes)
        augmented_nodes = augmented_nodes.at[node_layer_labels == (i+1)].set(root_nodes)

        # aggregate leaf node features to find root node features
        root_node_features = aggregate_leaf_nodes(leaf_node_features_i, labels_list_i, n_root_nodes)
        augmented_node_features = augmented_node_features.at[node_layer_labels == (i+1)].set(root_node_features)

    return augmented_node_features


def generate_augmented_nodes(data_dir: str):
    """Computes augmented nodes for all graphs in given datasets

    Parameters:
    -----------
    data_dir: str
        Path to directory where the data is stored

    Returns:
    ----------
    No values are returned, instead the function "compute_all_augmented_nodes" defined above is called
    """

    logging.info(SECTION_SEPERATOR2)
    logging.info('Calling generate_augmented_nodes()')
    logging.info(SECTION_SEPERATOR2 + '\n')

    # real node co-ordinates of the reference geometry
    ref_real_nodes = np.load(f'{data_dir}/geometryData/reference-coords.npy')

    # real node features of the reference geometry
    real_node_features = np.load(f'{data_dir}/geometryData/real-node-features.npy')

    # augmented topology ($E$ in manuscript) calculated on representative graph using the function "generate_augmented_graph()" defined above
    augmented_topology = np.load(f'{data_dir}/geometryData/augmented-topology.npy')

    # augmented node co-ordinates of the reference geometry
    ref_augmented_coords = np.load(f'{data_dir}/geometryData/reference-augmented-coords.npy')

    # gives the root node label in the immedietely higher layer of virtual nodes, for all nodes but laster layer of virtual nodes
    kmeans_labels_list = np.load(f'{data_dir}/geometryData/kmeans-labels-list.npy', allow_pickle=True)#.astype(np.int32)

    # variable indicating which layer each augmented node belongs to
    node_layer_labels = np.load(f'{data_dir}/geometryData/node-layer-labels.npy')

    # compute features of augmented nodes
    augmented_node_features = compute_augmented_nodes(ref_real_nodes, real_node_features, node_layer_labels, kmeans_labels_list)

    logging.info(f'augmented_node_features.shape: {augmented_node_features.shape}\n')


    node_features_mean = augmented_node_features.mean(0)
    node_features_std  = augmented_node_features.std(0)

    logging.info(f'node_features_mean: {node_features_mean}')
    logging.info(f'node_features_std : {node_features_std}')

    augmented_node_features = (augmented_node_features- node_features_mean) / node_features_std

    logging.info(f'saving node features')
    np.save(f'{data_dir}/geometryData/augmented-node-features.npy', augmented_node_features)

    stats_save_dir = f'{data_dir}/normalisationStatistics'
    if not os.path.isdir(stats_save_dir): os.mkdir(stats_save_dir)

    jnp.save(f'{stats_save_dir}/node-features-mean.npy', node_features_mean)
    jnp.save(f'{stats_save_dir}/node-features-std.npy' , node_features_std)


####################################
## Edge Feature Generation
####################################

def nodes_to_edge_features(nodes: jnp.array, senders: Sequence[int], receivers: Sequence[int]) -> jnp.array:
    """Computes edge features for all edges in graph simultaneously

    Parameters:
    -----------
    nodes: jnp.array
        Array of coords of all nodes (real and augmented) in graph
    senders: Sequence[int]
        Indices of sender nodes
    receivers: Sequence[int]
        Indices of receiver nodes

    Returns:
    ----------
    edge_features: jnp.array
        Array where each row gives the edge feature vector of the corresponding edge.
        The edge feature gives the relative position of the receiver node with respect
        to the sender node, and the distance between the two nodes

    """

    n_dim = nodes.shape[-1]

    # array of relative differences between senders and receiver nodes
    node_diff = nodes[receivers,:n_dim] - nodes[senders,:n_dim]

    # euclidean norm of each relative node differences (reshaped to 2d array)
    node_distance = jnp.sqrt((node_diff**2).sum(axis=1)).reshape(node_diff.shape[0],1)

    # edge features assigned to each edge is the concatenation of the above two arrays
    edge_features = jnp.concatenate((node_diff, node_distance), axis=1)

    return edge_features

def generate_edge_features(data_dir: str):
    """Computes edge features for reference geometry

    Parameters:
    -----------
    data_dir: str
        Path to directory where the data is stored

    Returns:
    ----------
    No values are returned, but the computed edge features are saved to data_dir/geometryData
    """

    logging.info(SECTION_SEPERATOR2)
    logging.info('Calling generate_edge_features()')
    logging.info(SECTION_SEPERATOR2 + '\n')

    # load "sparse" topology - one edge from each pair of "twin" directed edges in full augmented topology
    sparse_topology = jnp.load(f'{data_dir}/geometryData/sparse-topology.npy')
    logging.info(f'sparse_topology.shape: {sparse_topology.shape}\n')

    # load representative augmented coords
    ref_augmented_coords = np.load(f'{data_dir}/geometryData/reference-augmented-coords.npy')

    edge_features = nodes_to_edge_features(ref_augmented_coords, sparse_topology[:,0], sparse_topology[:,1])

    logging.info(f'edge_features.shape: {edge_features.shape}')

    edge_features_mean = edge_features.mean(0)
    edge_features_std  = edge_features.std(0)

    logging.info(f'edge_features_mean: {edge_features_mean}')
    logging.info(f'edge_features_std : {edge_features_std}')

    edge_features = (edge_features - edge_features_mean) / edge_features_std

    logging.info(f'saving edge features')
    jnp.save(f'{data_dir}/geometryData/edge-features.npy', edge_features)

    jnp.save(f'{data_dir}/normalisationStatistics/edge-features-mean.npy', edge_features_mean)
    jnp.save(f'{data_dir}/normalisationStatistics/edge-features-std.npy' , edge_features_std)

