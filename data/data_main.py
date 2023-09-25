"""
File: data_main.py
Author: David Dalton
Description: Data processing script

Processes data from raw format into augmented graph format, with node and edge features assigned.

For details on the augmented graph format, see Dalton, CMAME 2022 and: https://github.com/dodaltuin/passive-lv-gnn-emul

Example usage: the following command can be used to process the Liver data:

python -m data_main --mode="run_all" --data_dir="Liver"

Once data processing is complete, emulator training and validation can be performed from the main directory
"""

import data_process_utils as utils

from typing import Sequence

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_enum('mode', 'run_all', ['generate_topology', 'generate_nodes', 'generate_edges', 'run_all'], help = 'Select which data processing function to run')
flags.DEFINE_string('data_dir', 'TwistingBeamUnprocessed', help='Path to directory where raw data is stored')
flags.DEFINE_string('existing_topology_dir', "None", help='Path to directory where raw data is stored')
flags.DEFINE_string('existing_stats_dir', "None", help='(optional) Path to directory where normalisation summary stats are saved')
flags.DEFINE_integer('n_nearest_nbrs', 5, lower_bound=1, help='number of nearest neighbours to consider when defining root-root topology')
flags.DEFINE_integer('n_leaves', 4, lower_bound=2, help='(approximate) number of leaves each root node should have')
flags.DEFINE_integer('min_root_nodes', 2, lower_bound=1, help='minimum number of nodes to have final virtual layer')
FLAGS = flags.FLAGS

# (optional) hard code list of decreasing integers specifying the number of virtual nodes to create at each layer. If set to None, it will be calculated automatically using the shell input variables "n_leaves" and "min_root_nodes" defined above
n_nodes_per_layer: Sequence[int] = None

def main(_):

    if not utils.os.path.isdir(FLAGS.data_dir):
        raise NotADirectoryError(f'No directory at: {FLAGS.data_dir}')

    if FLAGS.mode in ['run_all', 'generate_topology']:
        utils.generate_augmented_topology(FLAGS.data_dir,
                                          FLAGS.existing_topology_dir,
                                          FLAGS.n_nearest_nbrs,
                                          n_nodes_per_layer,
                                          FLAGS.n_leaves,
                                          FLAGS.min_root_nodes)

    if FLAGS.mode in ['run_all', 'generate_nodes']:
        utils.generate_augmented_nodes(FLAGS.data_dir)

    if FLAGS.mode in ['run_all', 'generate_edges']:
        utils.generate_edge_features(FLAGS.data_dir)

if __name__ == "__main__":
    app.run(main)


