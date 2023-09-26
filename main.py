"""
File: main.py
Author: David Dalton
Description: Wrapper script to call PI-GNN emulation
"""

#############################################################################
## Package imports
#############################################################################

from absl import app
from absl import flags
from utils_training import run_fns_dict

#############################################################################
## Set up shell input variables
#############################################################################

flags.DEFINE_enum(   'mode', 'train_and_evaluate', ['train', 'evaluate', 'train_and_evaluate'], help = 'Fit model to training data or evaluate on test data')
flags.DEFINE_integer('n_epochs', 1000, lower_bound = 1, help = 'Number of epochs to train the model for')
flags.DEFINE_string( 'data_path', 'LeftVentricle', help = 'Name of sub-directory in "/data" where model data is stored')
flags.DEFINE_float(  'lr', 1e-4, lower_bound=0, help='Learning rate for training the network')
flags.DEFINE_string( 'trained_params_dir', "None", help='Path to directory with pre-trained network parameters')
flags.DEFINE_integer('K', 5, lower_bound=1, help='Number of message passing steps to perform')
flags.DEFINE_string( 'dir_label', '', help='Optional label to append to end of results save directory')
FLAGS = flags.FLAGS

def main(_):

    for run_fn in run_fns_dict[FLAGS.mode]:

        run_fn(FLAGS.data_path,
               FLAGS.K,
               FLAGS.n_epochs,
               FLAGS.lr,
               FLAGS.trained_params_dir,
               FLAGS.dir_label
               )

if __name__ == "__main__":
    app.run(main)

