"""
File: utils_visualisation.py
Author: David Dalton
Description: Generate .vtk and .ply files for visualisation of emulation results
"""

from paraview.simple import *
import numpy as np
import shutil
import os
import h5py
from absl import logging

h5_filename   = "sim_output.h5"
xdmf_filename = "sim_output.xdmf"

def find_quantile_indices(loss_array):
    """Find indices of the median and worst emulation errors"""

    # median (worst) error is 50th (100th) percentile
    loss_quantiles_labels = ['Median', 'Worst']
    loss_quantiles = [50, 100]

    # calculate the values which give rise to median/worst values
    loss_quantile_values = np.percentile(loss_array, loss_quantiles)

    # write results to dictionary
    index_dict = {}

    # loop over median and worst emulation results
    for i in range(len(loss_quantiles)):

        label_i = loss_quantiles_labels[i]
        loss_quantile_i = loss_quantile_values[i]

        # find the simulation which is closest to the quantile value calculated above
        index_i = np.argmin(np.abs(loss_array - loss_quantile_i))

        # write results to dictionary
        index_dict[label_i] = index_i

    return index_dict

def generate_savedir(label, index, results_save_dir):
    """Generate directory where visualisation files will be saved"""
    return f'{results_save_dir}resultsVisualisations/{label}_index{index}'

def copy_across_files(index_dict, true_disp, pred_disp, ref_coords, results_save_dir, data_save_dir):
    """Copy FEniCS template files and .npy results files to savedir"""

    # loop over median and worst emulation results
    for label, index in index_dict.items():

        # create directory to visualistion files
        savedir = generate_savedir(label, index, results_save_dir)
        if not os.path.isdir(savedir): os.makedirs(savedir)

        # save np arrays of results for this simulation index
        np.save(f'{savedir}/ref_geom.npy', ref_coords)
        np.save(f'{savedir}/true_disp.npy', true_disp[index])
        np.save(f'{savedir}/pred_disp.npy', pred_disp[index])
        np.save(f'{savedir}/losses.npy', true_disp[index] - pred_disp[index])

        #assert os.path.isfile(f'{data_save_dir}/sim_output.h5') and os.path.isfile(f'{data_save_dir}/sim_output.xdmf'), f"The files sim_output.h5 and sim_output.xdmf must exist in {data_save_dir} so that 3D visualisations can be made"

        # copy across FEniCS simulation output files for temporary writing of above information
        fenics_files_exist = os.path.isfile(f'{data_save_dir}/sim_output.h5') and os.path.isfile(f'{data_save_dir}/sim_output.xdmf')
        if not fenics_files_exist:
            logging.info(f"Warning - no 3D visualisations made because the template files sim_output.h5 and sim_output.xdmf do not exist in {data_save_dir}. For details on how these files can be generated, see data/DATA_FORMAT_REQUIREMENTS.md")
            exit()

        shutil.copy(f'{data_save_dir}/sim_output.h5',   f'{savedir}/{h5_filename}')
        shutil.copy(f'{data_save_dir}/sim_output.xdmf', f'{savedir}/{xdmf_filename}')


def generate_visualisation_files(index_dict, root_save_dir):
    """Generate .vtk and .ply visualisation files"""

    # loop over median and worst emulation results
    for label, index in index_dict.items():

        # create directory to visualistion files
        savedir = generate_savedir(label, index, root_save_dir)

        # find names of the npy files in savedir (without .npy extension)
        npy_files = [f[:-4] for f in os.listdir(savedir) if f.endswith('.npy')]


        for file_i in npy_files:

            # load the data from the .npy file generated in "copy_across_files"
            np_arr = np.load(f'{savedir}/{file_i}.npy')

            # create zero displacement for reference geometry for consistent plotting
            if file_i=="ref_geom": np_arr*=0.

            # write np array values to .h5 file
            f1 = h5py.File(f'{savedir}/{h5_filename}', "r+")
            f1['VisualisationVector']['0'].write_direct(np_arr)
            f1.close()

            # read .xdmf mesh file from fenics
            xdmf_geometry = XDMFReader(registrationName=xdmf_filename, FileNames=[f'{savedir}/{xdmf_filename}'])
            point_labels  = xdmf_geometry.PointArrayStatus[0]

            # extract surface from input geometry
            surface_model = ExtractSurface(registrationName='surface_model', Input=xdmf_geometry)

            # create reference geometry variable by extracting surface normals
            reference_configuration = GenerateSurfaceNormals(registrationName='reference_configuration', Input=surface_model)

            # update reference configuration by warping with given vectors
            updated_configuration = WarpByVector(registrationName='updated_configuration', Input=reference_configuration)
            updated_configuration.Vectors = ['POINTS', point_labels]

            # get colour map from the displacement vector magnitude
            colour_map = GetColorTransferFunction(point_labels)

            # save vtk files for paraview visualisation
            SaveData(f'{savedir}/{file_i}.vtk', proxy=updated_configuration, PointDataArrays=['Normals', point_labels])

            # save ply files for Augmented Reality visualistions
            SaveData(f'{savedir}/{file_i}.ply', proxy=updated_configuration, PointDataArrays=['Normals', point_labels],
                EnableColoring=1,
                ColorArrayName=['POINTS', point_labels],
                LookupTable=colour_map)


def make_3D_visualisations(Utrue, Upred, ref_coords, data_path, results_save_dir, logging):
    """Wrapper function which calls each of the above functions in turn"""

    pred_errors = (((Utrue - Upred)**2).sum(-1)**.5).mean(-1)

    index_dict = find_quantile_indices(pred_errors)

    logging.info(f'Index of simulation with median (worst) prediction errors: {index_dict["Median"]} ({index_dict["Worst"]})')

    copy_across_files(index_dict, Utrue, Upred, ref_coords, results_save_dir,f'data/{data_path}/geometryData')

    generate_visualisation_files(index_dict, results_save_dir)

