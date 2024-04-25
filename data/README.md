# data

This directory contains simulation data and setup details for five mechanical models: [``OnceClampedBeam``](OnceClampedBeam), [``TwiceClampedBeam``](TwiceClampedBeam), [``TwistingBeam``](TwistingBeam), [``Liver``](Liver) and [``LeftVentricle``](LeftVentricle).

The above datasets have been already been processed into a format suitable for running emulation, as described in the parent directory [``README.md``](/README.md).

For completeness, data processing code is provided here. This code is contained inside [``data_process_utils.py``](data_process_utils.py), which can be called using [``data_main.py``](data_main.py) as is detailed below. The requirement formats for the raw unprocessed data to allow the above scripts to run are detailed in [``DATA_FORMAT_REQUIREMENTS.md``](DATA_FORMAT_REQUIREMENTS.md). We also provide an example of an unprocessed dataset generated using ``FEniCS`` in [``TwistingBeamUnprocessed``](TwistingBeamUnprocessed).

Note that an "unprocessed" dataset will still require some pre-processing of the results from e.g. FEniCS - for an example of how to apply this pre-processing, see [FenicsMeshProcessingLV.ipynb](fenicsMeshProcessing/FenicsMeshProcessingLV.ipynb).

For more details on augmented graph graph generation (i.e. the use of virtual nodes/edges), see Section 2.3 of [Dalton et. al (2022)](https://doi.org/10.1016/j.cma.2022.115645) and the associated [GitHub repository](https://github.com/dodaltuin/passive-lv-gnn-emul).

## Processing Data

All datasets were processes by calling ``data_main.py`` with default parameters. For instance, to process the data in [``TwistingBeamUnprocessed``](TwistingBeamUnprocessed), run:

```
python -m data_main --data_dir="TwistingBeamUnprocessed"
```


# Directory Layout

## Files:

### [``data_main.py``](data_main.py)

Main script for processing raw data into format suitable for PIGNN emulation

### [``data_process_utils.py``](data_process_utils.py)

Utility functions for processing raw data

### [``DATA_FORMAT_REQUIREMENTS.md``](DATA_FORMAT_REQUIREMENTS.md)

Details of the required format for raw data

## Subdirectories:


### [``OnceClampedBeam``](OnceClampedBeam) and [``TwiceClampedBeam``](TwiceClampedBeam)

Models used in Section 3.1 of the manuscript

### [``TwistingBeam``](TwistingBeam) 

Model with 343 nodes used in Section 3.2 of the manuscript

### [``Liver``](Liver)

Model used in Section 3.3 of the manuscript

### [``LeftVentricle``](LeftVentricle)

Slighly different model to that used in Section 3.4 of the manuscript - for details, see the [``README``](/README) in parent directory

