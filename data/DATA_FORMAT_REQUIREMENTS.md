# Data Format Requirements

For each model directory, two subdirectories are required, ``geometryData/`` and ``simulationData/``, which we detail here. Once these directories have been set up, [``data_main.py``](data_main.py) can be called to process the data into a format suitable for running the PIGNN emulator.We provide [``TwistingBeamUnProcessed/``](TwistingBeamUnProcessed//) as an example of an unprocessed directory - the other directories have already been processed.

For the below, we let *Nnode* (*Nelem*) denote the number of finite-element nodes (edges) in the underlying finite-element mesh geometry and *Nsim* the number of test simulations that have been performed. *Nedge* gives the number of nodes in the graph representation of the mesh, *Nfeat* gives the number of features each node is assigned, and *Nparam* is the number of material parameters in the constitutive law


## geometryData/

### REQUIRED:

```constitutive_law.py```   : Python script which initialises the constitutive law of the material and some other properties. For further details, see the layout of this file in any of models in this subdirectory.


```elements.npy```          : (*Nelem* * 4) array, where the $ith$ row gives the indices of the four nodes in the $ith$ element in the mesh.

```interior-points.npy```   : (*Nnode* * 1) array of boolean values where the $ith$ row indicatese if the $ith$ node in the FE mesh is not on a Dirichlet boundary.

```real-node-features.npy```: (*Nnode* * *Nfeat) array giving feature vectors assigned to each node in the finite-element mesh.

```real-node-topology.npy```: (*Nedge* * 2) array giving sender/receiver pairs in the graph representation of the finite-element mesh.

```reference-coords.npy```  : (*Nnode* * 3) array giving coordinates of the geometry in its reference configuration.


### OPTIONAL:


```disp-add.npy```: (*Nnode* * 3) array used to satisfy inhomogeneous Dirichlet boundary conditions (as for example in the *TwistingBeam* model, see its [``constitutive_law.py``](TwistingBeam/geometryData/constitutive_law.py) file for more details).

```fibre-field-elementwise.npy```: (*Nelem* * 3) array of fibre orientations for each element in mesh, if material is non-isotropic (as for example in LeftVentricle model).

```pressure-surface-facets.npy```: (*Nfacet* * 4) array of which gives details of the facets which make up the surface to which a pressure/traction force is applied. Only required if the model includes a traction force (like the *LeftVentricle* model). The first element of each row gives the index of the element that the triangular facet belongs to (we are assuming a tetrahedral mesh), while the final three elements give the indices of the three nodes which make up the facet .  For further details, see the file [``LeftVentricle/geometryData/pressure-surface-facets.npy``](LeftVentricle/geometryData/pressure-surface-facets.npy).

```sim_output.xdmf``` and ```sim_output.h5```: files used to generate 3D .vtk and .ply files to visualise emulation results. These files can be generated in FEniCS as follows: can be generated using FEniCS by running the following in Python after the displacement ``u`` has been solved for:

```
file = XDMFFile("sim_output.xdmf")
file.write(u, 0)
```

## simulationData/


```displacement.npy```: (*Nsim* * *Nnode* * 3) array giving displacement arrays for all *Nsim* test simulations.

```theta.npy```       : (*Nnode* * *Nparam*) array giving global material parameter values that the above simulations were run with. 

```pe-values.npy```   : (*Nnode*,) array giving the minimum potential energy state that corresponds to the above simulationr results, computed using FEniCS
