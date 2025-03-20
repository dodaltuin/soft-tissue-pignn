import meshio
import numpy as np
# referenced from https://gist.github.com/finsberg/2a919ad0beb0f4f169294ce34bea6fed#file-gmsh2dolfin-py

msh_file = "Part1.msh"
msh = meshio.gmsh.read(msh_file)

line_cells = []
for cell in msh.cells:
    if cell.type == "tetra":
       triangle_cells = cell.data
    elif cell.type == "triangle":
        if len(line_cells) == 0:
            line_cells = cell.data
        else:
            line_cells = np.vstack([line_cells, cell.data])
            

line_data = []
for key in msh.cell_data_dict["gmsh:physical"].keys():
     if key == "triangle":
         if len(line_data) == 0:
                line_data = msh.cell_data_dict["gmsh:physical"][key]
         else:
                line_data = np.vstack(
                    [line_data, msh.cell_data_dict["gmsh:physical"][key]]
                )
     elif key == "tetra":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
            
triangle_mesh = meshio.Mesh(
        points=msh.points,
        cells={"tetra": triangle_cells},
        cell_data={"name_to_read": [triangle_data]},
    )

line_mesh = meshio.Mesh(
        points=msh.points,
        cells=[("triangle", line_cells)],
        cell_data={"name_to_read": [line_data]},
    )

mesh_file = "mesh.xdmf"
line_file = "mf.xdmf"
meshio.write(mesh_file, triangle_mesh)
meshio.xdmf.write(line_file, line_mesh)


