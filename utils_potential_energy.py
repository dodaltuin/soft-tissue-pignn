"""
File: utils_potential_energy.py
Author: David Dalton - adjusted from Matlab code written by Hao Gao (https://github.com/HaoGao)
Description: Utility functions for calculating total potential energy
"""

import jax
import jax.numpy as jnp

from typing import Callable

def compute_area_N_facet(facet: jnp.ndarray, coords: jnp.ndarray, elements: jnp.ndarray):
    """Compute area and normal vector N for a triangular surface facet """

    # indices of the 3 nodes at the corner of the triangular
    node_indices = facet[1:]

    # coordinates of the corner nodes
    node_coords = coords[node_indices]

    # the centre point of the triangular facet
    face_centre = node_coords.mean(0)

    # indices of the element to which the surface facet belongs
    elem_index = facet[0]

    # indices of the 4 nodes at the corners of the tetrahedral element
    nodes_elem_index = elements[elem_index]

    # coordinates of the 4 corner nodes
    elem_nodes = coords[nodes_elem_index]

    # the centre point of the element
    elem_centre = elem_nodes.mean(0)

    # compute vectors which describe 3 sides of the triangular facet
    p0, p1, p2 = node_coords
    p1_0 = p1 - p0
    p2_0 = p2 - p0
    p2_1 = p2 - p1

    # find lengths of each side
    s0 = jnp.sqrt(((p1_0**2).sum()))
    s1 = jnp.sqrt(((p2_0**2).sum()))
    s2 = jnp.sqrt(((p2_1**2).sum()))

    # use Heron's formula to compute the area of the triangle
    s = (s0 + s1 + s2) / 2.
    area = jnp.sqrt(s*(s-s0)*(s-s1)*(s-s2))

    # the surface normal vector is orthognal to the plane in which the edge vectors lie
    Nt = jnp.cross(p2_0, p1_0)

    # rescale normal vector N to unit length
    N_mag = jnp.sqrt(((Nt**2).sum()))
    N = Nt / N_mag

    # ensure that N is pointing outwards
    v_in = elem_centre - face_centre
    N = jnp.where(jnp.dot(v_in, N) > 0, -N, N)

    return area, N

def compute_element_vol(element: jnp.ndarray, ref_coords: jnp.ndarray):
    """Compute volume of a tetrahedral element (reference configuration)"""

    # recentre coords to have origin equal to the first coord
    v10 = ref_coords[element[1]] - ref_coords[element[0]]
    v20 = ref_coords[element[2]] - ref_coords[element[0]]
    v30 = ref_coords[element[3]] - ref_coords[element[0]]

    # find tetrahedron volume using standard formula (see here for example: https://en.wikipedia.org/wiki/Tetrahedron#General_properties)
    v20_cross_v10 = jnp.cross(v20, v10)
    element_vol = jnp.abs(jnp.dot(v30, v20_cross_v10))/6.

    return element_vol

def compute_def_gradient_element(element: jnp.ndarray, ref_coords: jnp.ndarray, cur_coords:jnp.ndarray, Jtransform = lambda J: J):
    """Compute deformation gradient of a tetrahedral element given specified reference and current positions"""

    # extract reference and current coords of element
    node_0 = ref_coords[element]
    node_1 = cur_coords[element]

    # calculate the deformation gradient F as in Eq. (17) of the manuscript
    node_0_centered = (node_0[1:] - node_0[0]).T
    node_1_centered = (node_1[1:] - node_1[0]).T
    F = jnp.matmul(node_1_centered, jnp.linalg.inv(node_0_centered))

    # J is simply the determinant of F
    J = jnp.linalg.det(F)

    # apply transformation to J to prevent negative values or values close to zero (see Section 2.4.3 of the manuscript)
    J = Jtransform(J)

    return F, J


def compute_internal_work_element(F: jnp.ndarray, J: jnp.ndarray, elem_vol: jnp.ndarray, elem_fibre: jnp.ndarray, theta: jnp.ndarray, constitutive_law: Callable):
    """Compute internal work done in single element"""

    # internal work in single element computed as from first summand in Eq. (20) of the manuscript
    Psi_internal = constitutive_law(theta, F, J, elem_fibre) * elem_vol

    return Psi_internal


def compute_body_work_element(element: jnp.ndarray, elem_vol: jnp.ndarray, disp: jnp.ndarray, body_force: jnp.ndarray):
    """Compute body work done in single element"""

    # body work in single element computed as from second summand in Eq. (20) of the manuscript
    element_disp = disp[element].mean(0)
    body_force_work = (body_force * element_disp).sum() * elem_vol

    return body_force_work


def compute_surface_work_element(F: jnp.ndarray, J: jnp.ndarray, disp: jnp.ndarray, area_normals: jnp.ndarray, surface_force: jnp.ndarray):
    """Compute surface work done on single triangular surface facet"""

    # Nanson's formula (Eq. (18) of the manuscript)
    JNdA = -J*jnp.matmul(jnp.linalg.inv(F).T, area_normals)

    # $u_A^{fl}$ in the manuscript
    surface_disp = disp.mean(0)

    # surface work on single triangular facet computed as from third summand in Eq. (20) of the manuscript
    pressure_work = jnp.dot(surface_disp, JNdA) * surface_force

    return pressure_work


# vmap above functions which are defined per element/surface facet, to work over an entire mesh
compute_area_N        = jax.vmap(compute_area_N_facet, in_axes=[0, None, None])
compute_vol           = jax.vmap(compute_element_vol, [0, None])
compute_def_gradient  = jax.vmap(compute_def_gradient_element, in_axes = [0] + [None]*3)
compute_internal_work = jax.vmap(compute_internal_work_element, in_axes = [0]*4 + [None]*2)
compute_body_work     = jax.vmap(compute_body_work_element, in_axes = [0]*2 + [None]*2)
compute_surface_work  = jax.vmap(compute_surface_work_element, in_axes = [0]*4 + [None])


def total_potential_energy(displacement: jnp.ndarray, theta: jnp.ndarray, ref_geom_data, external_forces):
    """Compute total potential energy by evaluating Eq. (20) of the manuscript"""

    # current coords are simple reference coords + displacement
    cur_coords = ref_geom_data.ref_coords + displacement

    # compute deformation gradient for each element in mesh
    F, J = compute_def_gradient(ref_geom_data.elements, ref_geom_data.ref_coords, cur_coords, ref_geom_data.Jtransform)

    # compute internal work done given specified constitutive law
    Psi_internal = compute_internal_work(F, J, ref_geom_data.element_vols, ref_geom_data._fibre_field, theta, ref_geom_data.constitutive_law).sum()

    # initialise external work to be zero
    Psi_external = 0.

    # compute external work due to body forces
    if external_forces.body_force is not None:

        Psi_external += compute_body_work(ref_geom_data.elements,
                                          ref_geom_data.element_vols,
                                          displacement,
                                          external_forces.body_force).sum()

    # compute external work due to surface forces
    if external_forces.surface_force is not None:

        F_surface = F[ref_geom_data.tri_surface_element_indices]
        J_surface = J[ref_geom_data.tri_surface_element_indices]
        disp_surface = displacement[ref_geom_data.tri_surface_node_indices]

        surface_force = external_forces.surface_force

        Psi_external += compute_surface_work(F_surface,
                                             J_surface,
                                             disp_surface,
                                             ref_geom_data.tri_surface_area_normals,
                                             surface_force).sum()

    # return total potential energy
    return Psi_internal - Psi_external

