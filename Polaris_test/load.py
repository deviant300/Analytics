"""To calculate all loads using polars"""

import cupy as cp  # CuPy for GPU acceleration
import scipy.sparse as cpx_sparse
import pyvista as pv
import numpy as np  # NumPy for handling some CPU-based operations
from tqdm import tqdm
import dask.array as da

def load_volume_mesh(filepath):
    """Load the Volume Mesh

    Args:
        filepath (string): filepath to mesh file

    Returns:
        Dataset: mesh as array
    """
    print("Loading volume mesh...")
    print(f"Reading file: {filepath}")
    mesh = pv.read(filepath)
    print("Volume mesh loaded successfully.")
    
    # Extract the image data as a NumPy array
    mesh_np = mesh.point_data[0]
    
    mesh_dask = da.from_array(mesh_np, chunks=(1000, 1000, 1000)) # Adjust chunk size as needed
    return mesh_dask

# ! Compute Jacobian -> find a native library
def compute_jacobian(element_nodes, dn_dxi):
    """Computes teh jacobian matrix

    Args:
        element_nodes (array): 3x3 matrice
        dN_dxi (array): derivatives of shape functions with respect to natural coordinates

    Returns:
        array: Jacobian
    """
    jacobian = da.zeros((3, 3), chunks=(3,3))  # 3x3 for a 3D element
    
    for i in range(4):
        jacobian += da.outer(dn_dxi[i], element_nodes[i])
    
    return jacobian

def compute_b_matrix(j_inv, dn_dxi):
    """Strain-Displacement matrix

    Args:
        J_inv (matrix): Inverse jacobian inverse matrix
        dN_dxi (matrx): derivatives of shape functions with respect to natural coordinates

    Returns:
        matrix: Strain-Displacement matrix
    """
    b = da.zeros((6, 12), chunks=(6, 12))  # 6 strains and 3 displacements per node, 4 nodes * 3 = 12
    for i in range(4):
        dn_dx = da.dot(j_inv, dn_dxi[i])
        b[0, i * 3] = dn_dx[0]  # ε_xx
        b[1, i * 3 + 1] = dn_dx[1]  # ε_yy
        b[2, i * 3 + 2] = dn_dx[2]  # ε_zz
        b[3, i * 3] = dn_dx[1]  # ε_xy
        b[3, i * 3 + 1] = dn_dx[0]
        b[4, i * 3 + 1] = dn_dx[2]  # ε_yz
        b[4, i * 3 + 2] = dn_dx[1]
        b[5, i * 3] = dn_dx[2]  # ε_zx
        b[5, i * 3 + 2] = dn_dx[0]
    return b

def compute_c_matrix(e, nu):
    """Elasticity matrix

    Args:
        E (flaot): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        matrix:  elasticity matrix
    """
    c = da.zeros((6, 6), chunks=(6, 6))  # 6x6 matrix for 3D stress-strain relationship
    factor = e / (1 + nu) / (1 - 2 * nu)
    c[0, 0] = c[1, 1] = c[2, 2] = factor * (1 - nu)
    c[3, 3] = c[4, 4] = c[5, 5] = e / 2 / (1 + nu)
    c[0, 1] = c[1, 0] = c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = factor * nu
    return c

def compute_element_stiffness(mesh, e, nu, nodes):
    """Compute Element Stiffness

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        E (float): Young's modulus
        nu (float): Poisson's ratio
        nodes (float): points on the mesh

    Returns:
        matrix: array of element stiffness
    """
    dn_dxi = da.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], chunks=(4,3))

    element_nodes = da.array(mesh.points[nodes])  # Ensure this is CuPy
    j = compute_jacobian(element_nodes, dn_dxi)
    det_j = da.linalg.det(j)

    if da.abs(det_j) < 1e-12:
        print("Warning: Degenerate element detected. Skipping element.")
        return None

    j_inv = da.linalg.inv(j)
    b = compute_b_matrix(j_inv, dn_dxi)
    c = compute_c_matrix(e, nu)
    k_elem = da.dot(b.T, da.dot(c, b)) * det_j
    return k_elem

def assemble_global_stiffness_efficient(k_global, k_elem, element):
    """Efficiently assembles the global stiffness matrix for a finite element model
    by adding the contributions from an element's stiffness matrix using sparse matrix operations.

    Args:
        K_global (cpx_sparse.csr_matrix): The global stiffness matrix
        in CSR (Compressed Sparse Row) format.
        K_elem (cpx_sparse.csr_matrix): The element stiffness matrix as a CuPy array
        of shape (num_nodes * num_dofs_per_node, num_nodes * num_dofs_per_node). element
        (cpx_sparse.csr_matrix):  An array or list of integers representing the global node indices
        that define the finite element. Each entry corresponds to a node in the global mesh.

    Returns:
        matrix: The updated global stiffness matrix after adding the contributions
        from the provided element's stiffness matrix. The input matrix is modified in
        place and returned for convenience.
    """
    num_dofs_per_node = 3
    num_nodes = len(element)

    # Compute global indices
    global_indices = da.array(
        [int(node * num_dofs_per_node) for node in element], dtype=int
    )

    # Pre-allocate arrays for data, rows, and cols
    total_entries = (num_nodes * num_dofs_per_node) ** 2
    data = da.zeros(total_entries, dtype=k_elem.dtype)
    rows = da.zeros(total_entries, dtype=da.int32)
    cols = da.zeros(total_entries, dtype=da.int32)

    # Use vectorized operations to compute indices
    node_indices = da.arange(num_nodes)
    dof_indices = da.arange(num_dofs_per_node)

    # Generate all combinations of node and DOF indices using meshgrid
    node_i, node_j = da.meshgrid(node_indices, node_indices)
    dof_k, dof_l = da.meshgrid(dof_indices, dof_indices)

    # Fill data array by flattening K_elem
    data[:] = k_elem.ravel()

    # Compute global indices for rows and cols
    rows[:] = global_indices[node_i.ravel()] + dof_k.ravel()
    cols[:] = global_indices[node_j.ravel()] + dof_l.ravel()

    # Create a COO matrix and add it to the global stiffness matrix
    k_global += cpx_sparse.coo_matrix(
        (data, (rows, cols)), shape=k_global.shape
    ).tocsr()

    return k_global

def compute_global_stiffness_matrix(mesh, e, nu):
    """Computes the global stiffness matrix for a finite element mesh
    using the material properties and element stiffness matrices.
    The global stiffness matrix is assembled by iterating over all elements
    in the mesh and summing their contributions.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        matrix: The global stiffness matrix in CSR (Compressed Sparse Row) format.
    """
    print("Computing global stiffness matrix...")
    k_global = cpx_sparse.coo_matrix(
        (mesh.n_points * 3, mesh.n_points * 3)
    )  # Start with COO format

    for i in tqdm(range(mesh.n_cells)):
        cell = mesh.get_cell(i)
        nodes = da.array(cell.point_ids).astype(int)  # Use NumPy array for indexing
        k_elem = compute_element_stiffness(mesh, e, nu, nodes)

        if k_elem is not None:
            k_global = assemble_global_stiffness_efficient(k_global, k_elem, nodes)

    print("Global stiffness matrix computed.")
    return k_global.tocsr()  # Convert to CSR format after assembly