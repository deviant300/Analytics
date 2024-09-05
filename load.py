"""To calculate all loads"""

import cupy as cp  # CuPy for GPU acceleration
import scipy.sparse as cpx_sparse
import pyvista as pv
import numpy as np  # NumPy for handling some CPU-based operations
from tqdm import tqdm


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
    return mesh

# ! Compute Jacobian -> find a native library
def compute_jacobian(element_nodes, dn_dxi):
    """Computes teh jacobian matrix

    Args:
        element_nodes (array): 3x3 matrice
        dN_dxi (array): derivatives of shape functions with respect to natural coordinates

    Returns:
        array: Jacobian
    """
    j = cp.zeros((3, 3))  # 3x3 for a 3D element
    for i in range(4):  # Assuming a 4-node tetrahedral element
        j += cp.outer(dn_dxi[i], element_nodes[i])
    return j


def compute_b_matrix(j_inv, dn_dxi):
    """Strain-Displacement matrix

    Args:
        J_inv (matrix): Inverse jacobian inverse matrix
        dN_dxi (matrx): derivatives of shape functions with respect to natural coordinates

    Returns:
        matrix: Strain-Displacement matrix
    """
    b = cp.zeros((6, 12))  # 6 strains and 3 displacements per node, 4 nodes * 3 = 12
    for i in range(4):
        dn_dx = cp.dot(j_inv, dn_dxi[i])
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
    c = cp.zeros((6, 6))  # 6x6 matrix for 3D stress-strain relationship
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
    dn_dxi = cp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    element_nodes = cp.array(mesh.points[nodes])  # Ensure this is CuPy
    j = compute_jacobian(element_nodes, dn_dxi)
    det_j = cp.linalg.det(j)

    if cp.abs(det_j) < 1e-12:
        print("Warning: Degenerate element detected. Skipping element.")
        return None

    j_inv = cp.linalg.inv(j)
    b = compute_b_matrix(j_inv, dn_dxi)
    c = compute_c_matrix(e, nu)
    k_elem = cp.dot(b.T, cp.dot(c, b)) * det_j
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
    global_indices = cp.array(
        [int(node * num_dofs_per_node) for node in element], dtype=cp.int32
    )

    # Pre-allocate arrays for data, rows, and cols
    total_entries = (num_nodes * num_dofs_per_node) ** 2
    data = cp.zeros(total_entries, dtype=k_elem.dtype)
    rows = cp.zeros(total_entries, dtype=cp.int32)
    cols = cp.zeros(total_entries, dtype=cp.int32)

    # Use vectorized operations to compute indices
    node_indices = cp.arange(num_nodes)
    dof_indices = cp.arange(num_dofs_per_node)

    # Generate all combinations of node and DOF indices using meshgrid
    node_i, node_j = cp.meshgrid(node_indices, node_indices)
    dof_k, dof_l = cp.meshgrid(dof_indices, dof_indices)

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
        nodes = np.array(cell.point_ids).astype(int)  # Use NumPy array for indexing
        k_elem = compute_element_stiffness(mesh, e, nu, nodes)

        if k_elem is not None:
            k_global = assemble_global_stiffness_efficient(k_global, k_elem, nodes)

    print("Global stiffness matrix computed.")
    return k_global.tocsr()  # Convert to CSR format after assembly


# Apply Loads
def apply_gravity_load(mesh, density):
    """applies gravity loads

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        density (float): Density in kg/m³

    Returns:
        matrix: returns weight per unit volume matrix
    """
    print("Applying gravity load...")
    f_global = cp.zeros(mesh.n_points * 3)
    f_global[2::3] -= density * 9.81  # Apply gravity to the z-axis (index 2)
    print("Gravity load applied.")
    return f_global


def apply_pore_pressure(mesh, water_table_depth, pore_pressure):
    """Applies pore pressure to the global force vector in a
    finite element mesh based on a given water table depth.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        water_table_depth (float): The depth of the water table relative to the
        global coordinate system.
        pore_pressure (float): The magnitude of the pore pressure to apply at nodes
        below the water table depth.

    Returns:
        array:  A 1D CuPy array representing the global force vector
        with pore pressure applied.
    """
    print("Applying pore pressure...")
    points = cp.array(mesh.points)  # Ensure points are a 2D CuPy array
    f_global = cp.zeros(mesh.n_points * 3)
    f_global[cp.where(points[:, 2] < water_table_depth)[0] * 3 + 2] += pore_pressure
    print("Pore pressure applied.")
    return f_global


def apply_surcharge_load(mesh, surcharge_load):
    """Applies a surcharge load to the global force vector in a finite element mesh.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        surcharge_load (float):  The magnitude of the surcharge load to apply at nodes
        on the top surface of the mesh.

    Returns:
        array: A 1D CuPy array representing the global force vector with the surcharge load applied.
    """
    print("Applying surcharge load...")
    points = cp.array(mesh.points)  # Ensure points are a 2D CuPy array
    f_global = cp.zeros(mesh.n_points * 3)
    max_z = cp.max(points[:, 2])  # Ensure max_z is computed with CuPy
    f_global[cp.where(points[:, 2] == max_z)[0] * 3 + 2] += surcharge_load
    print("Surcharge load applied.")
    return f_global


def apply_seismic_load(mesh, seismic_coefficient):
    """Applies a seismic load to the global force vector in a finite element mesh.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        seismic_coefficient (float): Seismic load factor

    Returns:
        array: A 1D CuPy array representing the global force vector with the seismic load applied.
    """
    print("Applying seismic load...")
    points = cp.array(mesh.points)  # Ensure points are a 2D CuPy array
    f_global = cp.zeros(mesh.n_points * 3)
    f_global[2::3] += seismic_coefficient * points[:, 2]
    print("Seismic load applied.")
    return f_global


def apply_loads(
    mesh, density, water_table_depth, pore_pressure, surcharge_load, seismic_coefficient
):
    """Applies various loads to the global force vector in a finite element mesh.

    Args:
        mesh (pyvista.DataSet): mesh (mesh object): The finite element mesh object that contains the 
        geometry and connectivity information of the entire domain.
        density (float): Density in kg/m³
        water_table_depth (flaot): The depth of the water table relative to the global coordinate 
        system.
        pore_pressure (float): The magnitude of the pore pressure to apply at nodes below the 
        water table depth.
        surcharge_load (float): The magnitude of the surcharge load to apply at nodes on the top
        surface of the mesh.
        seismic_coefficient (flaot): Seismic load factor

    Returns:
        array: A 1D CuPy array representing the global force vector with all loads
        (gravity, pore pressure, surcharge and seismic) applied.
    """
    print("Applying all loads...")
    f_global = cp.zeros(mesh.n_points * 3)
    f_global += apply_gravity_load(mesh, density)
    f_global += apply_pore_pressure(mesh, water_table_depth, pore_pressure)
    f_global += apply_surcharge_load(mesh, surcharge_load)
    f_global += apply_seismic_load(mesh, seismic_coefficient)
    print("All loads applied.")
    return f_global
# End-of-file (EOF)
