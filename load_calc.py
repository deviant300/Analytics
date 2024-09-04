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


def assign_material_properties(mesh, cohesion, friction_angle, density, E, nu):
    """Assign Material Properties

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        cohesion (int): Cohesion in Pascals
        friction_angle (int): Friction angle in degrees
        density (int): Density in kg/m³
        E (float): Young's modulus in Pascals
        nu (float): Poisson's ratio
    """
    print("Assigning material properties...")
    mesh["cohesion"] = np.full(mesh.n_cells, cohesion)  # Keep it as NumPy
    mesh["friction_angle"] = np.full(mesh.n_cells, friction_angle)  # Keep it as NumPy
    mesh["density"] = np.full(mesh.n_cells, density)  # Keep it as NumPy
    mesh["E"] = np.full(mesh.n_cells, E)  # Keep it as NumPy
    mesh["nu"] = np.full(mesh.n_cells, nu)  # Keep it as NumPy
    print("Material properties assigned.")


# ! Compute Jacobian -> find a native library
def compute_jacobian(element_nodes, dN_dxi):
    """Computes teh jacobian matrix

    Args:
        element_nodes (array): 3x3 matrice
        dN_dxi (array): derivatives of shape functions with respect to natural coordinates

    Returns:
        array: Jacobian
    """
    J = cp.zeros((3, 3))  # 3x3 for a 3D element
    for i in range(4):  # Assuming a 4-node tetrahedral element
        J += cp.outer(dN_dxi[i], element_nodes[i])
    return J


def compute_B_matrix(J_inv, dN_dxi):
    """Strain-Displacement matrix

    Args:
        J_inv (matrix): Inverse jacobian inverse matrix
        dN_dxi (matrx): derivatives of shape functions with respect to natural coordinates

    Returns:
        matrix: Strain-Displacement matrix
    """
    B = cp.zeros((6, 12))  # 6 strains and 3 displacements per node, 4 nodes * 3 = 12
    for i in range(4):
        dN_dx = cp.dot(J_inv, dN_dxi[i])
        B[0, i * 3] = dN_dx[0]  # ε_xx
        B[1, i * 3 + 1] = dN_dx[1]  # ε_yy
        B[2, i * 3 + 2] = dN_dx[2]  # ε_zz
        B[3, i * 3] = dN_dx[1]  # ε_xy
        B[3, i * 3 + 1] = dN_dx[0]
        B[4, i * 3 + 1] = dN_dx[2]  # ε_yz
        B[4, i * 3 + 2] = dN_dx[1]
        B[5, i * 3] = dN_dx[2]  # ε_zx
        B[5, i * 3 + 2] = dN_dx[0]
    return B


def compute_C_matrix(E, nu):
    """Elasticity matrix

    Args:
        E (flaot): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        matrix:  elasticity matrix
    """
    C = cp.zeros((6, 6))  # 6x6 matrix for 3D stress-strain relationship
    factor = E / (1 + nu) / (1 - 2 * nu)
    C[0, 0] = C[1, 1] = C[2, 2] = factor * (1 - nu)
    C[3, 3] = C[4, 4] = C[5, 5] = E / 2 / (1 + nu)
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = factor * nu
    return C


def compute_element_stiffness(mesh, E, nu, nodes):
    """Compute Element Stiffness

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        E (float): Young's modulus
        nu (float): Poisson's ratio
        nodes (float): points on the mesh

    Returns:
        matrix: array of element stiffness
    """
    dN_dxi = cp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    element_nodes = cp.array(mesh.points[nodes])  # Ensure this is CuPy
    J = compute_jacobian(element_nodes, dN_dxi)
    det_J = cp.linalg.det(J)

    if cp.abs(det_J) < 1e-12:
        print("Warning: Degenerate element detected. Skipping element.")
        return None

    J_inv = cp.linalg.inv(J)
    B = compute_B_matrix(J_inv, dN_dxi)
    C = compute_C_matrix(E, nu)
    K_elem = cp.dot(B.T, cp.dot(C, B)) * det_J
    return K_elem


def assemble_global_stiffness_efficient(K_global, K_elem, element):
    """Efficiently assembles the global stiffness matrix for a finite element model by adding the contributions from an element's stiffness matrix using sparse matrix operations.

    Args:
        K_global (cpx_sparse.csr_matrix): The global stiffness matrix in CSR (Compressed Sparse Row) format.
        K_elem (cpx_sparse.csr_matrix): The element stiffness matrix as a CuPy array of shape (num_nodes * num_dofs_per_node, num_nodes * num_dofs_per_node).
        element (cpx_sparse.csr_matrix):  An array or list of integers representing the global node indices that define the finite element. Each entry corresponds to a node in the global mesh.

    Returns:
        matrix: The updated global stiffness matrix after adding the contributions from the provided element's stiffness matrix. The input matrix is modified in place and returned for convenience.
    """
    num_dofs_per_node = 3
    num_nodes = len(element)

    # Compute global indices
    global_indices = cp.array(
        [int(node * num_dofs_per_node) for node in element], dtype=cp.int32
    )

    # Pre-allocate arrays for data, rows, and cols
    total_entries = (num_nodes * num_dofs_per_node) ** 2
    data = cp.zeros(total_entries, dtype=K_elem.dtype)
    rows = cp.zeros(total_entries, dtype=cp.int32)
    cols = cp.zeros(total_entries, dtype=cp.int32)

    # Use vectorized operations to compute indices
    node_indices = cp.arange(num_nodes)
    dof_indices = cp.arange(num_dofs_per_node)

    # Generate all combinations of node and DOF indices using meshgrid
    node_i, node_j = cp.meshgrid(node_indices, node_indices)
    dof_k, dof_l = cp.meshgrid(dof_indices, dof_indices)

    # Fill data array by flattening K_elem
    data[:] = K_elem.ravel()

    # Compute global indices for rows and cols
    rows[:] = global_indices[node_i.ravel()] + dof_k.ravel()
    cols[:] = global_indices[node_j.ravel()] + dof_l.ravel()

    # Create a COO matrix and add it to the global stiffness matrix
    K_global += cpx_sparse.coo_matrix(
        (data, (rows, cols)), shape=K_global.shape
    ).tocsr()

    return K_global


def compute_global_stiffness_matrix(mesh, E, nu):
    """Computes the global stiffness matrix for a finite element mesh using the material properties and element stiffness matrices. The global stiffness matrix is assembled by iterating over all elements in the mesh and summing their contributions.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        matrix: The global stiffness matrix in CSR (Compressed Sparse Row) format.
    """
    print("Computing global stiffness matrix...")
    K_global = cpx_sparse.coo_matrix(
        (mesh.n_points * 3, mesh.n_points * 3)
    )  # Start with COO format

    for i in tqdm(range(mesh.n_cells)):
        cell = mesh.get_cell(i)
        nodes = np.array(cell.point_ids).astype(int)  # Use NumPy array for indexing
        K_elem = compute_element_stiffness(mesh, E, nu, nodes)

        if K_elem is not None:
            K_global = assemble_global_stiffness_efficient(K_global, K_elem, nodes)

    print("Global stiffness matrix computed.")
    return K_global.tocsr()  # Convert to CSR format after assembly


# Apply Loads
def apply_gravity_load(mesh, density):
    """applies gravity loads

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        density (float): Density in kg/m³

    Returns:
        matrix: returns weight per unit volume matrix
    """
    print("Applying gravity load...")
    F_global = cp.zeros(mesh.n_points * 3)
    F_global[2::3] -= density * 9.81  # Apply gravity to the z-axis (index 2)
    print("Gravity load applied.")
    return F_global


def apply_pore_pressure(mesh, water_table_depth, pore_pressure):
    """Applies pore pressure to the global force vector in a finite element mesh based on a given water table depth.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        water_table_depth (float): The depth of the water table relative to the global coordinate system.
        pore_pressure (float): The magnitude of the pore pressure to apply at nodes below the water table depth.

    Returns:
        array:  A 1D CuPy array representing the global force vector with pore pressure applied.
    """
    print("Applying pore pressure...")
    points = cp.array(mesh.points)  # Ensure points are a 2D CuPy array
    F_global = cp.zeros(mesh.n_points * 3)
    F_global[cp.where(points[:, 2] < water_table_depth)[0] * 3 + 2] += pore_pressure
    print("Pore pressure applied.")
    return F_global


def apply_surcharge_load(mesh, surcharge_load):
    """Applies a surcharge load to the global force vector in a finite element mesh.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        surcharge_load (float):  The magnitude of the surcharge load to apply at nodes on the top surface of the mesh.

    Returns:
        array: A 1D CuPy array representing the global force vector with the surcharge load applied.
    """
    print("Applying surcharge load...")
    points = cp.array(mesh.points)  # Ensure points are a 2D CuPy array
    F_global = cp.zeros(mesh.n_points * 3)
    max_z = cp.max(points[:, 2])  # Ensure max_z is computed with CuPy
    F_global[cp.where(points[:, 2] == max_z)[0] * 3 + 2] += surcharge_load
    print("Surcharge load applied.")
    return F_global


def apply_seismic_load(mesh, seismic_coefficient):
    """Applies a seismic load to the global force vector in a finite element mesh.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        seismic_coefficient (float): Seismic load factor

    Returns:
        array: A 1D CuPy array representing the global force vector with the seismic load applied.
    """
    print("Applying seismic load...")
    points = cp.array(mesh.points)  # Ensure points are a 2D CuPy array
    F_global = cp.zeros(mesh.n_points * 3)
    F_global[2::3] += seismic_coefficient * points[:, 2]
    print("Seismic load applied.")
    return F_global


def apply_loads(
    mesh, density, water_table_depth, pore_pressure, surcharge_load, seismic_coefficient
):
    """Applies various loads to the global force vector in a finite element mesh.

    Args:
        mesh (pyvista.DataSet): mesh (mesh object): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        density (float): Density in kg/m³
        water_table_depth (flaot): The depth of the water table relative to the global coordinate system.
        pore_pressure (float): The magnitude of the pore pressure to apply at nodes below the water table depth.
        surcharge_load (float): The magnitude of the surcharge load to apply at nodes on the top surface of the mesh.
        seismic_coefficient (flaot): Seismic load factor

    Returns:
        array: A 1D CuPy array representing the global force vector with all loads (gravity, pore pressure, surcharge and seismic) applied.
    """
    print("Applying all loads...")
    F_global = cp.zeros(mesh.n_points * 3)
    F_global += apply_gravity_load(mesh, density)
    F_global += apply_pore_pressure(mesh, water_table_depth, pore_pressure)
    F_global += apply_surcharge_load(mesh, surcharge_load)
    F_global += apply_seismic_load(mesh, seismic_coefficient)
    print("All loads applied.")
    return F_global