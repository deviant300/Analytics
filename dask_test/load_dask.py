"""To calculate all loads using dask"""
import cupy as cp
import scipy.sparse as cpx_sparse
import pyvista as pv
import numpy as np

import dask
import dask.array as da

# ! <<<<<<<<<<<<<<<<<<<< math funcs >>>>>>>>>>>>>>>>>>>>
def det(x):
    """Wrapping the numpy function to calculate determinant
        into a dask function

    Args:
        x (array): The array whose determinant is needed

    Returns:
        float: determinant value
    """
    return np.linalg.det(x)
da_det = da.gufunc(det, signature="(i)->()", output_dtypes=float, vectorize=True)
# ? <<<<<<<<<<<<<<<<<<<< math funcs >>>>>>>>>>>>>>>>>>>>

def inv(x):
    """Wrapping the numpy function to calculate determinant
        into a dask function

    Args:
        x (array): The array whose determinant is needed

    Returns:
        float: determinant value
    """
    return np.linalg.inv(x)
da_inv = da.gufunc(inv, signature="(i)->()", output_dtypes=float, vectorize=True)

# ! <<<<<<<<<<<<<<<<<<<< math funcs >>>>>>>>>>>>>>>>>>>>

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

def compute_jacobian_dask(element_nodes, dn_dxi):
    """Computes the Jacobian matrix for each element in parallel using Dask.

    Args:
        element_nodes (dask.array.Array or cupy.ndarray): A 2D array 
        where each row represents the coordinates of an element's node 
        (e.g., 4x3 for a 4-node tetrahedral element).
        dn_dxi (dask.array.Array or cupy.ndarray): Derivatives of shape 
        functions with respect to natural coordinates.
        This should be a 2D array (4x3 for 4 nodes and 3 natural coordinates).

    Returns:
        dask.array.Array: Dask array containing Jacobian matrices for each element.
    """

    # Ensure inputs are Dask arrays
    if not isinstance(element_nodes, da.Array):
        element_nodes = da.from_array(element_nodes, chunks="auto")
    if not isinstance(dn_dxi, da.Array):
        dn_dxi = da.from_array(dn_dxi, chunks="auto")

    # Function to compute the Jacobian for a single element
    def compute_single_jacobian(elem_nodes, dn_dxi_single):
        j = cp.zeros((3, 3))  # 3x3 for a 3D element
        for i in range(4):  # Assuming a 4-node tetrahedral element
            j += cp.outer(dn_dxi_single[i], elem_nodes[i])
        return j

    # Apply the Jacobian computation for each element using map_blocks directly without a lambda
    jacobians = da.map_blocks(
        compute_single_jacobian, element_nodes, dn_dxi, dtype=cp.float32
    )

    return jacobians

def compute_b_matrix_dask(j_inv, dn_dxi):
    """Strain-Displacement matrix computed in parallel using Dask.

    Args:
        j_inv (dask.array.Array or cupy.ndarray): 
        Inverse of the Jacobian matrix for each element.
        dn_dxi (dask.array.Array or cupy.ndarray): Derivatives 
        of shape functions with respect to natural coordinates.

    Returns:
        dask.array.Array: Dask array containing Strain-Displacement matrices for each element.
    """

    # Ensure inputs are Dask arrays
    if not isinstance(j_inv, da.Array):
        j_inv = da.from_array(j_inv, chunks="auto")
    if not isinstance(dn_dxi, da.Array):
        dn_dxi = da.from_array(dn_dxi, chunks="auto")

    # Function to compute the B-matrix for a single element
    def compute_single_b_matrix(j_inv_single, dn_dxi_single):
        b = cp.zeros((6, 12))  # 6 strains and 3 displacements per node,
                               # 4 nodes * 3 = 12
        for i in range(4):  # Assuming a 4-node tetrahedral element
            dn_dx = cp.dot(j_inv_single, dn_dxi_single[i])
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

    # Apply the B-matrix computation for each element using map_blocks
    b_matrices = da.map_blocks(
        compute_single_b_matrix, j_inv, dn_dxi, dtype=cp.float32
    )

    return b_matrices

def compute_c_matrix_dask(e, nu):
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

def compute_element_stiffness_dask(mesh, e, nu, nodes):
    """Compute Element Stiffness for multiple elements in parallel using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        e (float): Young's modulus (scalar value)
        nu (float): Poisson's ratio (scalar value)
        nodes (array-like): The node indices for multiple elements

    Returns:
        dask.array.Array: Dask array of element stiffness matrices
    """
    # Use the same dn_dxi for each element
    dn_dxi = cp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Function to compute the stiffness matrix for a single element
    @dask.delayed
    def compute_single_element_stiffness(nodes_for_element):
        element_nodes = cp.array(mesh.points[nodes_for_element])  # Convert to CuPy array
        j = compute_jacobian_dask(element_nodes, dn_dxi)
        det_j = cp.linalg.det(j)

        if cp.abs(det_j) < 1e-12:
            print("Warning: Degenerate element detected. Skipping element.")
            return None

        j_inv = cp.linalg.inv(j)
        b = compute_b_matrix_dask(j_inv, dn_dxi)
        c = compute_c_matrix_dask(e, nu)
        k_elem = cp.dot(b.T, cp.dot(c, b)) * det_j
        return k_elem

    # List of delayed computations for each element's stiffness matrix
    stiffness_matrices = [
        compute_single_element_stiffness(nodes[i]) for i in range(len(nodes))
    ]

    # Convert the list of delayed results into a Dask array
    stiffness_matrices_dask = da.from_delayed(
                                                stiffness_matrices,
                                                shape=(
                                                    len(nodes),
                                                    12,
                                                    12
                                                    ), dtype=cp.float32)

    return stiffness_matrices_dask

@dask.delayed
def assemble_element(k_global, k_elem, element):
    """Assembles the global stiffness matrix for a single element."""
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
    element_matrix = cpx_sparse.coo_matrix(
        (data, (rows, cols)), shape=k_global.shape
    ).tocsr()

    # Return the modified k_global matrix
    return k_global + element_matrix

def assemble_global_stiffness_dask(k_global, k_elem_list, element_list):
    """Efficiently assembles the global stiffness matrix for a finite element model
    by adding contributions from all element stiffness matrices in parallel using Dask.

    Args:
        k_global (cupyx.scipy.sparse.csr_matrix): The global stiffness matrix
        in CSR (Compressed Sparse Row) format.
        k_elem_list (list): A list of element stiffness matrices.
        element_list (list): A list of element connectivity (global node indices for each element).

    Returns:
        dask.delayed.Delayed: A delayed object representing the assembled global stiffness matrix.
    """
    # List of delayed assembly tasks for each element
    tasks = [assemble_element
                            (
                            k_global,
                            k_elem_list[i],
                            element_list[i]
                            ) for i in range(len(k_elem_list))]

    # Use Dask's delayed to sum all tasks to assemble the global stiffness matrix
    assembled_matrix = dask.delayed(sum)(tasks)

    return assembled_matrix

@dask.delayed
def compute_global_stiffness_matrix_dask(mesh, e, nu):
    """Computes the global stiffness matrix for a finite element mesh
    using Dask for parallelization.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        e (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        dask.delayed.Delayed: The global stiffness matrix in CSR format, computed in parallel.
    """
    print("Computing global stiffness matrix in parallel...")

    # Initialize global stiffness matrix in COO format (for efficient assembly)
    k_global = cpx_sparse.coo_matrix((mesh.n_points * 3, mesh.n_points * 3))

    # List to store the assembly tasks for each element
    tasks = []

    # Iterate over the elements of the mesh in parallel
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        nodes = np.array(cell.point_ids).astype(int)  # Get global node indices

        # Compute the element stiffness matrix in parallel
        k_elem = dask.delayed(compute_element_stiffness_dask)(mesh, e, nu, nodes)

        # Assemble the global stiffness matrix for this element in parallel
        task = dask.delayed(assemble_global_stiffness_dask)(k_global, k_elem, nodes)
        tasks.append(task)

    # Sum all the tasks (element contributions) to get the final global stiffness matrix
    k_global_final = dask.delayed(sum)(tasks)

    print("Global stiffness matrix computed in parallel.")
    return k_global_final.tocsr()  # Convert to CSR format after assembly

def apply_gravity_load_dask(mesh, density):
    """Applies gravity loads in parallel using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that 
        contains the geometry and connectivity information of the 
        entire domain.
        density (float): Density in kg/m³

    Returns:
        dask.array.Array: A Dask array representing the gravity load 
        applied to each node in the mesh.
    """
    print("Applying gravity load in parallel...")

    # Initialize the global force vector using Dask, with 3 degrees of freedom per node
    f_global = da.zeros(mesh.n_points * 3, chunks='auto')

    # Apply gravity load to the z-axis in parallel
    gravity_load_z = density * 9.81

    # Create a Dask array to apply the gravity load only to the z-axis (index 2::3)
    f_global = f_global.map_blocks(
        lambda x: x - gravity_load_z,
        dtype=f_global.dtype
    )

    print("Gravity load applied in parallel.")
    return f_global

def apply_pore_pressure_dask(mesh, water_table_depth, pore_pressure):
    """Applies pore pressure to the global force vector in a
    finite element mesh based on a given water table depth using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        water_table_depth (float): The depth of the water table relative to the
        global coordinate system.
        pore_pressure (float): The magnitude of the pore pressure to apply at nodes
        below the water table depth.

    Returns:
        dask.array.Array: A Dask array representing the global force vector
        with pore pressure applied in parallel.
    """
    print("Applying pore pressure in parallel...")

    # Convert points to a Dask array with automatic chunking
    points = da.from_array(mesh.points, chunks='auto')

    # Initialize the global force vector as a Dask array of zeros
    f_global = da.zeros(mesh.n_points * 3, chunks='auto')

    # Find the nodes below the water table
    z_coords = points[:, 2]

    # Apply pore pressure to the z-axis for nodes below the water table
    below_water_table = z_coords < water_table_depth
    f_global = f_global.map_blocks(
        lambda f: f + pore_pressure,
        where=below_water_table[:, None],
        dtype=f_global.dtype
    )

    print("Pore pressure applied in parallel.")
    return f_global

def apply_surcharge_load_dask(mesh, surcharge_load):
    """Applies a surcharge load to the global force vector in a finite element mesh using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        surcharge_load (float): The magnitude of the surcharge load to apply at nodes
        on the top surface of the mesh.

    Returns:
        dask.array.Array: A Dask array representing the global force vector with 
        the surcharge load applied.
    """
    print("Applying surcharge load in parallel...")

    # Convert points to Dask array
    points = da.from_array(mesh.points, chunks='auto')

    # Initialize the global force vector as a Dask array of zeros
    f_global = da.zeros(mesh.n_points * 3, chunks='auto')

    # Compute the maximum z-coordinate using Dask
    max_z = da.max(points[:, 2])

    # Find the nodes on the top surface (with z = max_z)
    top_surface_nodes = points[:, 2] == max_z

    # Apply the surcharge load to the z-axis for the top surface nodes
    f_global = f_global.map_blocks(
        lambda f, top_nodes: f + surcharge_load * top_nodes[:, None],
        top_surface_nodes,
        dtype=f_global.dtype
    )

    print("Surcharge load applied in parallel.")
    return f_global

def apply_seismic_load_dask(mesh, seismic_coefficient):
    """Applies a seismic load to the global force vector in a finite element mesh using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        seismic_coefficient (float): Seismic load factor

    Returns:
        dask.array.Array: A Dask array representing the global force vector with the seismic 
        load applied.
    """
    print("Applying seismic load in parallel...")

    # Convert mesh points to a Dask array
    points = da.from_array(mesh.points, chunks='auto')

    # Initialize the global force vector as a Dask array of zeros
    f_global = da.zeros(mesh.n_points * 3, chunks='auto')

    # Apply seismic load in parallel (seismic load on z-axis, index 2::3)
    f_global = f_global.map_blocks(
        lambda fg, pts: fg + seismic_coefficient * pts[:, 2],
        points,
        dtype=f_global.dtype
    )

    print("Seismic load applied in parallel.")
    return f_global

def apply_loads_dask(
    mesh, density, water_table_depth, pore_pressure, surcharge_load, seismic_coefficient
):
    """Applies various loads to the global force vector in a finite element mesh using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the 
        geometry and connectivity information of the entire domain.
        density (float): Density in kg/m³
        water_table_depth (float): The depth of the water table relative to the global coordinate 
        system.
        pore_pressure (float): The magnitude of the pore pressure to apply at nodes below the 
        water table depth.
        surcharge_load (float): The magnitude of the surcharge load to apply at nodes on the top
        surface of the mesh.
        seismic_coefficient (float): Seismic load factor

    Returns:
        dask.array.Array: A Dask array representing the global force vector with all loads
        (gravity, pore pressure, surcharge, and seismic) applied in parallel.
    """
    print("Applying all loads in parallel...")

    # Apply all loads in parallel using the Dask versions of the functions
    gravity_load = apply_gravity_load_dask(mesh, density)
    pore_pressure_load = apply_pore_pressure_dask(mesh, water_table_depth, pore_pressure)
    surcharge_load = apply_surcharge_load_dask(mesh, surcharge_load)
    seismic_load = apply_seismic_load_dask(mesh, seismic_coefficient)

    # Sum all loads in parallel
    f_global = gravity_load + pore_pressure_load + surcharge_load + seismic_load

    print("All loads applied in parallel.")
    return f_global
# End-of-file (EOF)
