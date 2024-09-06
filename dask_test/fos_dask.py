"""To calculate fos using dask"""
import dask
import dask.array as da
import numpy as np
import cupy as cp  # CuPy for GPU acceleration
from cupyx.scipy.sparse import csr_matrix

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import bicgstab  # Fallback to CPU-based solver

from load_dask import compute_jacobian_dask
from load_dask import compute_b_matrix_dask
from load_dask import compute_c_matrix_dask

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

def calculate_mesh_shape_dask(mesh):
    """
    Calculate the shape of the mesh grid directly from a structured or semi-structured 3D mesh.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.

    Returns:
        tuple: The estimated shape of the mesh (nx, ny, nz).
    """
    # Find unique coordinate values along each axis
    x_column = mesh.points[:, 0]
    # Convert the NumPy array column to a Dask array
    dask_x_column = da.from_array(x_column, chunks=100)
    # Apply np.unique on each chunk
    unique_per_chunk = dask_x_column.map_blocks(np.unique)
    # Combine results across all chunks and compute the final unique values
    x_coords = np.unique(unique_per_chunk.compute())

    y_column = mesh.points[:, 1]
    # Convert the NumPy array column to a Dask array
    dask_y_column = da.from_array(y_column, chunks=100)
    # Apply np.unique on each chunk
    unique_per_chunk = dask_y_column.map_blocks(np.unique)
    # Combine results across all chunks and compute the final unique values
    y_coords = np.unique(unique_per_chunk.compute())

    z_column = mesh.points[:, 2]
    # Convert the NumPy array column to a Dask array
    dask_z_column = da.from_array(z_column, chunks=100)
    # Apply np.unique on each chunk
    unique_per_chunk = dask_z_column.map_blocks(np.unique)
    # Combine results across all chunks and compute the final unique values
    z_coords = np.unique(unique_per_chunk.compute())

    # Calculate the number of unique points along each axis to determine grid dimensions
    nx = len(x_coords)
    ny = len(y_coords)
    nz = len(z_coords)

    print(f"Calculated mesh shape: (nx, ny, nz) = ({nx}, {ny}, {nz})")

    return (nx, ny, nz)

def identify_fixed_nodes_dask(mesh):
    """Identifies fixed nodes in a finite element mesh based on specified boundary conditions.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.

    Returns:
        array: An array of indices representing the fixed nodes in the mesh.
    """
    fixed_nodes = []

    # Find minimum and maximum coordinates
    min_z = mesh.points[:, 2].min()
    min_x = mesh.points[:, 0].min()
    max_x = mesh.points[:, 0].max()
    min_y = mesh.points[:, 1].min()
    max_y = mesh.points[:, 1].max()

    # Loop through each node and check if it's a fixed node
    for i, node in enumerate(mesh.points):
        # Fix base mesh nodes (minimum elevation)
        if node[2] == min_z:
            fixed_nodes.append(i)
        # Fix nodes on vertical sides of pit
        elif node[0] == min_x or node[0] == max_x:
            fixed_nodes.append(i)
        elif node[1] == min_y or node[1] == max_y:
            fixed_nodes.append(i)

    return fixed_nodes

def apply_boundary_conditions_dask(k_global, f_global, fixed_nodes):
    """Applies boundary conditions to the global stiffness matrix 
    and global force vector for a finite element model.

    Args:
        K_global (matrix): The global stiffness matrix in CSR (Compressed Sparse Row) 
        format or as a CuPy dense array.
        F_global (array): The global force vector as a CuPy array. I
        t represents the applied forces at each node of the mesh.
        fixed_nodes (array): An array or list of integers representing the 
        indices of nodes that are fixed in the mesh.

    Returns:
        tuple: K_global (cupyx.scipy.sparse.csr_matrix or cp.ndarray), F_global (cp.ndarray)
    """
    print("Applying boundary conditions...")

    num_dofs_per_node = 3
    fixed_dofs = da.array(
        [
            node * num_dofs_per_node + i
            for node in fixed_nodes
            for i in range(num_dofs_per_node)
        ]
    )

    # Create a mask for non-fixed DOFs
    non_fixed_dofs = cp.ones(k_global.shape[0], dtype=bool)
    non_fixed_dofs[fixed_dofs] = False

    # Zero out the rows and columns for fixed DOFs in K_global
    k_global[fixed_dofs, :] = 0
    k_global[:, fixed_dofs] = 0

    # Set diagonal for fixed DOFs
    k_global[fixed_dofs, fixed_dofs] = 1

    # Zero out the corresponding entries in F_global
    f_global[fixed_dofs] = 0

    print(f"K_global shape: {k_global.shape}, F_global shape: {f_global.shape}")
    print("Boundary conditions applied.")
    return k_global, f_global

def matvec_dask(k_global, x):
    """Performs matrix-vector multiplication.

    Args:
        K_global (cupyx.scipy.sparse.csr_matrix): The global stiffness 
        matrix in CSR (Compressed Sparse Row) format or as a CuPy dense array.
        x (array): The vector to multiply with the global stiffness matrix, 
        typically representing nodal displacements or forces.

    Returns:
        array: The result of the matrix-vector multiplication as a CuPy array.
    """
    return k_global.dot(x)

def solve_displacements_dask(k_global, f_global, method="gmres"):
    """Solves for nodal displacements in a finite element mesh using iterative solvers.

    Args:
        K_global (cupyx.scipy.sparse.csr_matrix): _description_
        F_global (array): _description_
        method (str, optional): The iterative method to use for solving the linear system.
        Defaults to 'gmres'.

    Raises:
        ValueError: _description_

    Returns:
        array: The global displacement vector computed using the chosen iterative solver.
    """
    print(f"Solving for displacements using {method.upper()} with Dask...")

    # Ensure the global stiffness matrix is in CSR format for efficient matrix-vector operations
    if isinstance(k_global, da.Array):
        k_global = k_global.map_blocks(csr_matrix)  # Convert Dask chunks to CSR format if needed

    # Define a matrix-vector product function for the solver using Dask
    k_operator = LinearOperator(k_global.shape, matvec=lambda x: matvec_dask(k_global, x))

    print(f"K_global size: {k_global.shape}, F_global size: {f_global.shape}")

    if method == "gmres":
        # Ensure both k_global and f_global are computed to Cupy arrays before passing them to gmres
        k_global = k_global.compute()  # Compute the Dask array to CuPy if it's on the GPU
        f_global = f_global.compute()

        # Using GMRES solver on GPU
        u_global, info = gmres(k_operator, f_global, tol=1e-8, maxiter=3135)
    elif method == "bicgstab":
        print("Falling back to CPU-based BiCGSTAB solver...")
        # Transfer the matrix and vector to NumPy arrays (on the CPU)
        k_global_cpu = k_global.compute()  # Compute the Dask array to NumPy if it's on the CPU
        f_global_cpu = f_global.compute()

        # Use CPU-based BiCGSTAB solver
        u_global_cpu, info = bicgstab(k_global_cpu, f_global_cpu, rtol=1e-8, maxiter=3135)

        # Transfer result back to GPU as CuPy array
        u_global = cp.asarray(u_global_cpu)
    else:
        raise ValueError(f"Unknown method: {method}")

    if info != 0:
        print(f"{method.upper()} solver did not converge. Info: {info}")
    else:
        print(f"Displacements solved using {method.upper()}.")

    return u_global

def compute_stresses_dask(mesh, u_global, e, nu):
    """Computes the stress for each element in a finite element mesh 
    using the global displacement vector in parallel using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains 
        the geometry and connectivity information of the entire domain.
        U_global (dask.array.Array or cupy.ndarray): The global displacement vector 
        as a Dask or CuPy array.
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        dask.array.Array: A Dask array containing the maximum stress for each element in the mesh.
    """
    print("Computing stresses in parallel...")

    # Convert mesh points to a CuPy array once to avoid repeated conversions
    mesh_points = cp.array(mesh.points)
    dn_dxi = cp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Precompute range for DOFs
    dof_range = cp.arange(3)

    # Function to compute stress for a single element
    @dask.delayed
    def compute_element_stress(i):
        cell = mesh.get_cell(i)
        nodes = cp.array(cell.point_ids, dtype=cp.int32)
        element_nodes = mesh_points[nodes]

        j = compute_jacobian_dask(element_nodes, dn_dxi)
        det_j = cp.linalg.det(j)

        if cp.abs(det_j) < 1e-12:
            print(f"Warning: Degenerate element detected at cell {i}. Skipping.")
            return 0.0  # Return 0 stress for degenerate elements

        j_inv = cp.linalg.inv(j)
        b = compute_b_matrix_dask(j_inv, dn_dxi)
        c = compute_c_matrix_dask(e, nu)

        # Efficiently gather U_elem using advanced indexing
        u_elem = u_global[(nodes[:, None] * 3 + dof_range).ravel()]

        epsilon = cp.dot(b, u_elem)
        sigma = cp.dot(c, epsilon)
        return cp.max(sigma)

    # Create delayed tasks for each element's stress computation
    delayed_stresses = [compute_element_stress(i) for i in range(mesh.n_cells)]

    # Convert the delayed results into a Dask array for parallel computation
    stresses_dask = da.from_delayed(delayed_stresses, shape=(mesh.n_cells,), dtype=cp.float32)

    # Trigger the computation and return the Dask array
    print("Stresses computed in parallel.")
    return stresses_dask

def compute_stress_tensor_dask(mesh, u_global, e, nu):
    """Computes the stress tensor for each element in a finite element mesh 
    using the global displacement vector in parallel using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry 
        and connectivity information of the entire domain.
        U_global (dask.array.Array or cupy.ndarray): The global displacement 
        vector as a Dask or CuPy array.
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        dask.array.Array: A Dask array containing the stress tensor components 
        for each element in the mesh.
    """
    print("Computing stress tensor in parallel...")

    # Convert mesh points to a CuPy array once to avoid repeated conversions
    mesh_points = cp.array(mesh.points)
    dn_dxi = cp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Precompute range for DOFs
    dof_range = cp.arange(3)

    # Function to compute stress tensor for a single element
    @dask.delayed
    def compute_element_stress(i):
        cell = mesh.get_cell(i)
        nodes = cp.array(cell.point_ids, dtype=cp.int32)
        element_nodes = mesh_points[nodes]

        j = compute_jacobian_dask(element_nodes, dn_dxi)
        det_j = cp.linalg.det(j)

        if cp.abs(det_j) < 1e-12:
            print(f"Warning: Degenerate element detected at cell {i}. Skipping.")
            return cp.zeros(6)  # Return zero stress tensor for degenerate elements

        j_inv = cp.linalg.inv(j)
        b = compute_b_matrix_dask(j_inv, dn_dxi)
        c = compute_c_matrix_dask(e, nu)

        # Efficiently gather U_elem using advanced indexing
        u_elem = u_global[(nodes[:, None] * 3 + dof_range).ravel()]

        epsilon = cp.dot(b, u_elem)
        sigma = cp.dot(c, epsilon)

        # Return the 6 components of the stress tensor
        return cp.array([sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5]])

    # Create delayed tasks for each element's stress tensor computation
    delayed_stresses = [compute_element_stress(i) for i in range(mesh.n_cells)]

    # Convert the delayed results into a Dask array for parallel computation
    stresses_dask = da.stack([da.from_delayed(
                                            stress,
                                            shape=(6,),
                                            dtype=cp.float32
                                            ) for stress in delayed_stresses])

    print("Stress tensor computed in parallel.")
    return stresses_dask

def calculate_normal_stress_dask(stress_tensor, plane_normal):
    """Calculates the normal stress on a plane defined by a given normal vector in 
    parallel using Dask.

    Args:
        stress_tensor (dask.array.Array or array): A 2D array where each row contains 
        the 6 components of the stress tensor:[σ_xx, σ_yy, σ_zz, τ_xy, τ_xz, τ_yz].
        plane_normal (dask.array.Array or array): A 2D array where each row contains 
        the 3 components of the normal vector defining the plane: [nx, ny, nz].

    Raises:
        ValueError: If the length of the second dimension of `stress_tensor` is not 6, or 
                    if the second dimension of `plane_normal` is not 3.

    Returns:
        dask.array.Array: The normal stress for each combination of stress tensor and plane normal.
    """
    # Ensure stress_tensor has 6 components and plane_normal has 3 components
    nx, ny, nz = plane_normal
    if len(stress_tensor) != 6:
        raise ValueError(
            f"Expected 6 components in stress_tensor, got {len(stress_tensor)}"
        )

    sigma_xx, sigma_yy, sigma_zz = stress_tensor[:3]
    tau_xy, tau_xz, tau_yz = stress_tensor[3:]

    # Normal stress on the given plane
    normal_stress = (
        (nx**2) * sigma_xx
        + (ny**2) * sigma_yy
        + (nz**2) * sigma_zz
        + 2 * nx * ny * tau_xy
        + 2 * nx * nz * tau_xz
        + 2 * ny * nz * tau_yz
    )
    return normal_stress

def calculate_shear_strength_dask(cohesion, normal_stress, friction_angle):
    """Calculates the shear strength of a material using the Mohr-Coulomb failure 
    criterion in parallel using Dask.

    Args:
        cohesion (float, array, or dask.array.Array): The cohesion of the material. 
        Can be a scalar, CuPy array, or Dask array.
        normal_stress (float, array, or dask.array.Array): The normal stress acting 
        on the plane of potential failure. 
        Can be a scalar, CuPy array, or Dask array.
        friction_angle (float, array, or dask.array.Array): The internal friction 
        angle of the material, in degrees. This parameter determines the slope of 
        the failure envelope. Can be a scalar, CuPy array, or Dask array.

    Returns:
        float or dask.array.Array: The calculated shear strength using the Mohr-Coulomb 
        failure criterion.
    """
    # If inputs are not Dask arrays, convert them
    if not isinstance(cohesion, da.Array):
        cohesion = da.from_array(cohesion, chunks="auto")

    if not isinstance(normal_stress, da.Array):
        normal_stress = da.from_array(normal_stress, chunks="auto")

    if not isinstance(friction_angle, da.Array):
        friction_angle = da.from_array(friction_angle, chunks="auto")

    # Convert friction angle to radians and compute tangent
    tan_phi = da.map_blocks(cp.tan, da.map_blocks(cp.radians, friction_angle))

    # Calculate shear strength using the Mohr-Coulomb criterion
    shear_strength = cohesion + normal_stress * tan_phi

    return shear_strength

def compute_fos_dask(stresses, cohesions, friction_angles, mesh, plane_normal=(0, 0, 1)):
    """Calculates the Factor of Safety (FoS) for each element in a finite element mesh using Dask.

    Args:
        stresses (dask.array.Array or cupy.ndarray): A Dask array containing the stress 
        tensors for each element in the mesh.
        Each stress tensor is represented by 6 components:[σ_xx, σ_yy, σ_zz, τ_xy, τ_xz, τ_yz].
        cohesions (dask.array.Array or cupy.ndarray): A Dask array containing the 
        cohesion values for each element in the mesh.
        friction_angles (dask.array.Array or cupy.ndarray): A Dask array containing 
        the internal friction angles
        (in degrees) for each element in the mesh.
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry
        and connectivity information of the entire domain.
        plane_normal (tuple, optional): A tuple representing the normal vector of the plane
        on which to compute the normal stress. Defaults to (0, 0, 1).

    Raises:
        ValueError: If the stress tensor for any cell does not have the correct dimensions or size.

    Returns:
        dask.array.Array: A Dask array containing the Factor of Safety for each element in the mesh.
    """
    print("Calculating Factor of Safety (FoS) in parallel...")

    # Function to calculate FoS for each element
    @dask.delayed
    def compute_element_fos(i):
        # Extract the full stress tensor for the current cell
        stress_tensor = stresses[i]

        # Ensure the stress_tensor has the correct dimensions
        if stress_tensor.ndim != 1 or stress_tensor.size != 6:
            raise ValueError(
                f"Stress tensor {i} has incorrect dimensions or size: {stress_tensor.shape}"
            )

        # Calculate the normal stress on the specified plane
        normal_stress = calculate_normal_stress_dask(stress_tensor, plane_normal)

        if normal_stress > 0:
            # Get the spatially varying material properties for the current cell
            cohesion = cohesions[i]
            friction_angle = friction_angles[i]

            # Calculate shear strength for the element
            shear_strength = calculate_shear_strength_dask(
                cohesion, normal_stress, friction_angle
            )

            # Calculate FoS for the element
            return shear_strength / normal_stress

        return float("inf")  # If normal_stress <= 0, return infinity for safety

    # Create a list of delayed tasks for each element
    delayed_fos = [compute_element_fos(i) for i in range(mesh.n_cells)]

    # Convert the delayed results into a Dask array for parallel computation
    fos_dask = da.from_delayed(delayed_fos, shape=(mesh.n_cells,), dtype=cp.float32)

    print("Factor of Safety (FoS) calculated in parallel.")
    return fos_dask

def identify_failing_elements_dask(fos, threshold=1.0):
    """
    Identify elements with a Factor of Safety (FoS) below the specified threshold using Dask.

    Args:
        fos (dask.array.Array or cp.ndarray): Dask array or CuPy array of FoS values for 
        each element.
        threshold (float): Threshold value for FoS to identify failing elements.

    Returns:
        dask.array.Array: Dask array mask indicating elements at risk of failure.
    """
    print(f"Identifying elements with FoS ≤ {threshold}...")

    # Ensure the `fos` array is a Dask array
    if not isinstance(fos, da.Array):
        fos = da.from_array(fos, chunks="auto")

    # Identify failing elements using Dask's element-wise comparison
    failing_elements = fos <= threshold

    # Compute the number of failing elements
    num_failing_elements = da.count_nonzero(failing_elements)

    # Trigger computation to get the number of failing elements
    num_failing_elements_computed = num_failing_elements.compute()

    print(f"Number of failing elements identified: {num_failing_elements_computed}")

    return failing_elements
# End-of-file (EOF)
