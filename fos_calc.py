"""Identify Fixed Nodes"""

import cupy as cp  # CuPy for GPU acceleration
from cupyx.scipy.sparse import csr_matrix

from tqdm import tqdm

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import bicgstab  # Fallback to CPU-based solver

from load_calc import compute_jacobian
from load_calc import compute_B_matrix
from load_calc import compute_C_matrix


def identify_fixed_nodes(mesh):
    """Identifies fixed nodes in a finite element mesh based on specified boundary conditions.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.

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


# Apply Boundary Conditions
def apply_boundary_conditions(K_global, F_global, fixed_nodes):
    """Applies boundary conditions to the global stiffness matrix and global force vector for a finite element model.

    Args:
        K_global (matrix): The global stiffness matrix in CSR (Compressed Sparse Row) format or as a CuPy dense array.
        F_global (array): The global force vector as a CuPy array. It represents the applied forces at each node of the mesh.
        fixed_nodes (array): An array or list of integers representing the indices of nodes that are fixed in the mesh.

    Returns:
        tuple: K_global (cupyx.scipy.sparse.csr_matrix or cp.ndarray), F_global (cp.ndarray)
    """
    print("Applying boundary conditions...")

    num_dofs_per_node = 3
    fixed_dofs = cp.array(
        [
            node * num_dofs_per_node + i
            for node in fixed_nodes
            for i in range(num_dofs_per_node)
        ]
    )

    # Create a mask for non-fixed DOFs
    non_fixed_dofs = cp.ones(K_global.shape[0], dtype=bool)
    non_fixed_dofs[fixed_dofs] = False

    # Zero out the rows and columns for fixed DOFs in K_global
    K_global[fixed_dofs, :] = 0
    K_global[:, fixed_dofs] = 0

    # Set diagonal for fixed DOFs
    K_global[fixed_dofs, fixed_dofs] = 1

    # Zero out the corresponding entries in F_global
    F_global[fixed_dofs] = 0

    print(f"K_global shape: {K_global.shape}, F_global shape: {F_global.shape}")
    print("Boundary conditions applied.")
    return K_global, F_global


def matvec(K_global, x):
    """Performs matrix-vector multiplication.

    Args:
        K_global (cupyx.scipy.sparse.csr_matrix): The global stiffness matrix in CSR (Compressed Sparse Row) format or as a CuPy dense array.
        x (array): The vector to multiply with the global stiffness matrix, typically representing nodal displacements or forces.

    Returns:
        array: The result of the matrix-vector multiplication as a CuPy array.
    """
    return K_global.dot(x)


def solve_displacements(K_global, F_global, method="gmres"):
    """Solves for nodal displacements in a finite element mesh using iterative solvers.

    Args:
        K_global (cupyx.scipy.sparse.csr_matrix): _description_
        F_global (array): _description_
        method (str, optional): The iterative method to use for solving the linear system. Defaults to 'gmres'.

    Raises:
        ValueError: _description_

    Returns:
        array: The global displacement vector computed using the chosen iterative solver.
    """
    print(f"Solving for displacements using {method.upper()} on GPU...")

    # Ensure the global stiffness matrix is in CSR format for efficient matrix-vector operations
    if not isinstance(K_global, csr_matrix):
        K_global = K_global.tocsr()

    # Define a matrix-vector product function for the solver
    K_operator = LinearOperator(K_global.shape, matvec=lambda x: matvec(x, K_global))

    print(f"K_global size: {K_global.shape}, F_global size: {F_global.shape}")

    if method == "gmres":
        # Using GMRES solver on GPU
        U_global, info = gmres(K_operator, F_global, tol=1e-8, maxiter=3135)
    elif method == "bicgstab":
        print("Falling back to CPU-based BiCGSTAB solver...")
        # Transfer to CPU for SciPy's BiCGSTAB
        K_global_cpu = K_global.get()  # Transfer to CPU
        F_global_cpu = cp.asnumpy(F_global)  # Transfer to CPU
        U_global_cpu, info = bicgstab(
            K_global_cpu, F_global_cpu, tol=1e-8, maxiter=3135
        )
        U_global = cp.asarray(U_global_cpu)  # Transfer back to GPU
    else:
        raise ValueError(f"Unknown method: {method}")

    if info != 0:
        print(f"{method.upper()} solver did not converge. Info: {info}")
    else:
        print(f"Displacements solved using {method.upper()}.")

    return U_global


def compute_stresses(mesh, U_global, E, nu):
    """Computes the stress for each element in a finite element mesh using the global displacement vector.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        U_global (array): The global displacement vector as a CuPy array.
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        array: A CuPy array containing the maximum stress for each element in the mesh.
    """
    print("Computing stresses...")

    stresses = cp.zeros(mesh.n_cells)
    dN_dxi = cp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Convert mesh points to a CuPy array once to avoid repeated conversions
    mesh_points = cp.array(mesh.points)

    # Precompute range for DOFs
    dof_range = cp.arange(3)

    for i in tqdm(range(mesh.n_cells)):
        cell = mesh.get_cell(i)
        nodes = cp.array(cell.point_ids, dtype=cp.int32)
        element_nodes = mesh_points[nodes]

        J = compute_jacobian(element_nodes, dN_dxi)
        det_J = cp.linalg.det(J)

        if cp.abs(det_J) < 1e-12:
            print(f"Warning: Degenerate element detected at cell {i}. Skipping.")
            continue

        J_inv = cp.linalg.inv(J)
        B = compute_B_matrix(J_inv, dN_dxi)
        C = compute_C_matrix(E, nu)

        # Efficiently gather U_elem using advanced indexing
        U_elem = U_global[(nodes[:, None] * 3 + dof_range).ravel()]

        epsilon = cp.dot(B, U_elem)
        sigma = cp.dot(C, epsilon)
        stresses[i] = cp.max(sigma)

    print("Stresses computed.")
    return stresses


def compute_stress_tensor(mesh, U_global, E, nu):
    """Computes the stress tensor for each element in a finite element mesh using the global displacement vector.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        U_global (array): The global displacement vector as a CuPy array.
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        array: A CuPy array containing the stress tensor components for each element in the mesh.
    """
    print("Computing stress tensor...")

    stresses = cp.zeros(
        (mesh.n_cells, 6)
    )  # Store 6 components of the stress tensor for each cell
    dN_dxi = cp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Convert mesh points to a CuPy array once, to avoid repeated conversions
    mesh_points = cp.array(mesh.points)

    # Precompute range for DOFs
    dof_range = cp.arange(3)

    for i in tqdm(range(mesh.n_cells)):
        cell = mesh.get_cell(i)
        nodes = cp.array(cell.point_ids, dtype=cp.int32)
        element_nodes = mesh_points[nodes]

        J = compute_jacobian(element_nodes, dN_dxi)
        det_J = cp.linalg.det(J)

        if cp.abs(det_J) < 1e-12:
            print(f"Warning: Degenerate element detected at cell {i}. Skipping.")
            continue

        J_inv = cp.linalg.inv(J)
        B = compute_B_matrix(J_inv, dN_dxi)
        C = compute_C_matrix(E, nu)

        # Efficiently gather U_elem using advanced indexing
        U_elem = U_global[(nodes[:, None] * 3 + dof_range).ravel()]

        epsilon = cp.dot(B, U_elem)
        sigma = cp.dot(C, epsilon)

        # Store the stress tensor components
        stresses[i, 0] = sigma[0]  # σ_xx
        stresses[i, 1] = sigma[1]  # σ_yy
        stresses[i, 2] = sigma[2]  # σ_zz
        stresses[i, 3] = sigma[3]  # τ_xy
        stresses[i, 4] = sigma[4]  # τ_xz
        stresses[i, 5] = sigma[5]  # τ_yz

    print("Stress tensor computed.")
    return stresses


def calculate_normal_stress(stress_tensor, plane_normal):
    """Calculates the normal stress on a plane defined by a given normal vector.

    Args:
        stress_tensor (array): A 1D array containing the 6 components of the stress tensor:[σ_xx, σ_yy, σ_zz, τ_xy, τ_xz, τ_yz].
        plane_normal (array): A 1D array containing the 3 components of the normal vector defining the plane: [nx, ny, nz].

    Raises:
        ValueError: If the length of `stress_tensor` is not 6.

    Returns:
        float: The normal stress on the plane defined by `plane_normal`.
    """
    # Calculate the normal stress on a plane given by plane_normal
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


def calculate_shear_strength(cohesion, normal_stress, friction_angle):
    """Calculates the shear strength of a material using the Mohr-Coulomb failure criterion.

    Args:
        cohesion (float or array): The cohesion of the material, which represents the shear strength at zero normal stress. Can be a scalar or a CuPy array.
        normal_stress (float or array): The normal stress acting on the plane of potential failure. Can be a scalar or a CuPy array.
        friction_angle (float or array): The internal friction angle of the material, in degrees. This parameter determines the slope of the failure envelope.

    Returns:
        float or array: The calculated shear strength using the Mohr-Coulomb failure criterion. The output will be a scalar if both `cohesion` and `normal_stress` are scalars, or a CuPy array if either input is a CuPy array.
    """
    # Convert friction angle to radians and compute tangent
    tan_phi = cp.tan(cp.radians(friction_angle))

    # Calculate shear strength using the Mohr-Coulomb criterion
    shear_strength = cohesion + normal_stress * tan_phi

    return shear_strength


def compute_fos(stresses, cohesions, friction_angles, mesh, plane_normal=(0, 0, 1)):
    """Calculates the Factor of Safety (FoS) for each element in a finite element mesh.

    Args:
        stresses (array): A CuPy array containing the stress tensors for each element in the mesh. Each stress tensor is represented by 6 components:[σ_xx, σ_yy, σ_zz, τ_xy, τ_xz, τ_yz].
        cohesions (array): A CuPy array containing the cohesion values for each element in the mesh.
        friction_angles (array): A CuPy array containing the internal friction angles (in degrees) for each element in the mesh.
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry and connectivity information of the entire domain.
        plane_normal (tuple, optional): A tuple representing the normal vector of the plane on which to compute the normal stress. Defaults to (0, 0, 1).

    Raises:
        ValueError: If the stress tensor for any cell does not have the correct dimensions or size.

    Returns:
        array: A CuPy array containing the Factor of Safety for each element in the mesh.
    """
    print("Calculating Factor of Safety (FoS)...")

    # Initialize FoS array with infinity to handle any potential divisions by zero
    fos = cp.full(mesh.n_cells, float("inf"))

    # Iterate over each cell to calculate FoS
    for i in tqdm(range(mesh.n_cells)):
        # Extract the full stress tensor for the current cell
        stress_tensor = stresses[i]

        # Ensure the stress_tensor has the correct dimensions
        if stress_tensor.ndim != 1 or stress_tensor.size != 6:
            raise ValueError(
                f"Stress tensor for cell {i} has incorrect dimensions or size: {stress_tensor.shape}"
            )

        # Calculate the normal stress on the specified plane (e.g., horizontal plane)
        normal_stress = calculate_normal_stress(stress_tensor, plane_normal)

        if normal_stress > 0:
            # Get the spatially varying material properties for the current cell
            cohesion = cohesions[i]
            friction_angle = friction_angles[i]

            # Calculate shear strength for the element
            shear_strength = calculate_shear_strength(
                cohesion, normal_stress, friction_angle
            )

            # Calculate FoS for the element
            fos[i] = shear_strength / normal_stress

    print("Factor of Safety (FoS) calculated.")
    return fos