"""Contains the functions that provide analysis in dask"""
import dask
import dask.array as da
import cupy as cp  # CuPy for GPU acceleration

import numpy as np  # NumPy for handling some CPU-based operations
from scipy.ndimage import label

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

def norm(x):
    """Wrapping the numpy function to calculate determinant
        into a dask function

    Args:
        x (array): The array whose determinant is needed

    Returns:
        float: determinant value
    """
    return np.linalg.norm(x)
da_norm = da.gufunc(norm, signature="(i)->()", output_dtypes=float, vectorize=True)
# ! <<<<<<<<<<<<<<<<<<<< math funcs >>>>>>>>>>>>>>>>>>>>

def analyze_stress_strain_dask(stress_tensor, strain_tensor, yield_stress, ultimate_stress):
    """
    Analyze stress and strain fields to identify areas with significant plastic deformation
    or where stress limits are exceeded using Dask.

    Args:
        stress_tensor (dask.array.Array or cupy.ndarray): Stress tensor for each element.
        strain_tensor (dask.array.Array or cupy.ndarray): Strain tensor for each element.
        yield_stress (float): Yield stress of the material.
        ultimate_stress (float): Ultimate stress of the material.

    Returns:
        dict: Dictionaries containing masks for plastic deformation and high shear stress.
    """
    print("Analyzing stress and strain fields in parallel...")

    # Ensure tensors are Dask arrays
    if not isinstance(stress_tensor, da.Array):
        stress_tensor = da.from_array(stress_tensor, chunks='auto')
    if not isinstance(strain_tensor, da.Array):
        strain_tensor = da.from_array(strain_tensor, chunks='auto')

    # Identify plastic deformation using vectorized operations
    plastic_deformation_mask = da.any(strain_tensor > yield_stress, axis=1)

    # Calculate shear stress components using vectorized operations
    shear_stress = da.sqrt(
        stress_tensor[:, 3] ** 2 + stress_tensor[:, 4] ** 2 + stress_tensor[:, 5] ** 2
    )

    # Identify high shear stress areas
    high_shear_stress_mask = shear_stress > ultimate_stress

    print("Stress and strain analysis completed in parallel.")

    return {
        "plastic_deformation": plastic_deformation_mask,
        "high_shear_stress": high_shear_stress_mask,
    }

def analyze_stress_strain_optimized_dask(
    stress_tensor, strain_tensor, yield_stress, ultimate_stress
):
    """
    Optimized analysis of stress and strain fields to identify areas with significant plastic 
    deformation or where stress limits are exceeded using Dask for parallel processing.

    Args:
        stress_tensor (dask.array.Array or cupy.ndarray): Stress tensor for each element.
        strain_tensor (dask.array.Array or cupy.ndarray): Strain tensor for each element.
        yield_stress (float): Yield stress of the material.
        ultimate_stress (float): Ultimate stress of the material.

    Returns:
        dict: Dask arrays containing masks for plastic deformation and high shear stress.
    """
    print("Analyzing stress and strain fields (Optimized with Dask)...")

    # Ensure tensors are Dask arrays
    if not isinstance(stress_tensor, da.Array):
        stress_tensor = da.from_array(stress_tensor, chunks='auto')
    if not isinstance(strain_tensor, da.Array):
        strain_tensor = da.from_array(strain_tensor, chunks='auto')

    # Identify plastic deformation using vectorized operations with Dask
    plastic_deformation_mask = da.any(strain_tensor > yield_stress, axis=1)

    # Calculate shear stress components using Dask and identify high shear stress
    shear_stress = da.sqrt(
        stress_tensor[:, 3] ** 2 + stress_tensor[:, 4] ** 2 + stress_tensor[:, 5] ** 2
    )
    high_shear_stress_mask = shear_stress > ultimate_stress

    print("Optimized stress and strain analysis completed (Dask).")

    return {
        "plastic_deformation": plastic_deformation_mask,
        "high_shear_stress": high_shear_stress_mask,
    }

def connected_component_analysis_dask(mesh, failing_elements):
    """
    Perform connected component analysis to detect coherent failure surfaces using Dask.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failing_elements (dask.array.Array or cp.ndarray): Indices of elements with FoS ≤ 1.0.

    Returns:
        dask.array.Array: Array indicating the connected component label for each failing element.
    """
    print("Performing connected component analysis (Dask)...")

    # Convert failing_elements to a Dask array
    if not isinstance(failing_elements, da.Array):
        failing_elements = da.from_array(failing_elements, chunks='auto')

    # Convert the failing elements to NumPy (for label operation compatibility with SciPy)
    failing_elements_np = failing_elements.compute() if isinstance(
        failing_elements, da.Array) else failing_elements.get()

    # Initialize a mask array to identify failing elements
    element_mask = np.zeros(mesh.n_cells, dtype=bool)
    element_mask[failing_elements_np] = True

    # Define 3D connectivity structure for connected component analysis
    structure = np.ones((3, 3, 3), dtype=int)

    # Perform connected component analysis
    labeled_array, num_features = label(element_mask, structure=structure)

    print(f"Number of connected components identified: {num_features}")
    return da.from_array(labeled_array, chunks='auto')

def extract_failure_surfaces_dask(mesh, connected_components):
    """
    Extract failure surfaces based on connected components using Dask.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        connected_components (dask.array.Array or np.ndarray): Array of connected component labels.

    Returns:
        dask.delayed.Delayed: A delayed object representing the list of extracted surfaces for 
        each connected component.
    """
    print("Extracting failure surfaces in parallel...")

    # Ensure connected_components is a NumPy array or Dask array
    if isinstance(connected_components, da.Array):
        connected_components = connected_components.compute()  # Convert Dask array to NumPy array

    connected_components = np.asarray(connected_components)

    # Get the unique labels, skipping label 0 (non-failing elements)
    unique_labels = np.unique(connected_components)

    # Delayed list of failure surfaces
    failure_surfaces = []

    # Iterate over each unique label, skipping label 0 (non-failing elements)
    for component_label in unique_labels:
        if component_label == 0:
            continue

        # Create a delayed task to extract the surface for the current component
        task = dask.delayed(mesh.extract_cells)(connected_components == component_label)
        failure_surfaces.append(task)

    # Combine the tasks into a single delayed object for computation
    failure_surfaces_delayed = dask.delayed(failure_surfaces)

    print(f"Number of failure surfaces to extract: {len(failure_surfaces)}")
    return failure_surfaces_delayed

def slope_stability_analysis_dask(
    failure_surfaces, stress_tensor, fos, external_factors
):
    """
    Perform a detailed slope stability and failure mode analysis using Dask for parallel processing.

    Args:
        failure_surfaces (list): A list of extracted failure surfaces, each potentially representing 
        a different failure mechanism.
        stress_tensor (dask.array.Array or cupy.ndarray): A tensor containing stress values for 
        each element in the mesh.
        fos (dask.array.Array or cupy.ndarray): An array containing the calculated Factor of 
        Safety for each element.
        external_factors (dict): A dictionary containing external factors such as seismic 
        coefficients, water pressures, etc., that affect slope stability.

    Returns:
        dask.delayed.Delayed: A delayed object containing results of the slope stability analysis, 
        including identified failure modes and critical surfaces.
    """
    print("Performing slope stability analysis in parallel...")

    # Iterate over each failure surface in parallel
    tasks = []
    for index, surface in enumerate(failure_surfaces):
        task = dask.delayed(analyze_surface)(index, surface, stress_tensor, fos, external_factors)
        tasks.append(task)

    # Aggregate results from the delayed tasks
    final_results = dask.delayed(aggregate_results)(tasks)

    print("Slope stability analysis scheduled for parallel execution.")
    return final_results

# ! <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>
def analyze_surface(index, surface, stress_tensor, fos, external_factors):
    """Helper function to analyze a single failure surface."""
    # Extract stress and FoS values for the current surface
    surface_stress = stress_tensor[surface]
    surface_fos = fos[surface]

    # Identify if any part of the surface is critical (FoS < 1.0)
    critical = np.any(surface_fos < 1.0)

    # Calculate shear stresses (using a simplified formula)
    shear_stresses = cp.sqrt(cp.sum(surface_stress[:, 3:] ** 2, axis=1))

    # Determine the predominant failure mode
    predominant_failure_mode = (
        "sliding"
        if cp.any(shear_stresses > external_factors.get("shear_stress_threshold", 100))
        else "settling"
    )

    result = {
        "surface_index": index,
        "mode": predominant_failure_mode,
        "is_critical": critical,
    }

    # If critical, add the surface to the critical surfaces list
    critical_surfaces = []
    if critical:
        critical_surfaces.append(surface)

    return {"result": result, "critical_surfaces": critical_surfaces}

def aggregate_results(tasks):
    """Helper function to aggregate results from parallel tasks."""
    results = {"failure_modes": [], "critical_surfaces": []}

    # Gather all the results
    for task in tasks:
        result = task["result"]
        results["failure_modes"].append(result)
        results["critical_surfaces"].extend(task["critical_surfaces"])

    return results
# ? <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>

def analyze_failure_modes_dask(failure_surfaces):
    """
    Analyze the geometry and orientation of failure surfaces to determine 
    likely failure modes using Dask.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.

    Returns:
        dask.delayed.Delayed: A delayed object representing the list of likely 
        failure modes for each failure surface.
    """
    print("Analyzing geometry and orientation of failure surfaces in parallel...")

    # Create a list of delayed tasks for each failure surface
    tasks = []
    for surface in failure_surfaces:
        task = dask.delayed(analyze_single_surface)(surface)
        tasks.append(task)

    # Combine the tasks into a single delayed object for parallel execution
    failure_modes_delayed = dask.delayed(tasks)

    print("Failure mode analysis scheduled for parallel execution.")
    return failure_modes_delayed

# ! <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>
def analyze_single_surface(surface):
    """Helper function to analyze the geometry and orientation of a single failure surface."""
    # Convert UnstructuredGrid to PolyData for normal computation
    surface_polydata = surface.extract_surface()

    # Compute normals for the surface
    surface_with_normals = surface_polydata.compute_normals(
        point_normals=True, cell_normals=False, inplace=False
    )

    # Access the computed point normals
    normals = surface_with_normals.point_data["Normals"]
    average_normal = np.mean(normals, axis=0)
    average_normal = average_normal / np.linalg.norm(average_normal)  # Normalize

    # Determine failure mode based on surface orientation
    if np.abs(average_normal[2]) > 0.9:  # Mostly vertical
        failure_mode = "Toppling"
    elif np.abs(average_normal[0]) > 0.9 or np.abs(average_normal[1]) > 0.9:  # Mostly horizontal
        failure_mode = "Sliding"
    else:
        failure_mode = "Slope Failure"

    return failure_mode
# ? <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>

def analyze_shear_bands_dask(mesh, strain_tensor, shear_strain_threshold):
    """
    Enhanced identification of potential shear bands by analyzing areas 
    with concentrated shear strain using Dask and CuPy for parallel and GPU-accelerated computation.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        strain_tensor (dask.array.Array or cp.ndarray): Strain tensor for each element.
        shear_strain_threshold (float): Threshold for detecting high shear strain.

    Returns:
        pv.UnstructuredGrid: A subset of the mesh representing potential shear bands.
    """
    print("Identifying potential shear bands in parallel...")

    # Ensure strain_tensor is a Dask array or CuPy array
    if not isinstance(strain_tensor, (da.Array, cp.ndarray)):
        raise ValueError("strain_tensor must be a Dask or CuPy ndarray.")

    # If the strain tensor is not already a Dask array, convert it
    if not isinstance(strain_tensor, da.Array):
        strain_tensor = da.from_array(strain_tensor, chunks='auto')

    # Calculate the magnitude of shear strain in parallel using Dask and CuPy
    shear_strain_magnitude = da.sqrt(
        strain_tensor[:, 3] ** 2 + strain_tensor[:, 4] ** 2 + strain_tensor[:, 5] ** 2
    )

    # Identify elements with shear strain above the threshold in parallel
    shear_band_elements = da.where(shear_strain_magnitude > shear_strain_threshold)[0]

    # Compute the results to extract the shear band elements (trigger the parallel computation)
    shear_band_elements = shear_band_elements.compute()

    # Extract the shear band regions from the mesh
    shear_bands = mesh.extract_cells(shear_band_elements)

    print(f"Number of elements identified as shear bands: {len(shear_band_elements)}")
    return shear_bands

def analyze_slip_surface_dask(mesh, u_global, displacement_threshold):
    """
    Detect potential slip surfaces by analyzing zones of continuous displacement using Dask.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        u_global (dask.array.Array or cp.ndarray): Displacement vector for each node in the mesh.
        displacement_threshold (float): Threshold for detecting significant displacements.

    Returns:
        pv.UnstructuredGrid: A subset of the mesh representing potential slip surfaces.
    """
    print("Performing slip surface analysis in parallel...")

    # Ensure u_global is a Dask array or convert it
    if not isinstance(u_global, da.Array):
        u_global = da.from_array(u_global, chunks='auto')

    # Calculate the displacement magnitude for each node in parallel
    displacement_magnitude = da_norm(u_global.reshape(-1, 3), axis=1)

    # Identify elements where displacement exceeds the threshold in parallel
    slip_surface_elements = da.where(displacement_magnitude > displacement_threshold)[0]
    # Compute the slip surface elements (trigger the parallel computation)
    slip_surface_elements = slip_surface_elements.compute()

    # Extract the slip surface regions from the mesh
    slip_surfaces = mesh.extract_cells(slip_surface_elements)

    print(
        f"Number of elements identified as slip surfaces: {len(slip_surface_elements)}"
    )
    return slip_surfaces

def get_displacement_vectors_dask(failure_surfaces, u_global):
    """
    Calculate average displacement vectors for cells in each failure surface of the mesh using Dask 
    for large data handling.

    Args:
        failure_surfaces (list of pv.UnstructuredGrid): List of mesh subsets, 
        each representing a failure surface.
        u_global (dask.array.Array or cp.ndarray): Flattened array of displacements for all 
        nodes in the mesh. Expected to have a length that is a multiple of 3, as each node's 
        displacement is represented by three consecutive elements (x, y, z displacements).

    Returns:
        list of numpy.ndarray: Each element of the list is a numpy array 
        where each row represents the average displacement vector of all 
        nodes within a cell of the corresponding failure surface.

    Raises:
        ValueError: If any node index in the surfaces is invalid 
        or if `u_global` is not structured as expected.
    """
    # If u_global is a Dask array, compute it
    if isinstance(u_global, da.Array):
        u_global_np = u_global.compute()
    else:
        # If it's a CuPy array, convert it to NumPy
        u_global_np = u_global.get()

    # Ensure that the displacement array has the correct structure
    if u_global_np.ndim != 1 or (len(u_global_np) % 3) != 0:
        raise ValueError(
            "Global displacement array must be a flat array with length a multiple of 3."
        )

    displacement_vectors = []
    for surface in failure_surfaces:
        # Preallocate array to store average displacements for each cell
        surface_vectors = np.zeros((surface.n_cells, 3))

        for i in range(surface.n_cells):
            cell = surface.get_cell(i)
            node_indices = np.array(cell.point_ids).astype(int)

            if np.any(node_indices >= len(u_global_np) // 3):
                raise ValueError(
                    f"Invalid node index found in cell {i}; indices exceed displacement array size."
                )

            # Efficiently fetch displacements for all nodes in the cell
            node_displacements = u_global_np[
                node_indices * 3 : (node_indices * 3 + 3)
            ].reshape(-1, 3)
            surface_vectors[i] = np.mean(node_displacements, axis=0)

        displacement_vectors.append(surface_vectors)


    return displacement_vectors

def identify_failing_elements_dask(fos, threshold=1.0):
    """
    Identify elements with a Factor of Safety (FoS) below the specified threshold using Dask.

    Args:
        fos (dask.array.Array or cp.ndarray): Array of FoS values for each element.
        threshold (float): Threshold value for FoS to identify failing elements.

    Returns:
        dask.array.Array: Mask array indicating elements at risk of failure.
    """
    print(f"Identifying elements with FoS ≤ {threshold} in parallel...")

    # Ensure FoS is a Dask array
    if not isinstance(fos, da.Array):
        fos = da.from_array(fos, chunks='auto')

    # Use Dask to identify failing elements in parallel
    failing_elements = da.where(fos <= threshold)[0]

    print("Failing elements identified (Dask).")
    return failing_elements

def analyze_cause_of_failure_dask(failure_surfaces, stresses, strains, fos):
    """
    Analyze the causes of failure for each surface based on stress, strain, and FoS data using Dask.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.
        stresses (dask.array.Array or cp.ndarray): Stress tensor for each element.
        strains (dask.array.Array or cp.ndarray): Strain tensor for each element.
        fos (dask.array.Array or cp.ndarray): Factor of Safety for each element.

    Returns:
        dask.delayed.Delayed: Results of the cause of failure analysis, executed in parallel.
    """
    print("Analyzing causes of failure in parallel...")

    # List to store delayed tasks for analyzing each failure surface
    tasks = []

    for surface in failure_surfaces:
        # Delayed task to analyze a single failure surface
        task = dask.delayed(analyze_surface_failure)(surface, stresses, strains, fos)
        tasks.append(task)

    # Combine tasks into a single delayed object for parallel execution
    failure_causes = dask.delayed(tasks)

    print("Cause of failure analysis scheduled for parallel execution.")
    return failure_causes

# ! <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>
def analyze_surface_failure(surface, stresses, strains, fos):
    """
    Helper function to analyze the cause of failure for a single surface.

    Args:
        surface (pv.UnstructuredGrid): The surface being analyzed.
        stresses (dask.array.Array or cp.ndarray): Stress tensor for each element.
        strains (dask.array.Array or cp.ndarray): Strain tensor for each element.
        fos (dask.array.Array or cp.ndarray): Factor of Safety for each element.

    Returns:
        dict: Cause of failure information for the surface.
    """
    # Extract stress, strain, and FoS values for the current surface
    stress_values = stresses[surface]
    strain_values = strains[surface]
    fos_values = fos[surface]

    # Analyze stress and strain values
    stress_magnitude = cp.linalg.norm(stress_values, axis=1)
    strain_magnitude = cp.linalg.norm(strain_values, axis=1)

    # Determine cause of failure based on thresholds
    max_stress = cp.max(stress_magnitude)
    max_strain = cp.max(strain_magnitude)
    min_fos = cp.min(fos_values)

    # Return cause of failure data for this surface
    return {"max_stress": max_stress, "max_strain": max_strain, "min_fos": min_fos}
# ? <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>

def analyze_failure_directions_dask(failure_surfaces, mesh):
    """
    Extract failure directions based on the orientation of the failure surfaces using Dask.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.
        mesh (pv.UnstructuredGrid): The mesh of the model.

    Returns:
        dask.delayed.Delayed: A delayed object representing a list of failure directions 
        for each failure surface.
    """
    print("Extracting failure directions in parallel...")

    # List to store delayed tasks for each failure surface
    tasks = []

    for surface_cells in failure_surfaces:
        # Create a delayed task to analyze failure direction for each surface
        task = dask.delayed(analyze_surface_direction)(surface_cells, mesh)
        tasks.append(task)

    # Combine all the delayed tasks into a single delayed object
    failure_directions_delayed = dask.delayed(tasks)

    print("Failure direction analysis scheduled for parallel execution.")
    return failure_directions_delayed


def analyze_surface_direction(surface_cells, mesh):
    """
    Helper function to compute failure direction for a single failure surface.

    Args:
        surface_cells (list): Cells belonging to the surface.
        mesh (pv.UnstructuredGrid): The mesh of the model.

    Returns:
        str: Failure direction for the surface.
    """
    # Extract the cells for the current surface
    surface = mesh.extract_cells(surface_cells)
    surface_polydata = surface.extract_surface()

    # Compute normals for the surface
    surface_with_normals = surface_polydata.compute_normals(
        point_normals=True, cell_normals=False, inplace=False
    )

    # Access the computed point normals
    normals = surface_with_normals.point_data["Normals"]
    average_normal = np.mean(normals, axis=0)
    average_normal = average_normal / np.linalg.norm(average_normal)  # Normalize

    # Determine failure direction based on surface orientation
    if np.abs(average_normal[2]) > 0.9:  # Mostly vertical
        direction = "Vertical"
    elif np.abs(average_normal[0]) > 0.9 or np.abs(average_normal[1]) > 0.9:  # Mostly horizontal
        direction = "Horizontal"
    else:
        direction = "Oblique"

    return direction

def calculate_failure_magnitude_dask(stress_tensor, strain_tensor, fos):
    """
    Calculate the magnitude of failure based on stress, strain, and FoS using Dask.

    Args:
        stress_tensor (dask.array.Array or cp.ndarray): Stress tensor for each element.
        strain_tensor (dask.array.Array or cp.ndarray): Strain tensor for each element.
        fos (dask.array.Array or cp.ndarray): Factor of Safety for each element.

    Returns:
        dask.array.Array: Magnitude of failure for each element.
    """
    print("Calculating failure magnitude in parallel...")

    # Ensure input tensors are Dask arrays
    if not isinstance(stress_tensor, da.Array):
        stress_tensor = da.from_array(stress_tensor, chunks='auto')
    if not isinstance(strain_tensor, da.Array):
        strain_tensor = da.from_array(strain_tensor, chunks='auto')
    if not isinstance(fos, da.Array):
        fos = da.from_array(fos, chunks='auto')

    # Calculate stress and strain magnitudes using Dask
    stress_magnitude = da_norm(stress_tensor, axis=1)
    strain_magnitude = da_norm(strain_tensor, axis=1)

    # Ensure no division by zero or invalid operations
    fos = da.where(fos == 0, da.nan, fos)  # Replace zero FoS with NaN to avoid division errors

    # Define failure magnitude as a combination of stress, strain, and FoS
    failure_magnitude = stress_magnitude + strain_magnitude - fos

    # Handle any NaN values that might arise
    failure_magnitude = da.nan_to_num(failure_magnitude, nan=0.0, posinf=0.0, neginf=0.0)

    return failure_magnitude

def compute_resultant_direction_and_magnitude_dask(
    failure_surfaces, mesh, failure_magnitude
):
    """
    Compute the resultant direction and magnitude of failure for each failure surface using Dask.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_magnitude (dask.array.Array or cp.ndarray): Magnitude of failure for each element.

    Returns:
        dask.delayed.Delayed: Resultant directions as normalized vectors.
        dask.delayed.Delayed: Resultant magnitudes for each failure surface.
    """
    print("Computing resultant direction and magnitude in parallel...")

    # Convert the failure_magnitude to Dask array if it's not already
    if not isinstance(failure_magnitude, da.Array):
        failure_magnitude = da.from_array(failure_magnitude, chunks='auto')

    # List to store delayed tasks for each failure surface
    directions_tasks = []
    magnitudes_tasks = []

    for surface_cells in failure_surfaces:
        # Create a delayed task for computing direction and magnitude for each surface
        task = dask.delayed(compute_surface_direction_and_magnitude)(
            surface_cells, mesh, failure_magnitude
        )
        directions_tasks.append(task[0])  # Direction
        magnitudes_tasks.append(task[1])  # Magnitude

    # Combine the delayed tasks into a single delayed object
    resultant_directions = dask.delayed(directions_tasks)
    resultant_magnitudes = dask.delayed(magnitudes_tasks)

    print("Resultant direction and magnitude analysis scheduled for parallel execution.")
    return resultant_directions, resultant_magnitudes

# ! <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>
def compute_surface_direction_and_magnitude(surface_cells, mesh, failure_magnitude):
    """
    Helper function to compute direction and magnitude for a single failure surface.

    Args:
        surface_cells (list): Cells belonging to the surface.
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_magnitude (dask.array.Array or np.ndarray): Magnitude of failure for each element.

    Returns:
        tuple: Resultant direction as a normalized vector and resultant magnitude.
    """
    # Extract the cells for the current surface
    surface = mesh.extract_cells(surface_cells)
    surface_polydata = surface.extract_surface()

    # Compute normals for the surface
    surface_with_normals = surface_polydata.compute_normals(
        point_normals=True, cell_normals=False, inplace=False
    )
    normals = surface_with_normals.point_data["Normals"]

    # Convert normals to NumPy array
    normals_np = np.array(normals)

    # Calculate the resultant vector for direction by averaging the normals
    resultant_vector = np.mean(normals_np, axis=0)
    resultant_vector /= np.linalg.norm(resultant_vector)  # Normalize

    # Convert failure_magnitude to NumPy array if needed
    failure_magnitude_np = failure_magnitude.compute() if isinstance(failure_magnitude, da.Array) else failure_magnitude

    # Compute the resultant magnitude as the sum of failure magnitudes for this surface
    surface_failure_magnitude = failure_magnitude_np[surface_cells]
    resultant_magnitude = np.sum(surface_failure_magnitude)

    return resultant_vector, resultant_magnitude
# ? <<<<<<<<<<<<<<<<<<<< helper funcs >>>>>>>>>>>>>>>>>>>>
