"""COntains the functions that provide analysis"""
import cupy as cp  # CuPy for GPU acceleration

import numpy as np  # NumPy for handling some CPU-based operations
from scipy.ndimage import label

def analyze_stress_strain(stress_tensor, strain_tensor, yield_stress, ultimate_stress):
    """
    Analyze stress and strain fields to identify areas with significant plastic deformation
    or where stress limits are exceeded.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        stress_tensor (cp.ndarray): Stress tensor for each element.
        strain_tensor (cp.ndarray): Strain tensor for each element.
        yield_stress (float): Yield stress of the material.
        ultimate_stress (float): Ultimate stress of the material.

    Returns:
        dict: Dictionaries containing masks for plastic deformation and high shear stress.
    """
    print("Analyzing stress and strain fields...")

    # Identify plastic deformation using vectorized operations
    plastic_deformation_mask = cp.any(strain_tensor > yield_stress, axis=1)

    # Calculate shear stress components using vectorized operations
    shear_stress = cp.sqrt(
        stress_tensor[:, 3] ** 2 + stress_tensor[:, 4] ** 2 + stress_tensor[:, 5] ** 2
    )

    # Identify high shear stress areas
    high_shear_stress_mask = shear_stress > ultimate_stress

    print("Stress and strain analysis completed.")

    return {
        "plastic_deformation": plastic_deformation_mask,
        "high_shear_stress": high_shear_stress_mask,
    }

def analyze_stress_strain_optimized(
    stress_tensor, strain_tensor, yield_stress, ultimate_stress
):
    """
    Optimized analysis of stress and strain fields to identify areas with significant plastic 
    deformationb or where stress limits are exceeded.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        stress_tensor (cp.ndarray): Stress tensor for each element.
        strain_tensor (cp.ndarray): Strain tensor for each element.
        yield_stress (float): Yield stress of the material.
        ultimate_stress (float): Ultimate stress of the material.

    Returns:
        dict: Dictionaries containing masks for plastic deformation and high shear stress.
    """
    print("Analyzing stress and strain fields (Optimized)...")

    # Identify plastic deformation using vectorized operations
    plastic_deformation_mask = cp.any(strain_tensor > yield_stress, axis=1)

    # Calculate shear stress components and identify high shear stress
    shear_stress = cp.sqrt(
        stress_tensor[:, 3] ** 2 + stress_tensor[:, 4] ** 2 + stress_tensor[:, 5] ** 2
    )
    high_shear_stress_mask = shear_stress > ultimate_stress

    print("Optimized stress and strain analysis completed.")

    return {
        "plastic_deformation": plastic_deformation_mask,
        "high_shear_stress": high_shear_stress_mask,
    }

def connected_component_analysis(mesh, failing_elements):
    """
    Perform connected component analysis to detect coherent failure surfaces.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failing_elements (cp.ndarray): Indices of elements with FoS ≤ 1.0.

    Returns:
        np.ndarray: Array indicating the connected component label for each failing element.
    """
    print("Performing connected component analysis...")

    # Convert failing_elements from CuPy to NumPy array
    failing_elements_np = failing_elements.get()

    # Initialize a mask array to identify failing elements
    element_mask = np.zeros(mesh.n_cells, dtype=bool)
    element_mask[failing_elements_np] = True

    # Define 3D connectivity structure for 3D connected component analysis
    structure = np.ones((3, 3, 3), dtype=int)

    # Perform connected component analysis
    labeled_array, num_features = label(element_mask, structure=structure)

    print(f"Number of connected components identified: {num_features}")
    return labeled_array

def extract_failure_surfaces(mesh, connected_components):
    """
    Extract failure surfaces based on connected components.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        connected_components (np.ndarray): Array of connected component labels.

    Returns:
        list: A list of extracted surfaces for each connected component.
    """
    print("Extracting failure surfaces...")

    failure_surfaces = []

    # Ensure connected_components is a NumPy array
    connected_components = np.asarray(connected_components)

    # Iterate over each unique label in the connected components, skipping label 0
    unique_labels = np.unique(connected_components)
    for component_label in unique_labels:
        if component_label == 0:
            continue  # Skip non-failing elements

        # Extract the cells for the current connected component
        mask = connected_components == component_label
        surface = mesh.extract_cells(mask)
        failure_surfaces.append(surface)

    print(f"Number of failure surfaces extracted: {len(failure_surfaces)}")
    return failure_surfaces

def slope_stability_analysis(
    failure_surfaces, stress_tensor, fos, external_factors
):
    """
    Perform a detailed slope stability and failure mode analysis based on the provided mesh data, 
    failure surfaces, and external factors.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model, 
        typically representing a slope in geotechnical analysis.
        failure_surfaces (list): A list of extracted failure surfaces, 
        each potentially representing a different failure mechanism.
        stress_tensor (cp.ndarray): A tensor containing stress values for 
        each element in the mesh.
        strain_tensor (cp.ndarray): A tensor containing strain values for 
        each element in the mesh.
        fos (cp.ndarray): An array containing the calculated Factor of 
        Safety for each element.
        external_factors (dict): A dictionary containing external factors such as 
        seismic coefficients, water pressures, etc., that affect slope stability.

    Returns:
        dict: A dictionary containing results of the slope stability analysis, 
        including identified failure modes and critical surfaces.
    """
    print("Performing slope stability analysis...")

    results = {"failure_modes": [], "critical_surfaces": []}

    # Evaluate each failure surface for potential failure modes and criticality
    for index, surface in enumerate(failure_surfaces):
        # Assume the index corresponds to mesh cell indices for simplicity in this example
        surface_stress = stress_tensor[surface]
        surface_fos = fos[surface]

        # Identify if any part of the surface is below a FoS threshold
        critical = np.any(surface_fos < 1.0)
        if critical:
            results["critical_surfaces"].append(surface)

        # Identify failure modes based on stress characteristics
        shear_stresses = cp.sqrt(
            cp.sum(surface_stress[:, 3:] ** 2, axis=1)
        )  # Simplified shear stress calculation
        predominant_failure_mode = (
            "sliding"
            if cp.any(
                shear_stresses > external_factors.get("shear_stress_threshold", 100)
            )
            else "settling"
        )

        results["failure_modes"].append(
            {
                "surface_index": index,
                "mode": predominant_failure_mode,
                "is_critical": critical,
            }
        )

    print("Slope stability analysis completed.")
    return results

def analyze_failure_modes(failure_surfaces):
    """
    Analyze the geometry and orientation of failure surfaces to determine likely failure modes.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.

    Returns:
        list: A list of likely failure modes for each failure surface.
    """
    print("Analyzing geometry and orientation of failure surfaces...")

    failure_modes = []

    for surface in failure_surfaces:
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

        # Print the average normal for inspection
        print(f"Average normal vector: {average_normal}")

        # Determine failure mode based on surface orientation
        if np.abs(average_normal[2]) > 0.9:  # Mostly vertical
            failure_mode = "Toppling"
        elif (
            np.abs(average_normal[0]) > 0.9 or np.abs(average_normal[1]) > 0.9
        ):  # Mostly horizontal
            failure_mode = "Sliding"
        else:
            failure_mode = "Slope Failure"

        failure_modes.append(failure_mode)
        print(f"Surface analyzed: Mode = {failure_mode}")

    return failure_modes

def analyze_shear_bands(mesh, strain_tensor, shear_strain_threshold):
    """
    Enhanced identification of potential shear bands by analyzing areas 
    with concentrated shear strain directly on the GPU.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        strain_tensor (cp.ndarray): Strain tensor for each element, assumed to be on the GPU.
        shear_strain_threshold (float): Threshold for detecting high shear strain.

    Returns:
        pv.UnstructuredGrid: A subset of the mesh representing potential shear bands.
    """
    print("Identifying potential shear bands...")

    # Ensure strain tensor is a CuPy array for GPU acceleration
    if not isinstance(strain_tensor, cp.ndarray):
        raise ValueError("strain_tensor must be a CuPy ndarray.")

    # Calculate the magnitude of shear strain using GPU
    shear_strain_magnitude = cp.sqrt(
        strain_tensor[:, 3] ** 2 + strain_tensor[:, 4] ** 2 + strain_tensor[:, 5] ** 2
    )

    # Identify elements with shear strain above the threshold
    shear_band_elements = cp.nonzero(shear_strain_magnitude > shear_strain_threshold)[
        0
    ].get()

    # Extract the shear band regions from the mesh
    shear_bands = mesh.extract_cells(shear_band_elements)

    print(f"Number of elements identified as shear bands: {len(shear_band_elements)}")
    return shear_bands

def analyze_slip_surface(mesh, u_global, displacement_threshold):
    """
    Detect potential slip surfaces by analyzing zones of continuous displacement.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        U_global (cp.ndarray): Displacement vector for each node in the mesh.
        displacement_threshold (float): Threshold for detecting significant displacements.

    Returns:
        pv.UnstructuredGrid: A subset of the mesh representing potential slip surfaces.
    """
    print("Performing slip surface analysis...")

    # Calculate the displacement magnitude for each node
    displacement_magnitude = cp.linalg.norm(u_global.reshape(-1, 3), axis=1)
    displacement_magnitude_np = cp.asnumpy(displacement_magnitude)

    # Identify elements where displacement exceeds the threshold
    slip_surface_elements = np.where(
        displacement_magnitude_np > displacement_threshold
    )[0]

    # Extract the slip surface regions from the mesh
    slip_surfaces = mesh.extract_cells(slip_surface_elements)

    print(
        f"Number of elements identified as slip surfaces: {len(slip_surface_elements)}"
    )
    return slip_surfaces

def get_displacement_vectors(failure_surfaces, u_global):
    """
    Calculate average displacement vectors for cells in each failure surface of the mesh.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model, 
        not directly used but relevant if expansion needed.
        failure_surfaces (list of pv.UnstructuredGrid): List of mesh subsets, 
        each representing a failure surface.
        U_global_np (numpy.ndarray): Flattened array of displacements for all nodes in the mesh. 
        Expected to have a length that is a multiple of 3, as each node's displacement is 
        represented by three consecutive elements (x, y, z displacements).

    Returns:
        list of numpy.ndarray: Each element of the list is a numpy array 
        where each row represents the average displacement vector of all 
        nodes within a cell of the corresponding failure surface.

    Raises:
        ValueError: If any node index in the surfaces is invalid 
        or if `U_global_np` is not structured as expected.
    """
    u_global_np = u_global.get()

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

def identify_failing_elements(fos, threshold=1.0):
    """
    Identify elements with a Factor of Safety (FoS) below the specified threshold.

    Args:
        fos (cp.ndarray): Array of FoS values for each element.
        threshold (float): Threshold value for FoS to identify failing elements.

    Returns:
        cp.ndarray: Mask array indicating elements at risk of failure.
    """
    print(f"Identifying elements with FoS ≤ {threshold}...")
    failing_elements = cp.where(fos <= threshold)[0]
    print(f"Number of failing elements identified: {len(failing_elements)}")
    return failing_elements

def analyze_cause_of_failure(failure_surfaces, stresses, strains, fos):
    """
    Analyze the causes of failure for each surface based on stress, strain, and FoS data.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.
        stresses (cp.ndarray): Stress tensor for each element.
        strains (cp.ndarray): Strain tensor for each element.
        fos (cp.ndarray): Factor of Safety for each element.

    Returns:
        dict: Results of the cause of failure analysis.
    """
    print("Analyzing causes of failure...")

    failure_causes = []

    for surface in failure_surfaces:
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
        max_fos = cp.min(fos_values)

        cause = {"max_stress": max_stress, "max_strain": max_strain, "min_fos": max_fos}
        failure_causes.append(cause)

    print("Cause of failure analysis completed.")
    return failure_causes


def analyze_failure_directions(failure_surfaces, mesh):
    """
    Extract failure directions based on the orientation of the failure surfaces.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.
        mesh (pv.UnstructuredGrid): The mesh of the model.

    Returns:
        list: A list of failure directions for each failure surface.
    """
    print("Extracting failure directions...")

    failure_directions = []

    for surface_cells in failure_surfaces:
        # Extract the cells for the current surface
        surface = mesh.extract_cells(surface_cells)
        surface_polydata = surface.extract_surface()

        # Compute normals for the surface if not already available
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
        elif (
            np.abs(average_normal[0]) > 0.9 or np.abs(average_normal[1]) > 0.9
        ):  # Mostly horizontal
            direction = "Horizontal"
        else:
            direction = "Oblique"

        failure_directions.append(direction)
        print(f"Surface analyzed: Direction = {direction}")

    return failure_directions


def calculate_failure_magnitude(stress_tensor, strain_tensor, fos):
    """
    Calculate the magnitude of failure based on stress, strain, and FoS.

    Args:
        stress_tensor (cp.ndarray): Stress tensor for each element.
        strain_tensor (cp.ndarray): Strain tensor for each element.
        fos (cp.ndarray): Factor of Safety for each element.

    Returns:
        cp.ndarray: Magnitude of failure for each element.
    """
    print("Calculating failure magnitude...")

    # Calculate stress and strain magnitudes
    stress_magnitude = cp.linalg.norm(stress_tensor, axis=1)
    strain_magnitude = cp.linalg.norm(strain_tensor, axis=1)

    # Ensure no division by zero or invalid operations
    fos = cp.where(
        fos == 0, cp.nan, fos
    )  # Replace zero FoS with NaN to avoid division errors

    # Define failure magnitude as a combination of stress, strain, and FoS
    failure_magnitude = stress_magnitude + strain_magnitude - fos

    # Handle any NaN values that might arise
    failure_magnitude = cp.nan_to_num(
        failure_magnitude, nan=0.0, posinf=0.0, neginf=0.0
    )

    return failure_magnitude

def compute_resultant_direction_and_magnitude(
    failure_surfaces, mesh, failure_magnitude
):
    """
    Compute the resultant direction and magnitude of failure for each failure surface.

    Args:
        failure_surfaces (list): List of extracted failure surfaces.
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_magnitude (cp.ndarray): Magnitude of failure for each element.

    Returns:
        list: Resultant directions as normalized vectors.
        list: Resultant magnitudes for each failure surface.
    """
    print("Computing resultant direction and magnitude...")

    resultant_directions = []
    resultant_magnitudes = []

    failure_magnitude_np = failure_magnitude.get()  # Convert to NumPy array once

    for surface_cells in failure_surfaces:
        # Extract the cells for the current surface
        surface = mesh.extract_cells(surface_cells)
        surface_polydata = surface.extract_surface()

        # Compute normals for the surface if not already available
        surface_with_normals = surface_polydata.compute_normals(
            point_normals=True, cell_normals=False, inplace=False
        )
        normals = surface_with_normals.point_data["Normals"]

        # Convert to NumPy
        normals_np = np.array(normals)

        # Calculate the resultant vector for direction by averaging the normals
        resultant_vector = np.mean(normals_np, axis=0)
        resultant_vector /= np.linalg.norm(resultant_vector)  # Normalize

        # Compute the resultant magnitude as the sum of failure magnitudes for this surface
        surface_failure_magnitude = failure_magnitude_np[surface_cells]
        resultant_magnitude = np.sum(surface_failure_magnitude)

        resultant_directions.append(resultant_vector)
        resultant_magnitudes.append(resultant_magnitude)

    print("Resultant direction and magnitude computation completed.")
    return resultant_directions, resultant_magnitudes
# End-of-file (EOF)
