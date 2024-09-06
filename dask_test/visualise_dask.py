"""Visaulization functions in dask"""
import dask.array as da
import pyvista as pv
import cupy as cp
import numpy as np
import matplotlib.cm as cm

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

def plot_fixed_nodes_dask(mesh, fixed_nodes, point_size=10, mesh_opacity=0.5):
    """Plots a finite element mesh and highlights the fixed nodes using Dask.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry 
        and connectivity information of the entire domain.
        fixed_nodes (dask.array.Array or array-like): An array or list of indices representing 
        the fixed nodes in the mesh.
        point_size (int, optional): The size of the points used to highlight the fixed nodes. 
        Default is 10. Defaults to 10.
        mesh_opacity (float, optional): The opacity of the mesh. Value should be between 
        0 (completely transparent) and 1 (completely opaque). Defaults to 0.5.
    """
    # Ensure fixed_nodes is a Dask array for parallel processing if it's not already
    if not isinstance(fixed_nodes, da.Array):
        fixed_nodes = da.from_array(fixed_nodes, chunks='auto')

    # Compute the fixed node positions in parallel (this step will be parallelized if the data is large)
    fixed_points = fixed_nodes.map_blocks(lambda indices: mesh.points[indices]).compute()

    # Initialize PyVista plotter with a specified window size
    plotter = pv.Plotter(window_size=(1000, 600))
    plotter.add_text("Fixed Nodes", font_size=12)

    # Plot the entire mesh with adjustable opacity
    plotter.add_mesh(mesh, color="white", show_edges=True, opacity=mesh_opacity)

    # Highlight the fixed nodes
    plotter.add_points(
        fixed_points, color="red", point_size=point_size, render_points_as_spheres=True
    )

    # Display the plot
    plotter.show()

def identify_failing_elements_dask(fos, threshold=1.0):
    """
    Identify elements with a Factor of Safety (FoS) below the specified threshold using Dask.

    Args:
        fos (dask.array.Array or array-like): Array of FoS values for each element.
        threshold (float, optional): Threshold value for FoS to identify failing elements. Defaults to 1.0.

    Returns:
        dask.array.Array: Boolean mask array indicating elements at risk of failure.
    """
    print(f"Identifying elements with FoS ≤ {threshold}...")

    # Convert fos to a Dask array if it isn't one already
    if not isinstance(fos, da.Array):
        fos = da.from_array(fos, chunks='auto')
        print("Converted FoS array to Dask array with automatic chunking.")

    # Create a Boolean mask where FoS is less than or equal to the threshold
    failing_elements = fos <= threshold
    print("Created Boolean mask for failing elements.")

    # Count the number of failing elements using Dask's count_nonzero
    num_failing_elements = da.count_nonzero(failing_elements)
    print("Counting the number of failing elements...")

    # Compute the count to trigger the evaluation
    computed_num_failing_elements = num_failing_elements.compute()
    print(f"Number of failing elements identified: {computed_num_failing_elements}")

    return failing_elements

def visualize_failure_analysis_dask(
    mesh, failure_surfaces, stress_tensor, strain_tensor, fos
):
    """
    Visualize failure surfaces and stress/strain distributions using Dask for parallel processing.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list): List of extracted failure surfaces.
        stress_tensor (dask.array.Array or cp.ndarray): Stress tensor for each element.
        strain_tensor (dask.array.Array or cp.ndarray): Strain tensor for each element.
        fos (dask.array.Array or cp.ndarray): Factor of Safety for each element.
    """
    print("Visualizing failure analysis results...")

    plotter = pv.Plotter()

    # Visualize the original mesh with low opacity for background context
    plotter.add_mesh(
        mesh, opacity=0.1, show_edges=True, color="lightgrey", label="Original Mesh"
    )

    # Visualize each failure surface
    for i, surface in enumerate(failure_surfaces):
        plotter.add_mesh(surface, color="red", label=f"Failure Surface {i+1}")

    # Ensure stress_tensor, strain_tensor, and fos are Dask arrays and compute them
    if isinstance(stress_tensor, da.Array):
        stress_tensor_np = stress_tensor.compute()
    else:
        stress_tensor_np = cp.asnumpy(stress_tensor)

    if isinstance(strain_tensor, da.Array):
        strain_tensor_np = strain_tensor.compute()
    else:
        strain_tensor_np = cp.asnumpy(strain_tensor)

    if isinstance(fos, da.Array):
        fos_np = fos.compute()
    else:
        fos_np = cp.asnumpy(fos)

    # Add scalar bars and overlay stress, strain, and FoS on the mesh
    plotter.add_mesh(
        mesh,
        scalars=stress_tensor_np[:, 0],  # Visualize only the first component of stress
        cmap="coolwarm",
        opacity=0.6,
        label="Stress Distribution",
    )
    plotter.add_scalar_bar("Stress", title="Stress (MPa)")

    plotter.add_mesh(
        mesh,
        scalars=strain_tensor_np[:, 0],  # Visualize only the first component of strain
        cmap="viridis",
        opacity=0.6,
        label="Strain Distribution",
    )
    plotter.add_scalar_bar("Strain", title="Strain (%)")

    plotter.add_mesh(
        mesh, scalars=fos_np, cmap="jet", opacity=0.6, label="FoS Distribution"
    )
    plotter.add_scalar_bar("Factor of Safety", title="FoS")

    plotter.show()

def visualize_mesh_and_failure_surfaces_dask(
    mesh, failure_surfaces, mesh_color="white", mesh_opacity=0.5, surface_color="red"
):
    """
    Visualize the original mesh and the extracted failure surfaces using Dask for parallel processing
    of large datasets, with adjustable visual properties.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list): List of extracted failure surfaces.
        mesh_color (str): Color of the mesh. Default is "white".
        mesh_opacity (float): Opacity of the mesh, ranging from 0 (transparent) to 1 (opaque). 
        Default is 0.5.
        surface_color (str): Color used for failure surfaces. Default is "red".
    """
    print("Visualizing mesh and failure surfaces in parallel...")

    plotter = pv.Plotter()

    # Visualize the original mesh with adjustable opacity and color
    plotter.add_mesh(
        mesh,
        color=mesh_color,
        opacity=mesh_opacity,
        show_edges=True,
        label="Original Mesh",
    )

    # Ensure failure_surfaces are processed using Dask arrays if needed
    for i, surface in enumerate(failure_surfaces):
        if isinstance(surface, da.Array):
            surface = surface.compute()  # Convert Dask array to NumPy if necessary

        plotter.add_mesh(
            surface,
            color=surface_color,
            show_edges=True,
            label=f"Failure Surface {i+1}",
        )

    plotter.add_legend()
    plotter.show()

def visualize_stress_strain_fields_dask(mesh, stress_tensor, strain_tensor):
    """
    Visualize stress and strain fields on the mesh using Dask for performance optimizations 
    and parallel processing, with error handling.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        stress_tensor (dask.array.Array or cp.ndarray): Stress tensor for each element. 
        Shape should be (n_elements, n_stress_components).
        strain_tensor (dask.array.Array or cp.ndarray): Strain tensor for each element. 
        Shape should be (n_elements, n_strain_components).
    """
    print("Visualizing stress and strain fields...")

    # Validate inputs
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise ValueError("The mesh must be a PyVista UnstructuredGrid.")
    if stress_tensor.shape[1] != 3 or strain_tensor.shape[1] != 3:
        raise ValueError(
            "Stress and strain tensors must have 3 components per element."
        )

    # Ensure stress_tensor and strain_tensor are Dask arrays
    if not isinstance(stress_tensor, da.Array):
        stress_tensor = da.from_array(stress_tensor, chunks='auto')
    if not isinstance(strain_tensor, da.Array):
        strain_tensor = da.from_array(strain_tensor, chunks='auto')

    # Calculate the magnitude of stress and strain manually
    stress_magnitude = da.sqrt(da.sum(stress_tensor**2, axis=1))
    strain_magnitude = da.sqrt(da.sum(strain_tensor**2, axis=1))

    # Compute the Dask arrays and convert them to NumPy for visualization
    stress_magnitude_np = stress_magnitude.compute()  # Trigger parallel computation
    strain_magnitude_np = strain_magnitude.compute()

    plotter = pv.Plotter(shape=(1, 2))  # Two plots side by side

    # Stress field visualization
    plotter.subplot(0, 0)
    plotter.add_text("Stress Field", font_size=12)
    plotter.add_mesh(mesh, scalars=stress_magnitude_np, cmap="coolwarm", show_edges=True)
    plotter.add_scalar_bar("Stress Magnitude", format="%.2e")

    # Strain field visualization
    plotter.subplot(0, 1)
    plotter.add_text("Strain Field", font_size=12)
    plotter.add_mesh(mesh, scalars=strain_magnitude_np, cmap="viridis", show_edges=True)
    plotter.add_scalar_bar("Strain Magnitude", format="%.2e")

    plotter.show()

def visualize_fos_distribution_dask(mesh, fos, custom_range=None):
    """
    Visualize the Factor of Safety (FoS) distribution on the mesh using Dask for handling 
    large datasets, with options for customizing the color map range.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        fos (dask.array.Array or cp.ndarray): Factor of Safety for each element.
        custom_range (tuple, optional): Min and max values to define the color map scale. 
        Default is None.
    """
    print("Visualizing FoS distribution...")

    # If fos is a Dask array, compute it to convert to NumPy for PyVista
    if isinstance(fos, da.Array):
        fos_np = fos.compute()  # Trigger computation if Dask array
    else:
        fos_np = cp.asnumpy(fos)  # Convert CuPy to NumPy

    plotter = pv.Plotter()

    # Set up the scalar range based on FoS values or use custom range if provided
    scalar_range = custom_range if custom_range else (fos_np.min(), fos_np.max())

    # Set up a colormap that highlights critical values more distinctly
    cmap = "coolwarm"  # This colormap varies from blue (low) to red (high), good for safety factors

    plotter.add_mesh(
        mesh,
        scalars=fos_np,
        cmap=cmap,
        scalar_bar_args={"title": "Factor of Safety"},
        clim=scalar_range,
        show_edges=True,
    )

    # Enable user interaction for more detailed examination
    plotter.enable_zoom_style()
    plotter.enable_point_picking()

    plotter.show()

def visualize_connected_components_dask(mesh, connected_components):
    """
    Enhanced visualization of connected components of failing elements on the mesh using Dask 
    for efficient parallel processing.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        connected_components (dask.array.Array or np.ndarray): Array of connected component labels.
    """
    print("Visualizing connected components of failing elements...")

    # If connected_components is a Dask array, compute it to convert to NumPy for PyVista
    if isinstance(connected_components, da.Array):
        connected_components = connected_components.compute()

    plotter = pv.Plotter()
    unique_labels = np.unique(connected_components)

    # Generate a color map using Matplotlib
    cmap = cm.get_cmap("viridis", len(unique_labels) - 1)  # Exclude the zero label

    # Map each label to a color from the color map
    colors = [cmap(i) for i in range(len(unique_labels) - 1)]

    # Add each connected component with a unique color
    color_index = 0
    for label in unique_labels:
        if label == 0:
            continue  # Skip non-failing elements

        component = mesh.extract_cells(connected_components == label)
        plotter.add_mesh(
            component,
            color=colors[color_index],
            show_edges=True,
            label=f"Component {label}",
        )
        color_index += 1

    plotter.add_legend()
    # Enable user interaction for more detailed examination
    plotter.enable_zoom_style()
    plotter.enable_point_picking()
    plotter.show()

def visualize_shear_bands_and_slip_surfaces_dask(
    mesh,
    shear_bands,
    slip_surfaces,
    mesh_color="white",
    mesh_opacity=0.1,
    shear_band_color="blue",
    slip_surface_color="orange",
):
    """
    Enhanced visualization of shear bands and slip surfaces on the mesh using Dask for large 
    dataset handling, with customizable color and opacity settings.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        shear_bands (dask.array.Array or pv.UnstructuredGrid): Mesh subset representing shear bands.
        slip_surfaces (dask.array.Array or pv.UnstructuredGrid): Mesh subset representing slip 
        surfaces.
        mesh_color (str): Color for the original mesh.
        mesh_opacity (float): Opacity for the original mesh.
        shear_band_color (str): Color for the shear bands.
        slip_surface_color (str): Color for the slip surfaces.
    """
    print("Visualizing shear bands and slip surfaces...")

    plotter = pv.Plotter()

    # If shear_bands or slip_surfaces are Dask arrays, compute them before visualization
    if isinstance(shear_bands, da.Array):
        shear_bands = shear_bands.compute()  # Convert to NumPy or PyVista object if necessary

    if isinstance(slip_surfaces, da.Array):
        slip_surfaces = slip_surfaces.compute()  # Convert to NumPy or PyVista object if necessary

    # Visualize the original mesh with customizable transparency and color
    plotter.add_mesh(
        mesh,
        color=mesh_color,
        opacity=mesh_opacity,
        show_edges=True,
        label="Original Mesh",
    )

    # Visualize shear bands with a specified color
    plotter.add_mesh(
        shear_bands, color=shear_band_color, show_edges=True, label="Shear Bands"
    )

    # Visualize slip surfaces with a specified color
    plotter.add_mesh(
        slip_surfaces, color=slip_surface_color, show_edges=True, label="Slip Surfaces"
    )

    plotter.add_legend()
    plotter.enable_zoom_style()
    plotter.enable_terrain_style()
    plotter.show()

def plot_resultant_directions_and_magnitudes_dask(
    mesh, failure_surfaces, resultant_directions, resultant_magnitudes
):
    """
    Plot resultant directions and magnitudes on the mesh using Dask for large data handling.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list or dask.array.Array): List of extracted failure surfaces.
        resultant_directions (list or dask.array.Array): List of resultant directions for each failure surface.
        resultant_magnitudes (list or dask.array.Array): Resultant magnitudes for each failure surface.
    """
    print("Plotting resultant directions and magnitudes...")

    # If any of the inputs are Dask arrays, compute them to convert to NumPy
    if isinstance(failure_surfaces, da.Array):
        failure_surfaces = failure_surfaces.compute()

    if isinstance(resultant_directions, da.Array):
        resultant_directions = resultant_directions.compute()

    if isinstance(resultant_magnitudes, da.Array):
        resultant_magnitudes = resultant_magnitudes.compute()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, opacity=0.3, label="Mesh")

    # Iterate through failure surfaces and plot arrows for resultant directions
    for i, surface_cells in enumerate(failure_surfaces):
        # Extract the cells for the current surface
        surface = mesh.extract_cells(surface_cells)
        surface_polydata = surface.extract_surface()

        # Plot arrows representing resultant direction
        centers = surface_polydata.cell_centers().points
        resultant_vector = resultant_directions[i]

        # Scale the arrows by magnitude for visualization
        magnitude_scale = (
            resultant_magnitudes[i] / np.max(resultant_magnitudes)
            if np.max(resultant_magnitudes) > 0
            else 1.0
        )
        plotter.add_arrows(
            centers,
            np.tile(resultant_vector, (centers.shape[0], 1)),
            mag=0.1 * magnitude_scale,
            color="red",
        )

    # Add a color bar for failure magnitude
    plotter.add_scalar_bar(title="Failure Magnitude", n_labels=5)

    # Show the plot
    plotter.show()

def plot_original_mesh_dask(mesh):
    """
    Plot the original mesh using Dask for handling large datasets.

    Args:
        mesh (pv.UnstructuredGrid or dask.array.Array): The mesh of the model.
    """
    # If the mesh points or data are Dask arrays, compute them before plotting
    if isinstance(mesh.points, da.Array):
        mesh.points = mesh.points.compute()

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh, show_edges=True, opacity=0.5, color="lightgrey", label="Original Mesh"
    )
    plotter.add_scalar_bar(title="Original Mesh")
    plotter.show()

def plot_failure_results_dask(
    mesh, failure_surfaces, resultant_directions, resultant_magnitudes
):
    """
    Plot resultant directions and magnitudes on the mesh after failure analysis using Dask 
    for large data handling.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list or dask.array.Array): List of extracted failure surfaces.
        resultant_directions (list or dask.array.Array): List of resultant directions for each failure surface.
        resultant_magnitudes (list or dask.array.Array): Resultant magnitudes for each failure surface.
    """
    print("Plotting resultant directions and magnitudes...")

    # If any of the inputs are Dask arrays, compute them to convert to NumPy
    if isinstance(failure_surfaces, da.Array):
        failure_surfaces = failure_surfaces.compute()

    if isinstance(resultant_directions, da.Array):
        resultant_directions = resultant_directions.compute()

    if isinstance(resultant_magnitudes, da.Array):
        resultant_magnitudes = resultant_magnitudes.compute()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, opacity=0.3, label="Mesh")

    # Iterate through failure surfaces and plot arrows for resultant directions
    for i, surface_cells in enumerate(failure_surfaces):
        # Extract the cells for the current surface
        surface = mesh.extract_cells(surface_cells)
        surface_polydata = surface.extract_surface()

        # Plot arrows representing resultant direction
        centers = surface_polydata.cell_centers().points
        resultant_vector = resultant_directions[i]

        # Scale the arrows by magnitude for visualization
        magnitude_scale = (
            resultant_magnitudes[i] / np.max(resultant_magnitudes)
            if np.max(resultant_magnitudes) > 0
            else 1.0
        )
        plotter.add_arrows(
            centers,
            np.tile(resultant_vector, (centers.shape[0], 1)),
            mag=0.1 * magnitude_scale,
            color="red",
        )

    # Add a color bar for failure magnitude
    plotter.add_scalar_bar(title="Failure Magnitude", n_labels=5)

    # Show the plot
    plotter.show()

def plot_failing_cells_dask(mesh, failing_elements):
    """
    Plot the failing cells on the mesh after failure analysis using Dask for large data handling.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failing_elements (dask.array.Array or cp.ndarray): Indices of elements with FoS ≤ threshold.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh, show_edges=True, opacity=0.3, color="lightgrey", label="Mesh"
    )

    # If failing_elements is a Dask array, compute it
    if isinstance(failing_elements, da.Array):
        failing_elements_np = failing_elements.compute()
    else:
        # If failing_elements is a CuPy array, convert it to NumPy
        failing_elements_np = failing_elements.get()

    # Extract and highlight failing cells
    failing_cells = mesh.extract_cells(failing_elements_np)
    plotter.add_mesh(failing_cells, color="red", label="Failing Cells")

    # Show plot
    plotter.show()

def plot_failure_features_dask(
    mesh, failing_elements, failure_surfaces, resultant_directions, resultant_magnitudes
):
    """
    Plot the mesh with failure features using Dask for large data handling.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failing_elements (dask.array.Array or cp.ndarray): Indices of elements with FoS ≤ threshold.
        failure_surfaces (list or dask.array.Array): List of extracted failure surfaces.
        resultant_directions (list or dask.array.Array): List of resultant directions for each failure surface.
        resultant_magnitudes (list or dask.array.Array): Resultant magnitudes for each failure surface.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, opacity=0.3, label="Mesh")

    # Compute Dask or CuPy arrays
    if isinstance(failing_elements, da.Array):
        failing_elements_np = failing_elements.compute()
    else:
        failing_elements_np = failing_elements.get()  # If CuPy array

    if isinstance(failure_surfaces, da.Array):
        failure_surfaces = failure_surfaces.compute()

    if isinstance(resultant_directions, da.Array):
        resultant_directions = resultant_directions.compute()

    if isinstance(resultant_magnitudes, da.Array):
        resultant_magnitudes = resultant_magnitudes.compute()

    # Highlight failing elements
    failing_cells = mesh.extract_cells(failing_elements_np)
    plotter.add_mesh(failing_cells, color="red", label="Failing Cells")

    # Plot failure directions and magnitudes
    for i, surface_cells in enumerate(failure_surfaces):
        surface = mesh.extract_cells(surface_cells)
        surface_polydata = surface.extract_surface()

        # Plot arrows representing resultant direction
        centers = surface_polydata.cell_centers().points
        resultant_vector = resultant_directions[i]

        # Scale the arrows by magnitude for visualization
        magnitude_scale = (
            resultant_magnitudes[i] / np.max(resultant_magnitudes)
            if np.max(resultant_magnitudes) > 0
            else 1.0
        )
        plotter.add_arrows(
            centers,
            np.tile(resultant_vector, (centers.shape[0], 1)),
            mag=0.1 * magnitude_scale,
            color="blue",
        )

    # Annotate key failure features (example positions, adjust as needed)
    plotter.add_text("Crown", position=(0.1, 0.8), color="black", font_size=12)
    plotter.add_text("Minor Scarp", position=(0.2, 0.7), color="black", font_size=12)
    plotter.add_text("Main Body", position=(0.4, 0.5), color="black", font_size=12)
    plotter.add_text("Toe of Rupture", position=(0.6, 0.3), color="black", font_size=12)

    # Add color bar for failure magnitude
    plotter.add_scalar_bar(title="Failure Magnitude", n_labels=5)

    # Show plot
    plotter.show()
# End-of-file (EOF)
