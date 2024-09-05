"""Visaulization functions"""
import cupy as cp  # CuPy for GPU acceleration
import pyvista as pv
import numpy as np  # NumPy for handling some CPU-based operations

def plot_fixed_nodes(mesh, fixed_nodes, point_size=10, mesh_opacity=0.5):
    """Plots a finite element mesh and highlights the fixed nodes.

    Args:
        mesh (pyvista.DataSet): The finite element mesh object that contains the geometry 
        and connectivity information of the entire domain.
        fixed_nodes (array): An array or list of indices representing the fixed nodes in the mesh.
        point_size (int, optional): The size of the points used to highlight the fixed nodes. 
        Default is 10. Defaults to 10.
        mesh_opacity (float, optional): The opacity of the mesh. Value should be between 
        0 (completely transparent) and 1 (completely opaque). Defaults to 0.5.
    """
    # Initialize PyVista plotter with a specified window size
    plotter = pv.Plotter(window_size=(1000, 600))
    plotter.add_text("Fixed Nodes", font_size=12)

    # Plot the entire mesh with adjustable opacity
    plotter.add_mesh(mesh, color="white", show_edges=True, opacity=mesh_opacity)

    # Highlight the fixed nodes
    fixed_points = mesh.points[fixed_nodes]
    plotter.add_points(
        fixed_points, color="red", point_size=point_size, render_points_as_spheres=True
    )

    # Display the plot
    plotter.show()

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

    # Directly use a Boolean mask to identify failing elements
    failing_elements = fos <= threshold

    # Use the sum of the Boolean array to count the number of failing elements
    num_failing_elements = cp.count_nonzero(failing_elements)

    print(f"Number of failing elements identified: {num_failing_elements}")
    return failing_elements

def visualize_failure_analysis(
    mesh, failure_surfaces, stress_tensor, strain_tensor, fos
):
    """
    Visualize failure surfaces and stress/strain distributions.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list): List of extracted failure surfaces.
        stress_tensor (cp.ndarray): Stress tensor for each element.
        strain_tensor (cp.ndarray): Strain tensor for each element.
        fos (cp.ndarray): Factor of Safety for each element.
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

    # Convert CuPy arrays to NumPy for visualization in PyVista
    stress_tensor_np = cp.asnumpy(stress_tensor)
    strain_tensor_np = cp.asnumpy(strain_tensor)
    fos_np = cp.asnumpy(fos)

    # Add scalar bars and overlay stress, strain, and FoS on the mesh
    plotter.add_mesh(
        mesh,
        scalars=stress_tensor_np[:, 0],
        cmap="coolwarm",
        opacity=0.6,
        label="Stress Distribution",
    )
    plotter.add_scalar_bar("Stress", title="Stress (MPa)")

    plotter.add_mesh(
        mesh,
        scalars=strain_tensor_np[:, 0],
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

def visualize_mesh_and_failure_surfaces(
    mesh, failure_surfaces, mesh_color="white", mesh_opacity=0.5, surface_color="red"
):
    """
    Visualize the original mesh and the extracted failure surfaces with adjustable visual 
    properties.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list): List of extracted failure surfaces.
        mesh_color (str): Color of the mesh. Default is "white".
        mesh_opacity (float): Opacity of the mesh, ranging from 0 (transparent) to 1 (opaque). 
        Default is 0.5.
        surface_color (str): Color used for failure surfaces. Default is "red".
    """
    print("Visualizing mesh and failure surfaces...")

    plotter = pv.Plotter()

    # Visualize the original mesh with adjustable opacity and color
    plotter.add_mesh(
        mesh,
        color=mesh_color,
        opacity=mesh_opacity,
        show_edges=True,
        label="Original Mesh",
    )

    # Visualize each failure surface with a distinct color and customizable options
    for i, surface in enumerate(failure_surfaces):
        plotter.add_mesh(
            surface,
            color=surface_color,
            show_edges=True,
            label=f"Failure Surface {i+1}",
        )

    plotter.add_legend()
    plotter.show()

def visualize_stress_strain_fields(mesh, stress_tensor, strain_tensor):
    """
    Visualize stress and strain fields on the mesh, enhanced with performance 
    optimizations and error handling.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        stress_tensor (cp.ndarray): Stress tensor for each element. 
        Shape should be (n_elements, n_stress_components).
        strain_tensor (cp.ndarray): Strain tensor for each element. 
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

    plotter = pv.Plotter(shape=(1, 2))  # Two plots side by side

    # Efficiently convert tensors from CuPy to NumPy
    stress_magnitude = cp.linalg.norm(stress_tensor, axis=1).get()
    strain_magnitude = cp.linalg.norm(strain_tensor, axis=1).get()

    # Stress field visualization
    plotter.subplot(0, 0)
    plotter.add_text("Stress Field", font_size=12)
    plotter.add_mesh(mesh, scalars=stress_magnitude, cmap="coolwarm", show_edges=True)
    plotter.add_scalar_bar("Stress Magnitude", format="%.2e")

    # Strain field visualization
    plotter.subplot(0, 1)
    plotter.add_text("Strain Field", font_size=12)
    plotter.add_mesh(mesh, scalars=strain_magnitude, cmap="viridis", show_edges=True)
    plotter.add_scalar_bar("Strain Magnitude", format="%.2e")

    plotter.show()

def visualize_fos_distribution(mesh, fos, custom_range=None):
    """
    Visualize the Factor of Safety (FoS) distribution on the mesh with options for 
    customizing the color map range.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        fos (cp.ndarray): Factor of Safety for each element.
        custom_range (tuple, optional): Min and max values to define the color map scale. 
        Default is None.
    """
    print("Visualizing FoS distribution...")

    # Convert CuPy array to NumPy for PyVista compatibility
    fos_np = cp.asnumpy(fos)

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
    plotter.enable_zoom()
    plotter.enable_picking()

    plotter.show()

def visualize_connected_components(mesh, connected_components):
    """
    Enhanced visualization of connected components of failing elements on the mesh.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        connected_components (np.ndarray): Array of connected component labels.
    """
    print("Visualizing connected components of failing elements...")

    plotter = pv.Plotter()
    unique_labels = np.unique(connected_components)

    # Generate a list of colors for each component
    colors = pv.plotting.get_cmap(
        "viridis", len(unique_labels) - 1
    )  # Exclude the zero label

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
    plotter.enable_zoom()  # Allow users to zoom into specific components
    plotter.enable_picking()  # Allow picking of components to display more details
    plotter.show()

def visualize_shear_bands_and_slip_surfaces(
    mesh,
    shear_bands,
    slip_surfaces,
    mesh_color="white",
    mesh_opacity=0.1,
    shear_band_color="blue",
    slip_surface_color="orange",
):
    """
    Enhanced visualization of shear bands and slip surfaces on the mesh with customizable color and opacity settings.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        shear_bands (pv.UnstructuredGrid): Mesh subset representing shear bands.
        slip_surfaces (pv.UnstructuredGrid): Mesh subset representing slip surfaces.
        mesh_color (str): Color for the original mesh.
        mesh_opacity (float): Opacity for the original mesh.
        shear_band_color (str): Color for the shear bands.
        slip_surface_color (str): Color for the slip surfaces.
    """
    print("Visualizing shear bands and slip surfaces...")

    plotter = pv.Plotter()

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
    plotter.enable_zoom()
    plotter.enable_rotation()
    plotter.show()

def plot_resultant_directions_and_magnitudes(
    mesh, failure_surfaces, resultant_directions, resultant_magnitudes
):
    """
    Plot resultant directions and magnitudes on the mesh.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list): List of extracted failure surfaces.
        resultant_directions (list): List of resultant directions for each failure surface.
        resultant_magnitudes (list): Resultant magnitudes for each failure surface.
    """
    print("Plotting resultant directions and magnitudes...")

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, opacity=0.3, label="Mesh")

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

    # Add color bar for failure magnitude
    plotter.add_scalar_bar(title="Failure Magnitude", n_labels=5)

    # Show plot
    plotter.show()

def plot_original_mesh(mesh):
    """
    Plot the original mesh before any failure analysis.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh, show_edges=True, opacity=0.5, color="lightgrey", label="Original Mesh"
    )
    plotter.add_scalar_bar(title="Original Mesh")
    plotter.show()

def plot_failure_results(
    mesh, failure_surfaces, resultant_directions, resultant_magnitudes
):
    """
    Plot resultant directions and magnitudes on the mesh after failure analysis.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failure_surfaces (list): List of extracted failure surfaces.
        resultant_directions (list): List of resultant directions for each failure surface.
        resultant_magnitudes (list): Resultant magnitudes for each failure surface.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, opacity=0.3, label="Mesh")

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

    # Add color bar for failure magnitude
    plotter.add_scalar_bar(title="Failure Magnitude", n_labels=5)
    plotter.show()


# Function to highlight failing cells on the mesh
def plot_failing_cells(mesh, failing_elements):
    """
    Plot the failing cells on the mesh after failure analysis.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failing_elements (cp.ndarray): Indices of elements with FoS ≤ threshold.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh, show_edges=True, opacity=0.3, color="lightgrey", label="Mesh"
    )

    # Convert failing elements to NumPy array for plotting
    failing_elements_np = failing_elements.get()

    # Extract and highlight failing cells
    failing_cells = mesh.extract_cells(failing_elements_np)
    plotter.add_mesh(failing_cells, color="red", label="Failing Cells")

    # Show plot
    plotter.show()

def plot_failure_features(
    mesh, failing_elements, failure_surfaces, resultant_directions, resultant_magnitudes
):
    """
    Plot the mesh with failure features, similar to the provided image.

    Args:
        mesh (pv.UnstructuredGrid): The mesh of the model.
        failing_elements (cp.ndarray): Indices of elements with FoS ≤ threshold.
        failure_surfaces (list): List of extracted failure surfaces.
        resultant_directions (list): List of resultant directions for each failure surface.
        resultant_magnitudes (list): Resultant magnitudes for each failure surface.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, opacity=0.3, label="Mesh")

    # Highlight failing elements
    failing_elements_np = failing_elements.get()
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