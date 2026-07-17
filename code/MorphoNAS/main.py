import os
import numpy as np
from .morphogen_display import MorphogenDisplay
from .grid import Grid
from .neuron_graph_display import NeuronGraphDisplay
from .genome import Genome
import networkx as nx
import json
import matplotlib.pyplot as plt
import time
import random
from .genetic_algorithm import GeneticAlgorithm
import zlib
from .experiment_runner import ExperimentRunner
from scipy.sparse import lil_matrix
from .fitness_functions import TargetGraphFitnessFunction, StructuralGraphFitnessFunction, NetworkFitnessFunction, GymFitnessFunction
from .genome_strategies import AggressiveMutationStrategy, ExtendedMatrixMutationStrategy, BlockPreservationCrossoverStrategy
from PIL import Image


def _convert_png_to_grayscale(file_path):
    """
    Convert a PNG file to grayscale immediately after saving.
    
    Args:
        file_path (str): Path to the PNG file to convert
    """
    try:
        with Image.open(file_path) as img:
            # Convert to grayscale
            grayscale_img = img.convert('L')
            # Save back to the same file
            grayscale_img.save(file_path, 'PNG')
    except Exception as e:
        print(f"Warning: Could not convert {file_path} to grayscale: {str(e)}")


def generate_and_test_genome(genome, index):
    """Test a genome and generate its params file."""
    grid = Grid(genome)
    grid.add_cell((grid.size_x//2, grid.size_y//2), "progenitor")
    
    # Run simulation
    for _ in range(500):
        grid.step()
    grid.final_step()
    
    # Create graph for network analysis
    G = grid.get_graph();
    
    # Calculate network parameters
    nodes_no_incoming = [node for node in G.nodes() if G.in_degree(node) == 0]
    params = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_isolated": len(nodes_no_incoming),
        "density": float(nx.density(G)) if G.number_of_nodes() > 0 else 0.0,
        "avg_degree": float(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0),
        "is_strongly_connected": nx.is_strongly_connected(G) if G.number_of_nodes() > 0 else False,
        "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False
    }
    
    # Save genome and params
    base_path = os.path.join(os.path.dirname(__file__), '..', 'tests')
    genome_path = os.path.join(base_path, 'fixtures', f'{index:02d}_genome.json')
    params_path = os.path.join(base_path, 'references', f'{index:02d}_params.json')
    
    genome.to_json(filepath=genome_path)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

def generate_new_genome_variations():
    """Generate new genome variations with random patterns."""
    for i in range(1, 31):  # 1 through 30
        # Create genome with varying parameters
        # Randomly choose dimensions between 2 and 5 for each pattern
        patterns = [
            np.random.rand(np.random.randint(2, 6), np.random.randint(2, 6))
            for _ in range(5)
        ]
        
        fixture_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures', f'genome_5_morphogens_stub.json')
        genome = Genome.from_json(filepath=fixture_path)
        
        # Create list of normalized patterns directly
        genome.diffusion_patterns = [pattern / pattern.sum() for pattern in patterns]
        
        generate_and_test_genome(genome, i)

def generate_genome_references():
    """Generate parameter references for all existing genome files."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    
    for filename in sorted(os.listdir(fixtures_dir)):
        if filename.endswith('_genome.json'):
            index = int(filename.split('_')[0])
            genome_path = os.path.join(fixtures_dir, filename)
            genome = Genome.from_json(filepath=genome_path)
            generate_and_test_genome(genome, index)

def run_simulation(genome=None, verbose=True, display_weights=False, save_displays=False, save_dir=None, use_kamada_kawai=False, display_node_numbers=True, capture_step=None, capture_steps=None, max_morphogen_cols=3, max_neuron_cols=3, dpi=100, morphogen_display_scale=1.0, neuron_display_scale=1.0, morphogen_multi_step_display_scale=1.0, neuron_multi_step_display_scale=1.0, grayscale=False, font_size=12):
    if genome is None:
        #fixture_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures', '01_genome.json')
        # 80 / 480
        #fixture_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', '01_nodes_edges_match', 'results', 'basic_experiment_03a', 'best_genome.json')
        # 8 / 20
        fixture_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', '01_nodes_edges_match', 'results', 'basic_experiment_02', 'best_genome.json')
        #fixture_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', '02_graph_comparison', 'results', 'graph_comparison_01A_structural', 'best_genome.json')
        #fixture_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', '02_graph_comparison', 'results', 'graph_comparison_04', 'best_genome.json')
        genome = Genome.from_json(filepath=fixture_path)
        genome_dir = os.path.dirname(fixture_path)
    else:
        # If genome is provided directly, use provided save_dir or current directory
        genome_dir = save_dir if save_dir is not None else os.getcwd()
    
    grid = Grid(genome)
    
    if capture_step is not None:
        # Run simulation up to the specified step for morphogen display
        grid.add_cell((grid.size_x//2, grid.size_y//2), "progenitor")
        
        # Run simulation up to the specified step
        for i in range(min(capture_step, grid.max_growth_steps)):
            grid.step()
            if verbose and i % 100 == 0:
                print(f"Step {i}; cells: {grid.cell_count()}")
        
        # Capture displays at the specified step
        if save_displays:
            # Use non-interactive backend for saving
            with plt.style.context('default'), plt.rc_context({'backend': 'Agg'}):
                # Save morphogen display at the specified step
                morphogen_display = MorphogenDisplay(grid, scale=morphogen_display_scale, font_size=font_size)
                morphogen_display.on_step()
                step_suffix = f"_step_{capture_step}" if capture_step is not None else ""
                morphogen_file = os.path.join(genome_dir, f'morphogen_display{step_suffix}.png')
                plt.savefig(morphogen_file, dpi=dpi, bbox_inches=None)
                plt.close(morphogen_display.fig)
                if grayscale:
                    _convert_png_to_grayscale(morphogen_file)
                
                # Save neuron graph display at the specified step
                neuron_display = NeuronGraphDisplay(grid, use_kamada_kawai=use_kamada_kawai, display_node_numbers=display_node_numbers, scale=neuron_display_scale, font_size=font_size)
                neuron_display.on_step()
                neuron_file = os.path.join(genome_dir, f'neuron_graph{step_suffix}.png')
                plt.savefig(neuron_file, dpi=dpi, bbox_inches=None)
                plt.close(neuron_display.fig)
                if grayscale:
                    _convert_png_to_grayscale(neuron_file)
        
        # Continue with the full simulation for final neuron graph
        for i in range(capture_step, grid.max_growth_steps):
            grid.step()
            if verbose and i % 100 == 0:
                print(f"Step {i}; cells: {grid.cell_count()}")
        
        grid.final_step()
        
        if save_displays:
            # Use non-interactive backend for saving
            with plt.style.context('default'), plt.rc_context({'backend': 'Agg'}):
                # Save neuron graph display (final state)
                neuron_display = NeuronGraphDisplay(grid, use_kamada_kawai=use_kamada_kawai, display_node_numbers=display_node_numbers, scale=neuron_display_scale, font_size=font_size)
                neuron_display.on_step()
                neuron_file = os.path.join(genome_dir, 'neuron_graph.png')
                plt.savefig(neuron_file, dpi=dpi, bbox_inches=None)
                plt.close(neuron_display.fig)
                if grayscale:
                    _convert_png_to_grayscale(neuron_file)
    
    elif capture_steps is not None:
        # Special mode: capture multiple steps and display in grid
        grid.add_cell((grid.size_x//2, grid.size_y//2), "progenitor")
        
        # Sort and validate steps
        capture_steps = sorted(capture_steps)
        max_step = max(capture_steps)
        
        # Run simulation up to the maximum step
        for i in range(min(max_step, grid.max_growth_steps)):
            grid.step()
            if verbose and i % 100 == 0:
                print(f"Step {i}; cells: {grid.cell_count()}")
        
        # Continue to completion for final neuron graph
        for i in range(max_step, grid.max_growth_steps):
            grid.step()
            if verbose and i % 100 == 0:
                print(f"Step {i}; cells: {grid.cell_count()}")
        
        grid.final_step()
        
        if save_displays:
            # Use non-interactive backend for saving
            with plt.style.context('default'), plt.rc_context({'backend': 'Agg'}):
                # Create multi-step morphogen display
                _create_multi_step_morphogen_display(grid, capture_steps, genome_dir, max_morphogen_cols, dpi, morphogen_multi_step_display_scale, grayscale, font_size)
                
                # Create multi-step neuron graph display
                _create_multi_step_neuron_display(grid, capture_steps, genome_dir, use_kamada_kawai, display_node_numbers, max_neuron_cols, dpi, neuron_multi_step_display_scale, grayscale, font_size)
    
    else:
        # Run simulation normally (original behavior)
        grid = grid.run_simulation(verbose=verbose, display_weights=display_weights)
    
    if save_displays:
        # Use non-interactive backend for saving
        with plt.style.context('default'), plt.rc_context({'backend': 'Agg'}):
                # Save morphogen display (final state)
            morphogen_display = MorphogenDisplay(grid, scale=morphogen_display_scale, font_size=font_size)
            morphogen_display.on_step()
            morphogen_file = os.path.join(genome_dir, 'morphogen_display.png')
            plt.savefig(morphogen_file, dpi=dpi, bbox_inches=None)
            plt.close(morphogen_display.fig)
            if grayscale:
                _convert_png_to_grayscale(morphogen_file)
            
            # Save neuron graph display (final state)
            neuron_display = NeuronGraphDisplay(grid, use_kamada_kawai=use_kamada_kawai, display_node_numbers=display_node_numbers, scale=neuron_display_scale, font_size=font_size)
            neuron_display.on_step()
            neuron_file = os.path.join(genome_dir, 'neuron_graph.png')
            plt.savefig(neuron_file, dpi=dpi, bbox_inches=None)
            plt.close(neuron_display.fig)
            if grayscale:
                _convert_png_to_grayscale(neuron_file)
    
    return grid


def _create_multi_step_morphogen_display(grid, capture_steps, save_dir, max_cols=5, dpi=150, scale=1.0, grayscale=False, font_size=12):
    """Create a multi-step morphogen display using the same display as single step."""
    num_steps = len(capture_steps)
    
    # Calculate grid layout with max columns constraint
    cols = min(num_steps, max_cols)
    rows = (num_steps + cols - 1) // cols  # Ceiling division
    
    # Create figure with larger size for morphogen displays
    # Scale width to maintain individual display size regardless of grid layout
    individual_width = 8  # Width per individual display
    fig, axes = plt.subplots(rows, cols, figsize=(individual_width*cols*scale, 5*rows*scale))
    if num_steps == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Create a copy of the grid to replay simulation
    genome = Genome(
        max_growth_steps=grid.max_growth_steps,
        size_x=grid.size_x,
        size_y=grid.size_y,
        diffusion_rate=grid.diffusion_rate,
        num_morphogens=grid.num_morphogens,
        division_threshold=grid.division_threshold,
        cell_differentiation_threshold=grid.cell_differentiation_threshold,
        axon_growth_threshold=grid.axon_growth_threshold,
        max_axon_length=grid.max_axon_length,
        axon_connect_threshold=grid.axon_connect_threshold,
        self_connect_isolated_neurons_fraction=grid.self_connect_isolated_neurons_fraction,
        weight_adjustment_target=grid.weight_adjustment_target,
        weight_adjustment_rate=grid.weight_adjustment_rate,
        progenitor_secretion_rates=grid.progenitor_secretion_rates,
        neuron_secretion_rates=grid.neuron_secretion_rates,
        inhibition_matrix=grid.inhibition_matrix,
        diffusion_patterns=grid.diffusion_patterns
    )
    temp_grid = Grid(genome)
    temp_grid.add_cell((temp_grid.size_x//2, temp_grid.size_y//2), "progenitor")
    
    step_idx = 0
    for i in range(max(capture_steps) + 1):
        if i in capture_steps:
            # Capture morphogen display for this step using the same display class
            ax = axes[step_idx]
            
            # Create the morphogen display directly without the complex layout
            # Get RGB data for this step
            rgb_data = np.ones((temp_grid.size_x, temp_grid.size_y, 3))  # Start with white
            for j in range(min(3, temp_grid.num_morphogens)):
                rgb_data[:, :, j] -= np.clip(temp_grid.get_morphogen_array(j), 0, 1)
            
            # Display the morphogen lattice
            ax.imshow(rgb_data, vmin=0, vmax=1)
            
            # Set title with just step number
            ax.set_title(f'Step {i}', fontsize=int(font_size * scale))
            
            # Scale tick parameters (keep numeric labels, remove axis labels)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='both', 
                          labelsize=int(font_size / 12 * 10 * scale), width=1.0 * scale, length=3.0 * scale)
            
            # Scale the plot frame (spines)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0 * scale)
            
            # Draw cell borders and axons
            _draw_cell_borders_and_axons(ax, temp_grid, rgb_data, scale)
            
            step_idx += 1
        
        if i < max(capture_steps):
            temp_grid.step()
    
    # Hide unused subplots
    for i in range(num_steps, len(axes)):
        axes[i].axis('off')
    
    # Make subplots as close as possible
    plt.subplots_adjust(wspace=-0.6, left=0.01, right=0.99, top=0.95, bottom=0.05)
    morphogen_file = os.path.join(save_dir, 'morphogen_display_multi_steps.png')
    plt.savefig(morphogen_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    if grayscale:
        _convert_png_to_grayscale(morphogen_file)


def _draw_cell_borders_and_axons(ax, grid, img_data, scale=1.0):
    """Helper function to draw cell borders and axons on a given axis."""
    import matplotlib.patches as patches
    import matplotlib.path as mpath
    
    # Draw neuron connections first (thicker solid lines)
    source_indices, target_indices = grid.neuron_connections.nonzero()
    neuron_ids = grid.get_neuron_ids()
    for source_id, target_id in zip(source_indices + 1, target_indices + 1):
        if source_id in neuron_ids and target_id in neuron_ids:
            source_pos = grid.get_cell_position(source_id)
            target_pos = grid.get_cell_position(target_id)
            
            # Get the RGB color at source cell's position for the connection color
            source_color = img_data[source_pos]
            connection_color = 1 - source_color  # Invert RGB values
            connection_color = np.clip(connection_color, 0, 1)

            # Calculate shortest path considering wrapping
            dx = target_pos[1] - source_pos[1]
            dy = target_pos[0] - source_pos[0]
            
            # Adjust for wrapping
            if dx > grid.size_x/2:
                dx -= grid.size_x
            elif dx < -grid.size_x/2:
                dx += grid.size_x
                
            if dy > grid.size_y/2:
                dy -= grid.size_y
            elif dy < -grid.size_y/2:
                dy += grid.size_y

            # Calculate the endpoint considering the wrapping
            end_x = source_pos[1] + dx
            end_y = source_pos[0] + dy

            # Draw connection line using the calculated shortest path
            path_vertices = [
                (source_pos[1], source_pos[0]),  # Start point
                (end_x, end_y)  # End point using wrapped coordinates
            ]
            path = mpath.Path(path_vertices, [mpath.Path.MOVETO, mpath.Path.LINETO])
            patch = patches.PathPatch(path, facecolor='none', edgecolor=connection_color, 
                                    linewidth=2.0 * scale, alpha=0.9)
            ax.add_patch(patch)

    # Draw cells and growing axons
    for cell_id in grid.get_cell_ids():
        x, y = grid.get_cell_position(cell_id)

        # Get the RGB color at this cell's position
        cell_color = img_data[x, y]
        border_color = 1 - cell_color  # Invert RGB values
        border_color = np.clip(border_color, 0, 1)

        # Set linestyle based on cell type
        linestyle = ':' if grid.is_progenitor(cell_id) else '-'

        # Add cell border
        rect = patches.Rectangle(
            (y - 0.5, x - 0.5), 1, 1,
            linewidth=1.5 * scale,
            edgecolor=border_color,
            facecolor='none',
            linestyle=linestyle
        )
        ax.add_patch(rect)

        # Draw axon if this is a neuron with a growing axon (not yet connected)
        if (grid.is_neuron(cell_id) and 
            len(grid.get_axon(cell_id)) > 1 and 
            grid.neuron_connections[cell_id - 1].nnz == 0):  # Check if neuron has no connections
            _draw_axon_on_axis(ax, grid.get_axon(cell_id), border_color, grid, scale)


def _draw_axon_on_axis(ax, axon_points, color, grid, scale=1.0):
    """Draw an axon as a dotted curve on a given axis."""
    import matplotlib.patches as patches
    import matplotlib.path as mpath
    
    if len(axon_points) < 2:
        return

    # Convert axon points to display coordinates (center of cells)
    points = []
    for i in range(len(axon_points)):
        x, y = axon_points[i]
        
        if i > 0:
            # Get previous point
            prev_x, prev_y = axon_points[i-1]
            
            # Calculate differences considering wrapping
            dx = x - prev_x
            dy = y - prev_y
            
            # Adjust for wrapping
            if dx > grid.size_x/2:
                dx -= grid.size_x
            elif dx < -grid.size_x/2:
                dx += grid.size_x
                
            if dy > grid.size_y/2:
                dy -= grid.size_y
            elif dy < -grid.size_y/2:
                dy += grid.size_y
                
            # Add to the accumulated position
            last_x, last_y = points[-1]
            points.append((last_x + dx, last_y + dy))
        else:
            # First point - swap x,y for matplotlib's coordinate system
            points.append((y, x))
    
    # Create a smooth path through the points
    path_vertices = []
    codes = []
    
    # Start point
    path_vertices.append(points[0])
    codes.append(mpath.Path.MOVETO)
    
    # Add curved segments between points
    for i in range(1, len(points)):
        # Current point
        curr = np.array(points[i])
        prev = np.array(points[i-1])
        
        # Calculate control points for the curve
        dist = curr - prev
        ctrl1 = prev + dist * 0.3
        ctrl2 = curr - dist * 0.3
        
        path_vertices.extend([ctrl1, ctrl2, curr])
        codes.extend([mpath.Path.CURVE4] * 3)
    
    # Create and add the path
    path = mpath.Path(path_vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, 
                            linewidth=1.5 * scale, alpha=1, linestyle=':')  # Dotted line for growing axons
    ax.add_patch(patch)


def _create_multi_step_neuron_display(grid, capture_steps, save_dir, use_kamada_kawai, display_node_numbers, max_cols=4, dpi=150, scale=1.0, grayscale=False, font_size=12):
    """Create a multi-step neuron graph display with frames like single-step display."""
    num_steps = len(capture_steps)
    
    # Calculate grid layout with max columns constraint
    cols = min(num_steps, max_cols)
    rows = (num_steps + cols - 1) // cols  # Ceiling division
    
    # Create figure with larger size for neuron displays
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols*scale, 5*rows*scale))
    if num_steps == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Remove axis labels and tick labels from all subplots, scale spines
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
        # Scale the plot frame (spines)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0 * scale)
    
    # Create a copy of the grid to replay simulation
    genome = Genome(
        max_growth_steps=grid.max_growth_steps,
        size_x=grid.size_x,
        size_y=grid.size_y,
        diffusion_rate=grid.diffusion_rate,
        num_morphogens=grid.num_morphogens,
        division_threshold=grid.division_threshold,
        cell_differentiation_threshold=grid.cell_differentiation_threshold,
        axon_growth_threshold=grid.axon_growth_threshold,
        max_axon_length=grid.max_axon_length,
        axon_connect_threshold=grid.axon_connect_threshold,
        self_connect_isolated_neurons_fraction=grid.self_connect_isolated_neurons_fraction,
        weight_adjustment_target=grid.weight_adjustment_target,
        weight_adjustment_rate=grid.weight_adjustment_rate,
        progenitor_secretion_rates=grid.progenitor_secretion_rates,
        neuron_secretion_rates=grid.neuron_secretion_rates,
        inhibition_matrix=grid.inhibition_matrix,
        diffusion_patterns=grid.diffusion_patterns
    )
    temp_grid = Grid(genome)
    temp_grid.add_cell((temp_grid.size_x//2, temp_grid.size_y//2), "progenitor")
    
    step_idx = 0
    for i in range(max(capture_steps) + 1):
        if i in capture_steps:
            # Capture neuron graph for this step
            ax = axes[step_idx]
            
            # Create graph for this step
            G = temp_grid.get_graph()
            
            if G.number_of_nodes() > 0:
                # Calculate layout based on preference
                if use_kamada_kawai:
                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except nx.NetworkXError:
                        # Fall back to spring layout if Kamada-Kawai fails
                        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
                else:
                    # Use default spring layout
                    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
                
                # Determine nodes with no incoming connections
                nodes_no_incoming = [node for node in G.nodes() if G.in_degree(node) == 0]
                node_colors = ['yellow' if node in nodes_no_incoming else 'lightblue' for node in G.nodes()]
                
                # Calculate node size based on figure coordinate system
                # Base node size that works well for scale=1.0
                base_node_size = 1200
                # Scale node size by the square of the scale factor to maintain proportional area
                scaled_node_size = int(base_node_size * (scale ** 2))
                
                # Draw the graph
                nx.draw_networkx_nodes(G, pos, 
                                      node_color=node_colors,
                                      node_size=scaled_node_size,
                                      alpha=0.6,
                                      ax=ax)
                
                nx.draw_networkx_edges(G, pos,
                                      edge_color='gray',
                                      arrows=True,
                                      arrowsize=int(30 * scale),  # Scale arrow size
                                      width=1.0 * scale,  # Scale edge width
                                      connectionstyle="arc3,rad=0.1",  # Slight curve to avoid node overlap
                                      node_size=scaled_node_size,  # Pass node size to connection style
                                      ax=ax)
                
                # Always show node numbers inside the nodes
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=int(font_size / 12 * 10 * scale), font_weight='bold')
                
                # Calculate network statistics
                num_isolated = len(nodes_no_incoming)
                num_nodes = G.number_of_nodes()
                num_edges = G.number_of_edges()
                density = nx.density(G)
                avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
                
                # Check connectivity with safety checks for empty graphs
                is_strongly_connected = nx.is_strongly_connected(G) if num_nodes > 0 else False
                is_weakly_connected = nx.is_weakly_connected(G) if num_nodes > 0 else False
                
                # Update title with just neurons and connections count
                ax.set_title(f'Step {i} (V={num_nodes}; E={num_edges})', fontsize=int(font_size * scale))
                
                # Set axis limits with padding based on actual node positions
                pos_array = np.array(list(pos.values()))
                x_min, y_min = pos_array.min(axis=0) - 0.2
                x_max, y_max = pos_array.max(axis=0) + 0.2
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            else:
                # Empty graph
                ax.text(0.5, 0.5, 'No neurons', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=int(font_size * scale))
                ax.set_title(f'Step {i} (No neurons)', fontsize=int(font_size * scale))
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
            
            step_idx += 1
        
        if i < max(capture_steps):
            temp_grid.step()
    
    # Hide unused subplots
    for i in range(num_steps, len(axes)):
        axes[i].axis('off')
    
    # Add some gap between subplots
    plt.subplots_adjust(wspace=0.2, left=0.01, right=0.99, top=0.9, bottom=0.1)
    neuron_file = os.path.join(save_dir, 'neuron_graph_multi_steps.png')
    plt.savefig(neuron_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    if grayscale:
        _convert_png_to_grayscale(neuron_file)

def save_all_experiment_displays(capture_step=None):
    """Save displays for all experiments that have a best_genome.json file."""
    experiments_dir = os.path.join('experiments')
    
    # Track successful and failed saves
    successful = []
    failed = []
    
    # Process each experiment directory
    for experiment_dir in os.listdir(experiments_dir):
        experiment_path = os.path.join(experiments_dir, experiment_dir)
        if not os.path.isdir(experiment_path):
            continue
            
        # Look for results directory
        results_dir = os.path.join(experiment_path, 'results')
        if not os.path.exists(results_dir):
            continue
            
        # Process each subdirectory in results
        for result_dir in os.listdir(results_dir):
            result_path = os.path.join(results_dir, result_dir)
            if not os.path.isdir(result_path):
                continue
                
            # Check for best_genome.json
            best_genome_path = os.path.join(result_path, 'best_genome.json')
            if not os.path.exists(best_genome_path):
                continue
                
            print(f"\nProcessing {experiment_dir}/{result_dir}")
            try:
                # Load genome and run simulation
                genome = Genome.from_json(filepath=best_genome_path)
                run_simulation(genome=genome, verbose=False, save_displays=True, save_dir=result_path, use_kamada_kawai=True, display_node_numbers=False, capture_step=capture_step)
                successful.append(f"{experiment_dir}/{result_dir}")
            except Exception as e:
                print(f"Error processing {experiment_dir}/{result_dir}:")
                print(f"Error: {str(e)}")
                failed.append((f"{experiment_dir}/{result_dir}", str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("Display Save Summary")
    print("="*80)
    print(f"\nSuccessful saves ({len(successful)}):")
    for exp in successful:
        print(f"  ✓ {exp}")
    
    if failed:
        print(f"\nFailed saves ({len(failed)}):")
        for exp, error in failed:
            print(f"  ✗ {exp}")
            print(f"    Error: {error}")

def generate_display_references():
    """Generate reference images for display testing."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create references directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    
    # Set up consistent rendering settings
    plt.ioff()
    
    # Load all genome files
    for filename in sorted(os.listdir(fixtures_dir)):
        if filename.endswith('_genome.json'):
            case_name = filename.replace('_genome.json', '')
            genome_path = os.path.join(fixtures_dir, filename)
            
            print(f"Generating reference images for {case_name}")
            
            # Initialize simulation
            genome = Genome.from_json(filepath=genome_path)
            grid = Grid(genome)
            grid.add_cell((grid.size_x//2, grid.size_y//2), "progenitor")
            
            # Run simulation
            for _ in range(500):
                grid.step()
            grid.final_step()
            
            # Use Agg backend and consistent style for reference generation
            with plt.style.context('default'), plt.rc_context({'backend': 'Agg'}):
                # Generate morphogen display reference
                morphogen_fig = plt.figure(figsize=(10, 5))
                display = MorphogenDisplay(grid)
                display.on_step()
                plt.savefig(os.path.join(references_dir, f'{case_name}_morphogen.png'), 
                           dpi=100, bbox_inches=None)  # Remove tight bbox
                plt.close(morphogen_fig)
                
                # Generate neuron graph reference
                neuron_fig = plt.figure(figsize=(8, 8))
                neuron_display = NeuronGraphDisplay(grid)
                neuron_display.on_step()
                plt.savefig(os.path.join(references_dir, f'{case_name}_neuron.png'), 
                           dpi=100, bbox_inches=None)  # Remove tight bbox
                plt.close(neuron_fig)

def run_random_simulation():
    """Run simulation with a random genome."""
    print("Running random simulation")
    
    # Create seeded RNG for reproducibility
    rng = random.Random()
    
    # Generate random genome
    genome = Genome.random(rng)
    
    # Initialize grid and add starting cell
    grid = Grid(genome)
    grid.add_cell((grid.size_x//2, grid.size_y//2), "progenitor")
    
    # Create displays
    morphogen_display = MorphogenDisplay(grid, update_frequency=10)
    
    # Run simulation with displays updating each step
    start_time = time.time()
    for i in range(200):
        grid.step()
        if i % 100 == 0:
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            print(f"Step {i}; cells: {grid.cell_count()}; elapsed: {elapsed_ms:.2f} ms")
    grid.final_step()
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"Simulation completed in {elapsed_ms:.2f} ms")

    # Print final statistics
    print(f"Number of neurons connected to themselves: {(grid.neuron_connections.diagonal() > 0).sum()}")
    print(f"Total neuron connections: {grid.neuron_connections.sum()}")
    print(f"Final cell count: {grid.cell_count()}")
    print(f"Final neuron count: {grid.neuron_count()}")

    # Show the final state and keep displays open until user closes them
    morphogen_display.show(block=False)
    neuron_display = NeuronGraphDisplay(grid)
    neuron_display.show(block=True)

def generate_mutation_references():
    """Generate reference mutations for existing genome files."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create references directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    
    # Use the exact same seed as in tests
    rng = np.random.Generator(np.random.PCG64(42))  # Explicit RNG initialization
    
    for filename in sorted(os.listdir(fixtures_dir)):
        if filename.endswith('_genome.json'):
            case_name = filename.replace('_genome.json', '')
            genome_path = os.path.join(fixtures_dir, filename)
            
            # Load original genome
            genome = Genome.from_json(filepath=genome_path)
            
            # Generate reference mutation
            mutated_genome = genome.mutate(rng)
            
            # Save mutation reference
            reference_path = os.path.join(references_dir, f'{case_name}_genome_mutated.json')
            mutated_genome.to_json(filepath=reference_path)
            print(f"Generated mutation reference for {case_name}")

def generate_aggressive_mutation_references():
    """Generate reference mutations for existing genome files using aggressive mutation strategy."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create references directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    
    # Use the exact same seed as in tests
    rng = np.random.Generator(np.random.PCG64(42))  # Explicit RNG initialization
    
    for filename in sorted(os.listdir(fixtures_dir)):
        if filename.endswith('_genome.json'):
            case_name = filename.replace('_genome.json', '')
            genome_path = os.path.join(fixtures_dir, filename)
            
            # Load original genome
            genome = Genome.from_json(filepath=genome_path, meta_parameters_strategy=ExtendedMatrixMutationStrategy(), mutation_strategy=AggressiveMutationStrategy())
            
            # Generate reference mutation
            mutated_genome = genome.mutate(rng)
            
            # Save mutation reference   
            reference_path = os.path.join(references_dir, f'{case_name}_genome_mutated_aggressive.json')
            mutated_genome.to_json(filepath=reference_path)
            print(f"Generated aggressive mutation reference for {case_name}")

def generate_crossover_references():
    """Generate reference crossovers between consecutive genome files."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create references directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    
    # Use the exact same seed as in tests
    rng = np.random.Generator(np.random.PCG64(42))  # Explicit RNG initialization
    
    # Get sorted list of genome files
    genome_files = sorted([f for f in os.listdir(fixtures_dir) if f.endswith('_genome.json')])
    
    # Process pairs of consecutive genomes
    for i in range(len(genome_files) - 1):
        parent1_file = genome_files[i]
        parent2_file = genome_files[i + 1]
        
        # Extract case numbers from filenames
        case1 = parent1_file.split('_')[0]
        case2 = parent2_file.split('_')[0]
        
        # Load parent genomes
        parent1 = Genome.from_json(filepath=os.path.join(fixtures_dir, parent1_file))
        parent2 = Genome.from_json(filepath=os.path.join(fixtures_dir, parent2_file))
        
        # Skip if parents have different numbers of morphogens
        if parent1.num_morphogens != parent2.num_morphogens:
            print(f"Skipping crossover between {case1} and {case2} due to different morphogen counts")
            continue
        
        # Generate reference crossover
        child_genome = Genome.crossover(parent1, parent2, rng)
        
        # Save crossover reference
        reference_path = os.path.join(references_dir, f'{case1}_{case2}_genome_crossover.json')
        child_genome.to_json(filepath=reference_path)
        print(f"Generated crossover reference between {case1} and {case2}")

def generate_block_preservation_crossover_references():
    """Generate reference crossovers between original genomes and randomly created twins."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create references directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    
    # Use the exact same seed as in tests for crossover
    crossover_rng = np.random.Generator(np.random.PCG64(42))
    
    # Create a separate RNG for generating twin genomes
    twin_rng = np.random.Generator(np.random.PCG64(123))  # Different seed
    
    # Get the list of all genome files in fixtures
    all_genome_files = sorted([f for f in os.listdir(fixtures_dir) if f.endswith('_genome.json')])
    
    # Filter out XXa_genome.json files - only process original XX_genome.json files
    original_genome_files = [f for f in all_genome_files if not f.split('_')[0].endswith('a')]
    
    # Process each original genome file
    for genome_file in original_genome_files:
        # Extract case number from filename
        case = genome_file.split('_')[0]
        case_num = int(case)
        
        # Load original genome with block preservation strategy
        genome_path = os.path.join(fixtures_dir, genome_file)
        original_genome = Genome.from_json(
            filepath=genome_path, 
            crossover_strategy=BlockPreservationCrossoverStrategy()
        )
        
        # Determine morphogen count for twin
        # For first 20 genomes, keep the same number of morphogens
        # For the rest, randomize between 3 and 7
        if case_num <= 20:
            num_morphogens = original_genome.num_morphogens
        else:
            num_morphogens = twin_rng.integers(3, 8)  # Random between 3 and 7 inclusive
        
        # Generate random grid sizes and max_growth_steps
        random_size_x = int(twin_rng.integers(10, 40))
        random_size_y = int(twin_rng.integers(10, 40))
        random_max_growth_steps = int(twin_rng.integers(50, 300))
        
        # Create a random twin genome with desired morphogen count and random sizes
        twin_genome = Genome.random(
            rng=twin_rng,
            num_morphogens=num_morphogens,
            size_x=random_size_x,
            size_y=random_size_y,
            max_growth_steps=random_max_growth_steps,
            crossover_strategy=BlockPreservationCrossoverStrategy()
        )
        
        # Save the twin genome (will overwrite any existing XXa file)
        twin_file = f"{case}a_genome.json"
        twin_path = os.path.join(fixtures_dir, twin_file)
        twin_genome.to_json(filepath=twin_path)
        
        # Generate crossover between original and twin
        child_genome = Genome.crossover(original_genome, twin_genome, crossover_rng)
        
        # Save crossover reference
        reference_path = os.path.join(references_dir, f"{case}_{case}a_genome_block_crossover.json")
        child_genome.to_json(filepath=reference_path)
        
        print(f"Generated block crossover reference between {case} and {case}a:")
        print(f"  Original: {original_genome.num_morphogens} morphogens, size {original_genome.size_x}x{original_genome.size_y}, {original_genome.max_growth_steps} steps")
        print(f"  Twin: {twin_genome.num_morphogens} morphogens, size {twin_genome.size_x}x{twin_genome.size_y}, {twin_genome.max_growth_steps} steps")

def generate_different_morphogen_crossover_references():
    """Generate reference crossovers between genomes with different morphogen counts."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create references directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    
    # Use the exact same seed as in tests
    rng = np.random.Generator(np.random.PCG64(42))  # Explicit RNG initialization
    
    # Load specific genomes
    parent1 = Genome.from_json(filepath=os.path.join(fixtures_dir, '20_genome.json'))
    parent2 = Genome.from_json(filepath=os.path.join(fixtures_dir, '21_genome.json'))
    
    # Generate multiple crossover references to capture different random outcomes
    for i in range(5):  # Generate 5 different crossover examples
        # Generate reference crossover
        child_genome = Genome.crossover(parent1, parent2, rng)
        
        # Save crossover reference
        reference_path = os.path.join(references_dir, f'20_21_genome_crossover_diff_morphogens_{i+1}.json')
        child_genome.to_json(filepath=reference_path)
        print(f"Generated crossover reference {i+1} between genomes 20 and 21")

def generate_target_graph_fitness_references():
    """Generate reference fitness values for target graph comparisons."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create references directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    
    # Define test cases with target graphs and sample networks
    test_cases = [
        {
            "name": "simple_chain",
            "fitness_targets": {
                "adjacency_list": {
                    "1": [2],
                    "2": [3],
                    "3": [4],
                    "4": []
                },
                "node_count_tolerance": 2,
                "topology_weight": 0.7,
                "size_weight": 0.3
            },
            "test_networks": [
                # Perfect match
                {
                    "adjacency_list": {
                        "1": [2],
                        "2": [3],
                        "3": [4],
                        "4": []
                    }
                },
                # Extra node
                {
                    "adjacency_list": {
                        "1": [2],
                        "2": [3],
                        "3": [4],
                        "4": [5],
                        "5": []
                    }
                },
                # Missing connection
                {
                    "adjacency_list": {
                        "1": [2],
                        "2": [3],
                        "3": [],
                        "4": []
                    }
                }
            ]
        },
        {
            "name": "small_cycle",
            "fitness_targets": {
                "adjacency_list": {
                    "1": [2],
                    "2": [3],
                    "3": [1]
                },
                "node_count_tolerance": 1,
                "topology_weight": 0.8,
                "size_weight": 0.2
            },
            "test_networks": [
                # Perfect match
                {
                    "adjacency_list": {
                        "1": [2],
                        "2": [3],
                        "3": [1]
                    }
                },
                # Reversed cycle
                {
                    "adjacency_list": {
                        "1": [3],
                        "3": [2],
                        "2": [1]
                    }
                }
            ]
        }
    ]
    
    # Create a mock Grid class for testing
    class MockGrid:
        def __init__(self, adjacency_list):
            # Find max node ID
            max_node = max(int(node) for node in adjacency_list.keys())
            for targets in adjacency_list.values():
                if targets:
                    max_node = max(max_node, max(targets))
            
            # Create connectivity matrix
            self.neuron_connections = lil_matrix((max_node, max_node), dtype=float)
            
            # Fill connectivity matrix
            for source, targets in adjacency_list.items():
                source_idx = int(source) - 1
                for target in targets:
                    target_idx = int(target) - 1
                    self.neuron_connections[source_idx, target_idx] = 1.0
        
        def get_neuron_ids(self):
            source_nodes, target_nodes = self.neuron_connections.nonzero()
            all_nodes = np.unique(np.concatenate([source_nodes, target_nodes]))
            return [node + 1 for node in all_nodes]
    
    # Process each test case
    for case in test_cases:
        case_name = case["name"]
        
        # Save target configuration
        target_path = os.path.join(fixtures_dir, f'target_graph_{case_name}.json')
        with open(target_path, 'w') as f:
            json.dump(case["fitness_targets"], f, indent=4)
        
        # Create fitness function
        fitness_fn = TargetGraphFitnessFunction(case["fitness_targets"])
        
        # Calculate fitness for each test network
        for network in case["test_networks"]:
            # Create mock grid
            grid = MockGrid(network["adjacency_list"])
            
            # Calculate and store fitness
            network["expected_fitness"] = float(fitness_fn.evaluate(grid))
        
        # Save reference results
        reference_path = os.path.join(references_dir, f'target_graph_{case_name}_results.json')
        with open(reference_path, 'w') as f:
            json.dump(case["test_networks"], f, indent=4)
        
        print(f"Generated target graph fixtures for {case_name}")

def display_target_graph_from_config(config_path):
    """Display the target graph from a configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    target_graph = config['fitness_targets']['adjacency_list']
    target_fig = NeuronGraphDisplay.display_target_graph(target_graph)
    target_fig.show()

def verify_experiment(config_path):
    """
    Verify the results of an experiment by evaluating its best genome.
    
    Args:
        config_path (str): Path to the experiment configuration file
    """
    # Load experiment configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine experiment name and paths
    experiment_name = os.path.splitext(os.path.basename(config_path))[0]
    results_dir = os.path.join('experiments', 'results', experiment_name)
    best_genome_path = os.path.join(results_dir, 'best_genome.json')
    
    # Load best genome
    best_genome = Genome.from_json(filepath=best_genome_path)
    
    # Initialize fitness function
    fitness_class = globals()[config.get('fitness_function', 'NetworkFitnessFunction')]
    fitness_function = fitness_class(config['fitness_targets'])
    
    # Run simulation with best genome
    grid = Grid(best_genome)
    grid.run_simulation(verbose=False)
    
    # Evaluate fitness
    fitness = fitness_function.evaluate(grid)
    
    print(f"\nVerification Results for {experiment_name}:")
    print(f"Fitness Score: {fitness:.5f}")
    
    # Print additional network statistics
    source_indices, target_indices = grid.neuron_connections.nonzero()
    print(f"Number of neurons: {grid.neuron_count()}")
    print(f"Number of connections: {len(source_indices)}")
    
    # Create graph for additional analysis
    G = grid.get_graph();
    nodes_no_incoming = len([node for node in G.nodes() if G.in_degree(node) == 0])
    print(f"Neurons with no incoming connections: {nodes_no_incoming}")

def generate_structural_graph_fitness_references():
    """Generate reference fitness values for structural graph comparisons."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
    references_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'references')
    
    # Create directories if they don't exist
    os.makedirs(fixtures_dir, exist_ok=True)
    os.makedirs(references_dir, exist_ok=True)
    
    # Define test cases with target graphs and sample networks
    test_cases = [
        {
            "name": "basic_structure",
            "fitness_targets": {
                "adjacency_list": {
                    "1": [2, 3],
                    "2": [4],
                    "3": [4],
                    "4": []
                }
            },
            "test_networks": [
                # Perfect match
                {
                    "adjacency_list": {
                        "1": [2, 3],
                        "2": [4],
                        "3": [4],
                        "4": []
                    }
                },
                # Different node count
                {
                    "adjacency_list": {
                        "1": [2, 3],
                        "2": [4],
                        "3": [4],
                        "4": [5],
                        "5": []
                    }
                },
                # Different degree distribution
                {
                    "adjacency_list": {
                        "1": [2],
                        "2": [3, 4],
                        "3": [4],
                        "4": []
                    }
                }
            ]
        },
        {
            "name": "hub_structure",
            "fitness_targets": {
                "adjacency_list": {
                    "1": [2, 3, 4, 5],
                    "2": [1],
                    "3": [1],
                    "4": [1],
                    "5": [1]
                }
            },
            "test_networks": [
                # Perfect match
                {
                    "adjacency_list": {
                        "1": [2, 3, 4, 5],
                        "2": [1],
                        "3": [1],
                        "4": [1],
                        "5": [1]
                    }
                },
                # Missing connections
                {
                    "adjacency_list": {
                        "1": [2, 3, 4],
                        "2": [1],
                        "3": [1],
                        "4": [1],
                        "5": []
                    }
                },
                # Different hub node
                {
                    "adjacency_list": {
                        "1": [5],
                        "2": [5],
                        "3": [5],
                        "4": [5],
                        "5": [1, 2, 3, 4]
                    }
                }
            ]
        }
    ]
    
    # Create a mock Grid class for testing
    class MockGrid:
        def __init__(self, adjacency_list):
            # Find max node ID
            max_node = max(int(node) for node in adjacency_list.keys())
            for targets in adjacency_list.values():
                if targets:
                    max_node = max(max_node, max(targets))
            
            # Create connectivity matrix
            self.neuron_connections = lil_matrix((max_node, max_node), dtype=float)
            
            # Fill connectivity matrix
            for source, targets in adjacency_list.items():
                source_idx = int(source) - 1
                for target in targets:
                    target_idx = int(target) - 1
                    self.neuron_connections[source_idx, target_idx] = 1.0
        
        def get_neuron_ids(self):
            source_nodes, target_nodes = self.neuron_connections.nonzero()
            all_nodes = np.unique(np.concatenate([source_nodes, target_nodes]))
            return [node + 1 for node in all_nodes]
    
    # Process each test case
    for case in test_cases:
        case_name = case["name"]
        
        # Save target configuration
        target_path = os.path.join(fixtures_dir, f'structural_graph_{case_name}.json')
        with open(target_path, 'w') as f:
            json.dump(case["fitness_targets"], f, indent=4)
        
        # Create fitness function with default tolerances
        fitness_fn = StructuralGraphFitnessFunction(case["fitness_targets"])
        
        # Calculate fitness for each test network
        for network in case["test_networks"]:
            # Create mock grid
            grid = MockGrid(network["adjacency_list"])
            
            # Calculate and store fitness
            network["expected_fitness"] = float(fitness_fn.evaluate(grid))
        
        # Save reference results
        reference_path = os.path.join(references_dir, f'structural_graph_{case_name}_results.json')
        with open(reference_path, 'w') as f:
            json.dump(case["test_networks"], f, indent=4)
        
        print(f"Generated structural graph fixtures for {case_name}")

def run_all_experiments(force=False):
    """
    Run all experiments from config files in alphabetical order by full path.
    
    Args:
        force (bool): If True, run all experiments regardless of config changes
    """
    experiments_dir = os.path.join('experiments')
    
    # Track successful and failed experiments
    successful = []
    failed = []
    skipped = []
    
    # Collect all config file paths
    config_paths = []
    for experiment_dir in os.listdir(experiments_dir):
        experiment_path = os.path.join(experiments_dir, experiment_dir)
        if not os.path.isdir(experiment_path):
            continue
            
        config_dir = os.path.join(experiment_path, 'configs')
        if not os.path.exists(config_dir):
            continue
            
        # Add each config file path to the list
        for config_file in os.listdir(config_dir):
            if not config_file.endswith('.json'):
                continue
            config_paths.append(os.path.join(config_dir, config_file))
    
    # Sort config paths alphabetically
    config_paths.sort()
            
    # Process each config file in sorted order
    for config_path in config_paths:
        config_file = os.path.basename(config_path)
        print(f"\n{'='*80}")
        print(f"Running experiment: {config_path}")
        print(f"{'='*80}")
        
        try:
            # Run the experiment with force option
            runner = ExperimentRunner(config_path, force=force)
            best_genome, stats = runner.run()
            
            if best_genome is None and stats is None:
                # Experiment was skipped
                skipped.append(config_path)
                continue
            
            # Generate plots
            stats_path = os.path.join('experiments', experiment_dir, 'results', 
                                    runner.experiment_name, 'stats.json')
            runner.plot_fitness_history(stats_path)
            
            successful.append(config_path)
            
        except Exception as e:
            print(f"\nError running experiment {config_path}:")
            print(f"Error: {str(e)}")
            failed.append((config_path, str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Run Summary")
    print("="*80)
    print(f"\nSuccessful experiments ({len(successful)}):")
    for exp in successful:
        print(f"  ✓ {exp}")
    
    if skipped:
        print(f"\nSkipped experiments ({len(skipped)}):")
        for exp in skipped:
            print(f"  - {exp}")
    
    if failed:
        print(f"\nFailed experiments ({len(failed)}):")
        for exp, error in failed:
            print(f"  ✗ {exp}")
            print(f"    Error: {error}")

def main():
    #generate_block_preservation_crossover_references()
    #return
    ##run_all_experiments(force=False)  # Set force=True to run all experiments regardless of config changes
    #return
    save_all_experiment_displays()
    ##run_simulation(display_weights=True, save_displays=True, verbose=False)
    return
    #run_random_simulation()  # Use random simulation instead
    #generate_new_genome_variations()
    #generate_genome_references()
    #generate_display_references()
    #generate_mutation_references()
    #generate_crossover_references()
    #generate_different_morphogen_crossover_references()
    #generate_target_graph_fitness_references()
    #generate_structural_graph_fitness_references()
    #return
    # Run experiment
    #config_path = os.path.join('experiments', '01_nodes_edges_match', 'configs', 'basic_experiment_03a.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_cartpole_01.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_02.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_02-seed2.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_02-seed2-adaptive.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_02-seed2-adaptive-distinct-elite.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_02-seed2-adaptive-distinct-elite2.json')
    config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_02-seed2-adaptive-distinct-eliteA.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_02c.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_cma_es_01.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_cma_es_01a.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_cma_es_02.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_cma_es_03.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_cma_es_04.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_cma_es_05.json')
    #config_path = os.path.join('experiments', '03_gym_optimization', 'configs', 'gym_lunarlander_04.json')
    #verify_experiment(config_path)
    #return
    
    # Display target graph
    #display_target_graph_from_config(config_path)
    #plt.show(block=True)  # Add blocking to keep window open
    
    # Run the experiment
    runner = ExperimentRunner(config_path)
    best_genome, stats = runner.run()
    
    # Plot fitness history
#    stats_path = os.path.join('experiments', '03_gym_optimization', 'results', runner.experiment_name, 'stats.json')
#    runner.plot_fitness_history(stats_path)

if __name__ == "__main__":
    main()
