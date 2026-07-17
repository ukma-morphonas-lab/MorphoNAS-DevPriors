import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches  # For drawing cell borders
import matplotlib.path as mpath


class MorphogenDisplay:
    def __init__(self, grid, update_frequency=1, scale=1.0, font_size=12):
        self.grid = grid
        self.grid.add_listener(self)  # Register as a listener
        self.ready_for_next = True  # Flag to control flow
        self.update_frequency = update_frequency  # Update frequency as parameter
        self.scale = scale
        self.font_size = font_size

        # Enable interactive mode
        plt.ion()

        # Create figure with one main plot, colorbars on the right, and matrices below
        self.fig = plt.figure(figsize=(12 * scale, 8 * scale))

        # Create grid spec to arrange plots
        gs = self.fig.add_gridspec(2, grid.num_morphogens + 1,
                                 width_ratios=[3] + [0.3]*grid.num_morphogens,
                                 height_ratios=[2, 1])  # Main display area and matrix area

        # Main plot for combined RGB
        self.ax_main = self.fig.add_subplot(gs[0, 0])

        # Initialize RGB data with white base
        rgb_data = self._get_rgb_data()
        self.main_img = self.ax_main.imshow(rgb_data, vmin=0, vmax=1)
        self.ax_main.set_title('Combined Morphogens', fontsize=int(font_size * scale))
        
        # Scale main plot axes and frame
        self.ax_main.tick_params(axis='both', which='both', 
                                labelsize=int(font_size / 12 * 10 * scale), width=1.0 * scale, length=3.0 * scale)
        for spine in self.ax_main.spines.values():
            spine.set_linewidth(1.0 * scale)

        # Cell overlay (using patches for borders)
        self.cell_borders = []

        # Create colorbars for each morphogen
        self.colorbar_axes = []
        self.colorbar_images = []
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']  # Add more if needed

        for i in range(grid.num_morphogens):
            ax = self.fig.add_subplot(gs[0, i+1])
            img = ax.imshow(grid.get_morphogen_array(i),
                          cmap=plt.cm.get_cmap('Reds' if i == 0 else 'Greens' if i == 1 else 'Blues'),
                          vmin=0, vmax=1)
            cbar = plt.colorbar(img, ax=ax)
            ax.set_title(f'M{i}', fontsize=int(font_size / 12 * 10 * scale))
            ax.axis('off')
            # Scale colorbar plot frame (even though axis is off, spines might still be visible)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0 * scale)
            # Scale colorbar tick labels
            cbar.ax.tick_params(axis='both', which='both', 
                               labelsize=int(font_size / 12 * 10 * scale), width=1.0 * scale, length=3.0 * scale)
            self.colorbar_axes.append(ax)
            self.colorbar_images.append(img)

        # Add matrices display in the bottom row
        self.ax_matrices = self.fig.add_subplot(gs[1, :])
        self.ax_matrices.axis('off')
        self._update_matrices_display()

        # Connect events to just release the wait
        self.fig.canvas.mpl_connect('key_press_event', self.release_wait)
        self.fig.canvas.mpl_connect('button_press_event', self.release_wait)

        # Initial title update
        self.update_titles()

        plt.tight_layout()

        self.axon_paths = []
        self._update_cell_borders()

    def _get_rgb_data(self):
        """Generate the RGB data for the main display."""
        rgb_data = np.ones((self.grid.size_x, self.grid.size_y, 3))  # Start with white
        for i in range(min(3, self.grid.num_morphogens)):
            rgb_data[:, :, i] -= np.clip(self.grid.get_morphogen_array(i), 0, 1)  # Subtract concentration
        return rgb_data

    def update_titles(self):
        totals = [self.grid.get_morphogen_sum(i) for i in range(self.grid.num_morphogens)]
        title = f'Step {self.grid.iteration}\n'
        title += '\n'.join([f'M{i} total: {t:.6f}' for i, t in enumerate(totals)])
        self.ax_main.set_title(title, fontsize=int(self.font_size * self.scale))

    def _update_cell_borders(self):
        """Update the cell borders for display."""
        # Clear previous borders
        for border in self.cell_borders:
            border.remove()
        self.cell_borders = []

        # Clear previous axons
        for axon in self.axon_paths:
            axon.remove()
        self.axon_paths = []

        # Draw neuron connections first (thicker solid lines)
        source_indices, target_indices = self.grid.neuron_connections.nonzero()
        neuron_ids = self.grid.get_neuron_ids()
        for source_id, target_id in zip(source_indices + 1, target_indices + 1):
            if source_id in neuron_ids and target_id in neuron_ids:
                source_pos = self.grid.get_cell_position(source_id)
                target_pos = self.grid.get_cell_position(target_id)
                
                # Get the RGB color at source cell's position for the connection color
                source_color = self.main_img.get_array()[source_pos]
                connection_color = 1 - source_color  # Invert RGB values
                connection_color = np.clip(connection_color, 0, 1)

                # Calculate shortest path considering wrapping
                dx = target_pos[1] - source_pos[1]
                dy = target_pos[0] - source_pos[0]
                
                # Adjust for wrapping
                if dx > self.grid.size_x/2:
                    dx -= self.grid.size_x
                elif dx < -self.grid.size_x/2:
                    dx += self.grid.size_x
                    
                if dy > self.grid.size_y/2:
                    dy -= self.grid.size_y
                elif dy < -self.grid.size_y/2:
                    dy += self.grid.size_y

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
                                        linewidth=2.0 * self.scale, alpha=0.9)
                self.ax_main.add_patch(patch)
                self.axon_paths.append(patch)

        # Draw cells and growing axons
        for cell_id in self.grid.get_cell_ids():
            x, y = self.grid.get_cell_position(cell_id)

            # Get the RGB color at this cell's position
            cell_color = self.main_img.get_array()[x, y]
            border_color = 1 - cell_color  # Invert RGB values
            border_color = np.clip(border_color, 0, 1)

            # Set linestyle based on cell type
            linestyle = ':' if self.grid.is_progenitor(cell_id) else '-'

            # Add cell border
            rect = patches.Rectangle(
                (y - 0.5, x - 0.5), 1, 1,
                linewidth=1.5 * self.scale,
                edgecolor=border_color,
                facecolor='none',
                linestyle=linestyle
            )
            self.ax_main.add_patch(rect)
            self.cell_borders.append(rect)

            # Draw axon if this is a neuron with a growing axon (not yet connected)
            if (self.grid.is_neuron(cell_id) and 
                len(self.grid.get_axon(cell_id)) > 1 and 
                self.grid.neuron_connections[cell_id - 1].nnz == 0):  # Check if neuron has no connections
                self._draw_axon(self.grid.get_axon(cell_id), border_color)

    def _draw_axon(self, axon_points, color):
        """Draw an axon as a dotted curve."""
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
                if dx > self.grid.size_x/2:
                    dx -= self.grid.size_x
                elif dx < -self.grid.size_x/2:
                    dx += self.grid.size_x
                    
                if dy > self.grid.size_y/2:
                    dy -= self.grid.size_y
                elif dy < -self.grid.size_y/2:
                    dy += self.grid.size_y
                    
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
                                linewidth=1.5 * self.scale, alpha=1, linestyle=':')  # Dotted line for growing axons
        self.ax_main.add_patch(patch)
        self.axon_paths.append(patch)

    def release_wait(self, event):
        """Signal that we're ready for the next step"""
        self.ready_for_next = True

    def _update_matrices_display(self):
        """Update the display of diffusion and inhibition matrices."""
        self.ax_matrices.clear()
        self.ax_matrices.axis('off')
        
        # Split each matrix into lines
        matrix_lines = []
        
        # Add genome parameters at the top
        genome_params = f"$G_{{iter}}$: {self.grid.iteration}    $G_{{dim}}$: {self.grid.size_x}×{self.grid.size_y}\n\n"
        
        # Format diffusion patterns
        for i in range(self.grid.num_morphogens):
            # Each number will be 0.xxx, so calculate width based on pattern size
            pattern = self.grid.diffusion_patterns[i]
            pattern_width = 5 * pattern.shape[1] + (pattern.shape[1] - 1)  # 5 chars per number + spaces between
            header_width = max(pattern_width, 17)  # Use at least 17 chars or pattern width
            lines = [f"{'M' + str(i):^{header_width}}"]  # Center the header
            for row in pattern:
                lines.append(" ".join([f"{x:.3f}" for x in row]))
            matrix_lines.append(lines)
        
        # Join diffusion matrices horizontally, handling different heights
        diffusion_text = ""
        max_lines = max(len(lines) for lines in matrix_lines)
        
        # For each line index
        for line_idx in range(max_lines):
            # Get each matrix's line at this index, or empty string if matrix is shorter
            line_parts = []
            for matrix in matrix_lines:
                if line_idx < len(matrix):
                    # Get actual line and pad to match pattern width
                    pattern = self.grid.diffusion_patterns[matrix_lines.index(matrix)]
                    pattern_width = 5 * pattern.shape[1] + (pattern.shape[1] - 1)
                    header_width = max(pattern_width, 17)
                    line_parts.append(f"{matrix[line_idx]:{header_width}}")
                else:
                    # Add empty string with proper width for alignment
                    pattern = self.grid.diffusion_patterns[matrix_lines.index(matrix)]
                    pattern_width = 5 * pattern.shape[1] + (pattern.shape[1] - 1)
                    header_width = max(pattern_width, 17)
                    line_parts.append(" " * header_width)
            
            line = "    ".join(line_parts)
            diffusion_text += line + "\n"
        
        # Add a blank line
        diffusion_text += "\n"
        
        # Calculate spacing for thresholds
        matrix_width = 4 + 6 * self.grid.num_morphogens  # Width of inhibition matrix content
        header_spacing = 8  # Space between matrix and "Thresholds:"
        label_spacing = header_spacing + 2  # Additional indent for threshold labels
        header_row_spacing = label_spacing + 3  # Extra space for "Division" to account for "M0 M1..." header row
        
        # Format inhibition matrix and thresholds side by side
        inhib_text = f"{'Inhibition:':^{matrix_width}}" + " " * header_spacing + "$G_{fates}$:\n"
        inhib_text += "   " + "    ".join([f"M{i:d}" for i in range(self.grid.num_morphogens)])
        inhib_text += " " * header_row_spacing + "Division:           " + f"{self.grid.division_threshold:.3f}\n"
        
        for i, row in enumerate(self.grid.inhibition_matrix):
            line = f"M{i:d} " + " ".join([f"{x:.3f}" for x in row])
            if i == 0:
                line += " " * label_spacing + "Differentiation:    " + f"{self.grid.cell_differentiation_threshold:.3f}"
            elif i == 1:
                line += " " * label_spacing + "Axon Growth:        " + f"{self.grid.axon_growth_threshold:.3f}"
            elif i == 3:
                line += " " * label_spacing + "Weight Target:      " + f"{self.grid.weight_adjustment_target:.3f}"
            elif i == 4:
                line += " " * label_spacing + "Weight Rate:        " + f"{self.grid.weight_adjustment_rate:.3f}"
            line += "\n"
            inhib_text += line
        
        # Add secretion rates and axon parameters blocks side by side
        secretion_width = 15  # Approximate width for secretion rates block
        axon_spacing = 3  # Space between secretion rates and axon block
        
        inhib_text += "\nSecretion Rates:" + " " * (secretion_width+axon_spacing) + "$G_{axon}$:\n"
        inhib_text += "Progenitor: " + " ".join([f"{x:.3f}" for x in self.grid.progenitor_secretion_rates])
        inhib_text += " " * (axon_spacing + 2) + "Axon Connect:       " + f"{self.grid.axon_connect_threshold:.3f}\n"
        inhib_text += "Neuron:     " + " ".join([f"{x:.3f}" for x in self.grid.neuron_secretion_rates])
        inhib_text += " " * (axon_spacing + 2) + "Max Axon Length:    " + f"{self.grid.max_axon_length}\n"
        
        # Combine both texts
        full_text = genome_params + diffusion_text + inhib_text
        
        self.ax_matrices.text(0, 1, full_text, fontfamily='monospace', 
                            verticalalignment='top', horizontalalignment='left',
                            transform=self.ax_matrices.transAxes, fontsize=int(self.font_size / 12 * 10 * self.scale))

    def on_step(self):
        """Called by Grid after each step."""
        # Only update display every N steps
        if self.grid.iteration % self.update_frequency != 0:
            return
        
        # Update RGB display
        rgb_data = self._get_rgb_data()
        self.main_img.set_data(rgb_data)

        # Update individual morphogen displays
        for i, img in enumerate(self.colorbar_images):
            img.set_data(self.grid.get_morphogen_array(i))

        # Update cell borders
        self._update_cell_borders()
        
        # Update matrices display
        self._update_matrices_display()

        self.update_titles()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self, block=True):
        """Show the display. If block=False, don't block execution."""
        plt.show(block=block)
