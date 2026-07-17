import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class NeuronGraphDisplay:
    def __init__(self, grid, update_frequency=1, use_kamada_kawai=False, display_node_numbers=True, scale=1.0, font_size=12):
        self.grid = grid
        self.grid.add_listener(self)
        self.update_frequency = update_frequency
        self.use_kamada_kawai = use_kamada_kawai
        self.display_node_numbers = display_node_numbers
        self.scale = scale
        self.font_size = font_size
        
        # Enable interactive mode
        plt.ion()
        
        # Create figure
        self.fig = plt.figure(figsize=(8 * scale, 8 * scale))
        self.ax = self.fig.add_subplot(111)
        
        # Initialize graph
        self.G = nx.DiGraph()
        
        # Store node positions
        self.pos = None
        
        # Flag for control flow
        self.ready_for_next = True
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.release_wait)
        self.fig.canvas.mpl_connect('button_press_event', self.release_wait)
        
        # Initial draw
        self._update_graph()
        
        plt.tight_layout()

    def _update_graph(self):
        """Update the graph visualization."""
        self.ax.clear()
        
        # Create new graph
        self.G = self.grid.get_graph();
        
        # Calculate layout based on preference
        if self.use_kamada_kawai:
            try:
                self.pos = nx.kamada_kawai_layout(self.G)
            except nx.NetworkXError:
                # Fall back to spring layout if Kamada-Kawai fails
                self.pos = nx.spring_layout(self.G, k=1.5, iterations=50, seed=42)
        else:
            # Use default spring layout
            self.pos = nx.spring_layout(self.G, k=1.5, iterations=50, seed=42)
        
        # Determine nodes with no incoming connections
        nodes_no_incoming = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
        node_colors = ['yellow' if node in nodes_no_incoming else 'lightblue' for node in self.G.nodes()]
        
        # Calculate node size based on figure coordinate system
        # Base node size that works well for scale=1.0
        base_node_size = 400
        # Scale node size by the square of the scale factor to maintain proportional area
        scaled_node_size = int(base_node_size * (self.scale ** 2))
        
        # Draw the graph
        nx.draw_networkx_nodes(self.G, self.pos, 
                              node_color=node_colors,
                              node_size=scaled_node_size,
                              alpha=0.6,
                              ax=self.ax)
        
        nx.draw_networkx_edges(self.G, self.pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=int((10 if not self.display_node_numbers else 20) * self.scale),  # Arrow size scales linearly
                              width=1.0 * self.scale,  # Scale edge width
                              node_size=scaled_node_size,  # Pass node size to connection style
                              ax=self.ax)
        
        if self.display_node_numbers:
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=int(self.font_size / 12 * 10 * self.scale))
        
        # Calculate network statistics
        num_isolated = len(nodes_no_incoming)
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        density = nx.density(self.G)
        avg_degree = sum(dict(self.G.degree()).values()) / num_nodes if num_nodes > 0 else 0
        
        # Check connectivity with safety checks for empty graphs
        is_strongly_connected = nx.is_strongly_connected(self.G) if num_nodes > 0 else False
        is_weakly_connected = nx.is_weakly_connected(self.G) if num_nodes > 0 else False
        
        # Update title with statistics
        self.ax.set_title(f'Neural Network Graph (Nodes: {num_nodes}, Connections: {num_edges})\n'
                          f'Nodes with no incoming connections: {num_isolated}\n'
                          f'Density: {density:.3f} | Avg Degree: {avg_degree:.2f}\n'
                          f'Strongly Connected: {"Yes" if is_strongly_connected else "No"} | '
                          f'Weakly Connected: {"Yes" if is_weakly_connected else "No"}',
                          fontsize=int(self.font_size * self.scale))
        
        # Set axis limits with padding based on actual node positions
        if self.pos:  # Check if we have any positions
            pos_array = np.array(list(self.pos.values()))
            x_min, y_min = pos_array.min(axis=0) - 0.2
            x_max, y_max = pos_array.max(axis=0) + 0.2
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
        else:
            # Default limits for empty graph
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
        
        # Adjust layout to fit everything
        plt.tight_layout()
        
        # Draw the canvas
        self.fig.canvas.draw_idle()

    def release_wait(self, event):
        """Signal that we're ready for the next step"""
        self.ready_for_next = True

    def on_step(self):
        """Called by Grid after each step."""
        # Only update display every N steps
        if self.grid.iteration % self.update_frequency != 0:
            return
        
        self._update_graph()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self, block=True):
        plt.show(block=block)

    @staticmethod
    def display_target_graph(adjacency_list, display_node_numbers=True, scale=1.0, font_size=12):
        """Display a target graph from an adjacency list."""
        # Create figure
        fig = plt.figure(figsize=(8 * scale, 8 * scale))
        ax = fig.add_subplot(111)
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for source, targets in adjacency_list.items():
            G.add_node(int(source))
            for target in targets:
                G.add_edge(int(source), int(target))
        
        # Calculate layout
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
                             arrowsize=int(20 * scale),  # Arrow size scales linearly
                             width=1.0 * scale,  # Scale edge width
                             node_size=scaled_node_size,  # Pass node size to connection style
                             ax=ax)
        
        if display_node_numbers:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=int(font_size / 12 * 10 * scale))
        
        # Calculate network statistics
        num_isolated = len(nodes_no_incoming)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
        is_strongly_connected = nx.is_strongly_connected(G) if num_nodes > 0 else False
        is_weakly_connected = nx.is_weakly_connected(G) if num_nodes > 0 else False
        
        # Update title with statistics
        ax.set_title(f'Target Neural Network Graph (Nodes: {num_nodes}, Connections: {num_edges})\n'
                     f'Nodes with no incoming connections: {num_isolated}\n'
                     f'Density: {density:.3f} | Avg Degree: {avg_degree:.2f}\n'
                     f'Strongly Connected: {"Yes" if is_strongly_connected else "No"} | '
                     f'Weakly Connected: {"Yes" if is_weakly_connected else "No"}',
                     fontsize=int(12 * scale))
        
        # Set axis limits with padding
        pos_array = np.array(list(pos.values()))
        x_min, y_min = pos_array.min(axis=0) - 0.2
        x_max, y_max = pos_array.max(axis=0) + 0.2
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig 
