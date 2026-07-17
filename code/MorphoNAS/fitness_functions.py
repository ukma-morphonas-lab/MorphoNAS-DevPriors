import numpy as np
import networkx as nx

# Dynamically fix the ComplexWarning import issue in grakel
if np.__version__ >= '2.0.0':
    import numpy.exceptions
    np.ComplexWarning = numpy.exceptions.ComplexWarning

from grakel import WeisfeilerLehman, Graph
from grakel.kernels import WeisfeilerLehman

class BaseFitnessFunction:
    """Base class for fitness functions"""
    def __init__(self, targets, penalize_morphogens=False, penalize_steps=False, penalize_dimensions=False, penalize_connections=False,
                 max_unpenalized_connections=50, connection_half_decay=1000, min_connection_fitness=0.8):
        self.targets = targets
        self.penalize_morphogens = penalize_morphogens
        self.penalize_steps = penalize_steps
        self.penalize_dimensions = penalize_dimensions
        self.penalize_connections = penalize_connections
        self.max_unpenalized_connections = max_unpenalized_connections
        self.connection_half_decay = connection_half_decay
        self.min_connection_fitness = min_connection_fitness
        
    def _calculate_morphogen_penalty(self, num_morphogens):
        """Calculate penalty factor based on number of morphogens"""
        if not self.penalize_morphogens or num_morphogens <= 3:
            return 1.0
        return max(0.0, 1.0 - (0.01 * (num_morphogens - 3)))
    
    def _calculate_steps_penalty(self, num_steps):
        """Calculate penalty factor based on number of growth steps"""
        if not self.penalize_steps or num_steps <= 100:
            return 1.0
        return max(0.0, 1.0 - (0.0001 * (num_steps - 100)))
    
    def _calculate_dimensions_penalty(self, size_x, size_y):
        """Calculate penalty factor based on grid dimensions"""
        total_cells = size_x * size_y
        if not self.penalize_dimensions or total_cells <= 100:
            return 1.0
        return max(0.0, 1.0 - (0.000001 * (total_cells - 100)))

    def _calculate_connections_penalty(self, num_connections):
        """Calculate penalty factor based on number of connections using asymptotic decay.
        
        The penalty factor P_conn(N_C) is defined as:
        P_conn(N_C) = α + (1-α) * exp(-λ * max(0, N_C - θ_C))
        where:
        - N_C is the number of connections
        - θ_C is max_unpenalized_connections (threshold for no penalty)
        - λ is the decay rate = ln(2)/(θ_half - θ_C)
        - θ_half is connection_half_decay (where penalty reaches (1+α)/2)
        - α is min_connection_fitness (asymptotic minimum fitness)
        """
        if not self.penalize_connections or num_connections <= self.max_unpenalized_connections:
            return 1.0
            
        # Calculate decay rate λ
        decay_rate = np.log(2) / (self.connection_half_decay - self.max_unpenalized_connections)
        
        # Calculate penalty factor
        excess_connections = max(0, num_connections - self.max_unpenalized_connections)
        penalty = self.min_connection_fitness + (1 - self.min_connection_fitness) * np.exp(-decay_rate * excess_connections)
        
        return penalty
        
    def evaluate(self, grid):
        """Base evaluation that applies penalties"""
        base_fitness = self._evaluate(grid)
        
        # Apply morphogen penalty if enabled
        if self.penalize_morphogens:
            morphogen_penalty = self._calculate_morphogen_penalty(grid.num_morphogens)
            base_fitness *= morphogen_penalty
            
        # Apply steps penalty if enabled
        if self.penalize_steps:
            steps_penalty = self._calculate_steps_penalty(grid.max_growth_steps)
            base_fitness *= steps_penalty
            
        # Apply dimensions penalty if enabled
        if self.penalize_dimensions:
            dimensions_penalty = self._calculate_dimensions_penalty(grid.size_x, grid.size_y)
            base_fitness *= dimensions_penalty

        # Apply connections penalty if enabled
        if self.penalize_connections:
            G = grid.get_graph()
            connections_penalty = self._calculate_connections_penalty(G.number_of_edges())
            base_fitness *= connections_penalty
            
        return base_fitness
    
    def _evaluate(self, grid):
        """Must be implemented by subclasses"""
        raise NotImplementedError

class NetworkFitnessFunction(BaseFitnessFunction):
    """Evaluates fitness based on network properties like neuron count and connectivity"""
    
    def __init__(self, targets, penalize_morphogens=False, penalize_steps=False, penalize_dimensions=False, penalize_connections=False):
        super().__init__(targets, penalize_morphogens=penalize_morphogens, 
                        penalize_steps=penalize_steps, penalize_dimensions=penalize_dimensions,
                        penalize_connections=penalize_connections)
    
    def _evaluate(self, grid):
        """
        Evaluate fitness based on configured targets.
        Returns value between 0 and 1, where 1 is perfect fitness.
        """
        # Calculate neuron fitness
        neuron_count = grid.neuron_count()
        neuron_distance = abs(neuron_count - self.targets['neurons'])
        neuron_fitness = np.exp(-neuron_distance / self.targets['neuron_tolerance'])
        
        # Create graph for connectivity analysis
        G = grid.get_graph();
        
        # Calculate nodes with no incoming connections
        nodes_no_incoming = len([node for node in G.nodes() if G.in_degree(node) == 0])
        
        # Calculate connection count
        connection_count = G.number_of_edges()
        
        # Calculate remaining fitness components
        no_incoming_distance = abs(nodes_no_incoming - self.targets['no_incoming'])
        no_incoming_fitness = np.exp(-no_incoming_distance / self.targets['no_incoming_tolerance'])
        
        connection_distance = abs(connection_count - self.targets['connections'])
        connection_fitness = np.exp(-connection_distance / self.targets['connection_tolerance'])
        
        # Calculate final fitness
        fitness = neuron_fitness * no_incoming_fitness * connection_fitness
        
        # Check if degree vectors are provided in targets
        if 'indegrees' in self.targets and 'outdegrees' in self.targets:
            # Extract actual degree vectors from the graph
            # Sort nodes to ensure consistent ordering
            sorted_nodes = sorted(G.nodes())
            actual_indegrees = [G.in_degree(node) for node in sorted_nodes]
            actual_outdegrees = [G.out_degree(node) for node in sorted_nodes]
            
            # Get target degree vectors
            target_indegrees = self.targets['indegrees']
            target_outdegrees = self.targets['outdegrees']
            
            # Pad shorter vectors with zeros to match lengths
            max_len = max(len(actual_indegrees), len(target_indegrees))
            actual_indegrees.extend([0] * (max_len - len(actual_indegrees)))
            target_indegrees.extend([0] * (max_len - len(target_indegrees)))
            
            max_len_out = max(len(actual_outdegrees), len(target_outdegrees))
            actual_outdegrees.extend([0] * (max_len_out - len(actual_outdegrees)))
            target_outdegrees.extend([0] * (max_len_out - len(target_outdegrees)))
            
            # Calculate degree vector similarities
            indegree_diff = sum(abs(a - t) for a, t in zip(actual_indegrees, target_indegrees))
            outdegree_diff = sum(abs(a - t) for a, t in zip(actual_outdegrees, target_outdegrees))
            
            # Use tolerance from connection_tolerance for degree vector comparison
            degree_tolerance = self.targets.get('connection_tolerance', 2)
            
            indegree_fitness = np.exp(-indegree_diff / (max_len * degree_tolerance))
            outdegree_fitness = np.exp(-outdegree_diff / (max_len_out * degree_tolerance))
            
            # Include degree vector fitness in the overall calculation
            fitness *= indegree_fitness * outdegree_fitness
        
        # Apply connectivity penalty if configured
        if self.targets.get('require_weak_connectivity', False):
            if G.number_of_nodes() > 0 and not nx.is_weakly_connected(G):
                fitness *= self.targets.get('connectivity_penalty', 0.5)
        
        return fitness 

class TargetGraphFitnessFunction(BaseFitnessFunction):
    """Evaluates fitness using Graph Kernels (Weisfeiler-Lehman Kernel)"""
    
    def __init__(self, targets, wl_iterations=5, penalize_morphogens=False, penalize_steps=False, penalize_dimensions=False, penalize_connections=False):
        """
        Initialize with target graph specification.
        
        targets should include:
        - adjacency_list: Dict mapping node IDs to lists of target nodes
        - node_count_tolerance: How much deviation in total nodes is allowed
        - topology_weight: Weight given to graph similarity (0-1)
        - size_weight: Weight given to node count similarity (0-1)
        
        wl_iterations: Number of iterations for the Weisfeiler-Lehman Kernel
        """
        super().__init__(targets, penalize_morphogens=penalize_morphogens, 
                        penalize_steps=penalize_steps, penalize_dimensions=penalize_dimensions,
                        penalize_connections=penalize_connections)
        self.wl_iterations = wl_iterations
        
        # Create target graph from adjacency list
        self.target_graph = nx.DiGraph()
        for source, targets in self.targets['adjacency_list'].items():
            for target in targets:
                self.target_graph.add_edge(int(source), int(target))
        
        # Convert target graph to Grakel format
        self.target_grakel = self._convert_to_grakel(self.target_graph)
    
    def _convert_to_grakel(self, graph):
        """Converts a NetworkX graph to a Grakel-compatible format."""
        edge_list = list(graph.edges())  # Extract edges
        nodes = sorted(graph.nodes())  # Ensure consistent node ordering

        # Convert np.int64 keys to standard int
        nodes = [int(n) for n in nodes]

        # If the graph is empty, add a self-loop at node 0
        if not edge_list:
            if not nodes:  # No nodes at all, create a dummy node
                nodes = [0]
            edge_list = [(nodes[0], nodes[0])]

        # Assign unique string labels to every node
        node_labels = {int(node): str(i) for i, node in enumerate(nodes)}

        return Graph(edge_list, node_labels=node_labels)
    
    def _evaluate(self, grid):
        """Compare grid's network topology to target graph using Graph Kernels."""
        # Create graph from grid's connectivity
        actual_graph = grid.get_graph();

        # Convert actual graph to Grakel format
        actual_grakel = self._convert_to_grakel(actual_graph)
        
        # Compute Weisfeiler-Lehman Kernel similarity
        wl_kernel = WeisfeilerLehman(n_iter=self.wl_iterations, normalize=True)
        similarity_matrix = wl_kernel.fit_transform([self.target_grakel, actual_grakel])
        similarity = similarity_matrix[0, 1]  # Extract similarity score
        
        # Penalize graphs with orphan nodes
        total_nodes = actual_graph.number_of_nodes()
        orphan_nodes = sum(1 for node in actual_graph.nodes() if actual_graph.degree(node) == 0)
        orphan_penalty = 1 - (0.9 * (orphan_nodes / total_nodes)) if total_nodes > 0 else 0.1
        similarity *= orphan_penalty

        # Calculate node count similarity
        target_nodes = self.target_graph.number_of_nodes()
        actual_nodes = actual_graph.number_of_nodes()
        node_diff = abs(target_nodes - actual_nodes)
        size_fitness = np.exp(-node_diff / self.targets['node_count_tolerance'])
        
        # Combine metrics using configured weights
        topology_weight = self.targets.get('topology_weight', 0.7)
        size_weight = self.targets.get('size_weight', 0.3)
        
        return (topology_weight * similarity + size_weight * size_fitness)

class StructuralGraphFitnessFunction(BaseFitnessFunction):
    """Evaluates fitness by comparing basic structural properties between graphs"""
    
    def __init__(self, targets, connection_tolerance=2, node_tolerance=1, 
                 penalize_morphogens=False, penalize_steps=False, penalize_dimensions=False, penalize_connections=False):
        """
        Initialize with target graph specification.
        
        targets should include:
        - adjacency_list: Dict mapping node IDs to lists of target nodes
        
        connection_tolerance: How much deviation in connections is allowed
        node_tolerance: How much deviation in node count is allowed
        penalize_morphogens: If True, applies penalty for using more than 3 morphogens
        """
        super().__init__(targets, penalize_morphogens=penalize_morphogens, 
                        penalize_steps=penalize_steps, penalize_dimensions=penalize_dimensions,
                        penalize_connections=penalize_connections)
        self.connection_tolerance = connection_tolerance
        self.node_tolerance = node_tolerance
        
        # Create target graph from adjacency list
        self.target_graph = nx.DiGraph()
        for source, targets in self.targets['adjacency_list'].items():
            for target in targets:
                self.target_graph.add_edge(int(source), int(target))
    
    def _get_degree_sequence(self, graph, in_degree=True):
        """Returns sorted degree sequence, padded with zeros if needed"""
        if in_degree:
            degrees = [graph.in_degree(node) for node in graph.nodes()]
        else:
            degrees = [graph.out_degree(node) for node in graph.nodes()]
        degrees.sort(reverse=True)
        return degrees
    
    def _get_sorted_out_by_in_degrees(self, graph):
        """Returns sequence of out-degrees, sorted by in-degrees of nodes"""
        # Create list of (in_degree, out_degree) pairs
        degree_pairs = [(graph.in_degree(node), graph.out_degree(node)) 
                       for node in graph.nodes()]
        # Sort by in_degree (first element) and return out_degrees
        degree_pairs.sort(reverse=True)
        return [out_deg for _, out_deg in degree_pairs]
    
    def _evaluate(self, grid):
        """Compare structural properties between grid's network and target graph"""
        # Create graph from grid's connectivity
        actual_graph = grid.get_graph();
        
        # Compare neuron counts
        target_neurons = self.target_graph.number_of_nodes()
        actual_neurons = actual_graph.number_of_nodes()
        neuron_distance = abs(target_neurons - actual_neurons)
        neuron_fitness = np.exp(-neuron_distance / self.node_tolerance)
        
        # Compare connection counts
        target_connections = self.target_graph.number_of_edges()
        actual_connections = actual_graph.number_of_edges()
        connection_distance = abs(target_connections - actual_connections)
        connection_fitness = np.exp(-connection_distance / self.connection_tolerance)
        
        # Compare in-degree distributions
        target_in_degrees = self._get_degree_sequence(self.target_graph, in_degree=True)
        actual_in_degrees = self._get_degree_sequence(actual_graph, in_degree=True)
        
        # Pad shorter sequence with zeros
        max_len = max(len(target_in_degrees), len(actual_in_degrees))
        target_in_degrees.extend([0] * (max_len - len(target_in_degrees)))
        actual_in_degrees.extend([0] * (max_len - len(actual_in_degrees)))
        
        # Handle empty graphs case
        if max_len == 0:
            in_degree_fitness = 1.0  # Perfect match for empty graphs
        else:
            in_degree_diff = sum(abs(t - a) for t, a in zip(target_in_degrees, actual_in_degrees))
            in_degree_fitness = np.exp(-in_degree_diff / (max_len * self.connection_tolerance))
        
        # Compare out-degree distributions
        target_out_degrees = self._get_degree_sequence(self.target_graph, in_degree=False)
        actual_out_degrees = self._get_degree_sequence(actual_graph, in_degree=False)
        
        # Pad shorter sequence with zeros
        target_out_degrees.extend([0] * (max_len - len(target_out_degrees)))
        actual_out_degrees.extend([0] * (max_len - len(actual_out_degrees)))
        
        # Handle empty graphs case
        if max_len == 0:
            out_degree_fitness = 1.0  # Perfect match for empty graphs
        else:
            out_degree_diff = sum(abs(t - a) for t, a in zip(target_out_degrees, actual_out_degrees))
            out_degree_fitness = np.exp(-out_degree_diff / (max_len * self.connection_tolerance))
        
        # Compare out-degrees of nodes when sorted by in-degrees
        target_out_by_in = self._get_sorted_out_by_in_degrees(self.target_graph)
        actual_out_by_in = self._get_sorted_out_by_in_degrees(actual_graph)
        
        # Pad shorter sequence with zeros
        target_out_by_in.extend([0] * (max_len - len(target_out_by_in)))
        actual_out_by_in.extend([0] * (max_len - len(actual_out_by_in)))
        
        # Handle empty graphs case
        if max_len == 0:
            out_by_in_fitness = 1.0  # Perfect match for empty graphs
        else:
            out_by_in_diff = sum(abs(t - a) for t, a in zip(target_out_by_in, actual_out_by_in))
            out_by_in_fitness = np.exp(-out_by_in_diff / (max_len * self.connection_tolerance))
        
        # Combine all metrics with equal weights
        return (neuron_fitness * connection_fitness * 
                in_degree_fitness * out_degree_fitness * 
                out_by_in_fitness)

class HierarchicalGraphFitnessFunction(BaseFitnessFunction):
    """
    Evaluates fitness using a hierarchical approach that rewards
    incremental progress toward target graph similarity.
    """
    
    def __init__(self, targets, penalize_morphogens=False, penalize_steps=False, 
                 penalize_dimensions=False, penalize_connections=False, wl_iterations=3, progression_weights=None):
        """
        Initialize with target graph specification.
        
        Args:
            targets: Dict containing target graph specification:
                - adjacency_list: Dict mapping node IDs to lists of target nodes
                - node_count_tolerance: How much deviation in node count is allowed
                - connection_tolerance: How much deviation in connection count is allowed
            penalize_morphogens: Whether to penalize excess morphogens
            penalize_steps: Whether to penalize excess simulation steps
            penalize_dimensions: Whether to penalize excess grid dimensions
            wl_iterations: Number of iterations for Weisfeiler-Lehman kernel
            progression_weights: Dict with weights for each level of similarity:
                - size_weight: Weight for basic size similarity
                - degree_weight: Weight for degree distribution similarity
                - motif_weight: Weight for small motif similarity
                - structure_weight: Weight for overall structural similarity
        """
        super().__init__(targets, penalize_morphogens=penalize_morphogens, 
                        penalize_steps=penalize_steps, penalize_dimensions=penalize_dimensions,
                        penalize_connections=penalize_connections)
        
        # Default weights if not provided
        self.progression_weights = progression_weights or {
            'size_weight': 0.15,
            'degree_weight': 0.25,
            'motif_weight': 0.3,
            'structure_weight': 0.3
        }
        
        self.wl_iterations = wl_iterations
        
        # Create target graph from adjacency list
        self.target_graph = nx.DiGraph()
        for source, targets in self.targets['adjacency_list'].items():
            for target in targets:
                self.target_graph.add_edge(int(source), int(target))
        
        # Precompute target graph properties
        self._compute_target_properties()
        
        # Convert target graph to GraKel format for WL kernel
        if self.progression_weights['structure_weight'] > 0:
            self.target_grakel = self._convert_to_grakel(self.target_graph)
    
    def _compute_target_properties(self):
        """Precompute properties of the target graph for efficiency"""
        G = self.target_graph
        
        # Basic properties
        self.target_node_count = G.number_of_nodes()
        self.target_edge_count = G.number_of_edges()
        
        # Degree distributions
        self.target_in_degrees = sorted([G.in_degree(n) for n in G.nodes()], reverse=True)
        self.target_out_degrees = sorted([G.out_degree(n) for n in G.nodes()], reverse=True)
        
        # Connected components - handle empty graph case
        if G.number_of_nodes() == 0:
            self.target_weakly_connected = 0
            self.target_strongly_connected_count = 0
        else:
            try:
                self.target_weakly_connected = 1 if nx.is_weakly_connected(G) else 0
                self.target_strongly_connected_count = nx.number_strongly_connected_components(G)
            except nx.NetworkXPointlessConcept:
                # This handles cases where connectivity is undefined
                self.target_weakly_connected = 0
                self.target_strongly_connected_count = 0
        
        # Small motifs
        self.target_reciprocity = nx.overall_reciprocity(G) if G.number_of_edges() > 0 else 0
        self.target_triangles = sum(nx.triangles(G.to_undirected()).values()) / 3 if G.number_of_nodes() > 0 else 0
        
        # Centrality metrics for nodes (average and distribution)
        if G.number_of_nodes() > 0:
            in_centrality = nx.in_degree_centrality(G)
            out_centrality = nx.out_degree_centrality(G)
            self.target_centrality_in_avg = sum(in_centrality.values()) / len(in_centrality)
            self.target_centrality_out_avg = sum(out_centrality.values()) / len(out_centrality)
            self.target_centrality_in_dist = sorted(in_centrality.values(), reverse=True)
            self.target_centrality_out_dist = sorted(out_centrality.values(), reverse=True)
        else:
            self.target_centrality_in_avg = 0
            self.target_centrality_out_avg = 0
            self.target_centrality_in_dist = []
            self.target_centrality_out_dist = []
    
    def _convert_to_grakel(self, graph):
        """Converts a NetworkX graph to a Grakel-compatible format."""
        edge_list = list(graph.edges())  # Extract edges
        nodes = sorted(graph.nodes())    # Ensure consistent node ordering

        # Convert np.int64 keys to standard int
        nodes = [int(n) for n in nodes]

        # If the graph is empty, add a self-loop at node 0
        if not edge_list:
            if not nodes:  # No nodes at all, create a dummy node
                nodes = [0]
            edge_list = [(nodes[0], nodes[0])]

        # Assign unique string labels to every node
        node_labels = {int(node): str(i) for i, node in enumerate(nodes)}

        return Graph(edge_list, node_labels=node_labels)
    
    def _sigmoid_similarity(self, actual, target, tolerance):
        """Calculate similarity using sigmoid function for smoother gradient"""
        diff = abs(actual - target)
        
        # Use logistic function, but handle extreme values to prevent overflow
        if diff - tolerance > 35:
            # When x is large, 1/(1+exp(x)) approaches 0
            return 1e-15
        elif diff - tolerance < -35:
            # When x is very negative, 1/(1+exp(x)) approaches 1
            return 1.0
        else:
            # Normal calculation for reasonable values
            return 1.0 / (1.0 + np.exp(diff - tolerance))
    
    def _vector_similarity(self, actual_vec, target_vec, tolerance):
        """Compare two vectors (e.g., degree distributions) with padding"""
        # Pad shorter vector with zeros
        max_len = max(len(actual_vec), len(target_vec))
        if max_len == 0:
            return 1.0  # Perfect match for empty vectors
            
        actual_padded = list(actual_vec) + [0] * (max_len - len(actual_vec))
        target_padded = list(target_vec) + [0] * (max_len - len(target_vec))
        
        # Calculate differences with position-dependent weighting
        # Items at beginning of sorted lists are more important
        total_diff = 0
        total_weight = 0
        for i, (a, t) in enumerate(zip(actual_padded, target_padded)):
            # Position-dependent weight (higher positions are more important)
            weight = 1.0 / (1.0 + 0.5 * i)
            total_diff += weight * abs(a - t)
            total_weight += weight
        
        avg_diff = total_diff / total_weight if total_weight > 0 else 0
        return np.exp(-avg_diff / tolerance)
    
    def _evaluate_size_similarity(self, G):
        """Level 1: Evaluate basic size similarity"""
        # Node count similarity
        node_similarity = self._sigmoid_similarity(
            G.number_of_nodes(), 
            self.target_node_count, 
            self.targets.get('node_count_tolerance', 2)
        )
        
        # Edge count similarity
        edge_similarity = self._sigmoid_similarity(
            G.number_of_edges(), 
            self.target_edge_count, 
            self.targets.get('connection_tolerance', 5)
        )
        
        # Connected components similarity
        if G.number_of_nodes() == 0 and self.target_node_count == 0:
            # Both graphs are empty - perfect match
            connectivity_similarity = 1.0
        elif G.number_of_nodes() == 0 or self.target_node_count == 0:
            # One graph is empty but the other is not
            connectivity_similarity = 0.1
        else:
            try:
                weakly_connected = 1 if nx.is_weakly_connected(G) else 0
                connectivity_similarity = 1.0 - 0.5 * abs(weakly_connected - self.target_weakly_connected)
            except nx.NetworkXPointlessConcept:
                # For cases where connectivity is undefined but should be handled
                connectivity_similarity = 0.5
        
        # Combine with balanced weights
        return 0.4 * node_similarity + 0.4 * edge_similarity + 0.2 * connectivity_similarity
    
    def _evaluate_degree_similarity(self, G):
        """Level 2: Evaluate degree distribution similarity"""
        if G.number_of_nodes() == 0:
            return 0.1  # Minimal score for empty graphs
            
        # In-degree distribution similarity
        in_degrees = sorted([G.in_degree(n) for n in G.nodes()], reverse=True)
        in_degree_similarity = self._vector_similarity(
            in_degrees, 
            self.target_in_degrees,
            self.targets.get('connection_tolerance', 2)
        )
        
        # Out-degree distribution similarity
        out_degrees = sorted([G.out_degree(n) for n in G.nodes()], reverse=True)
        out_degree_similarity = self._vector_similarity(
            out_degrees, 
            self.target_out_degrees,
            self.targets.get('connection_tolerance', 2)
        )
        
        # Combination of in and out distributions
        return 0.5 * in_degree_similarity + 0.5 * out_degree_similarity
    
    def _evaluate_motif_similarity(self, G):
        """Level 3: Evaluate similarity of local motifs and patterns"""
        if G.number_of_nodes() < 2:
            return 0.1  # Minimal score for graphs too small for motifs
            
        # Reciprocity comparison
        reciprocity = nx.overall_reciprocity(G) if G.number_of_edges() > 0 else 0
        reciprocity_similarity = 1.0 - abs(reciprocity - self.target_reciprocity)
        
        # Triangle count comparison
        triangles = sum(nx.triangles(G.to_undirected()).values()) / 3
        triangle_similarity = self._sigmoid_similarity(
            triangles,
            self.target_triangles,
            self.targets.get('motif_tolerance', 2)
        )
        
        # Centrality distribution similarity
        if G.number_of_nodes() > 0:
            in_centrality = nx.in_degree_centrality(G)
            out_centrality = nx.out_degree_centrality(G)
            
            in_centrality_avg = sum(in_centrality.values()) / len(in_centrality)
            out_centrality_avg = sum(out_centrality.values()) / len(out_centrality)
            
            centrality_avg_similarity = 1.0 - 0.5 * (
                abs(in_centrality_avg - self.target_centrality_in_avg) + 
                abs(out_centrality_avg - self.target_centrality_out_avg)
            )
            
            in_centrality_dist = sorted(in_centrality.values(), reverse=True)
            out_centrality_dist = sorted(out_centrality.values(), reverse=True)
            
            centrality_dist_similarity = 0.5 * (
                self._vector_similarity(in_centrality_dist, self.target_centrality_in_dist, 0.1) +
                self._vector_similarity(out_centrality_dist, self.target_centrality_out_dist, 0.1)
            )
        else:
            centrality_avg_similarity = 0.1
            centrality_dist_similarity = 0.1
        
        # Combine motif similarities
        return (0.25 * reciprocity_similarity + 
                0.25 * triangle_similarity + 
                0.25 * centrality_avg_similarity + 
                0.25 * centrality_dist_similarity)
    
    def _evaluate_structural_similarity(self, G):
        """Level 4: Evaluate global structural similarity using graph kernel"""
        # Handle empty graph cases explicitly
        if self.target_graph.number_of_nodes() == 0 and G.number_of_nodes() == 0:
            return 1.0  # Perfect match for empty graphs
        
        if G.number_of_nodes() == 0 or self.target_graph.number_of_nodes() == 0:
            return 0.1  # Minimal score for one empty graph vs. non-empty
            
        # Convert actual graph to GraKel format
        actual_grakel = self._convert_to_grakel(G)
        
        # Compute Weisfeiler-Lehman Kernel similarity
        wl_kernel = WeisfeilerLehman(n_iter=self.wl_iterations, normalize=True)
        try:
            similarity_matrix = wl_kernel.fit_transform([self.target_grakel, actual_grakel])
            similarity = similarity_matrix[0, 1]  # Extract similarity score
        except Exception:
            # Fallback if kernel calculation fails
            similarity = 0.1
        
        return similarity
    
    def _evaluate(self, grid):
        """Evaluate graph similarity using the hierarchical approach"""
        # Create graph from grid's connectivity
        actual_graph = grid.get_graph();
        
        # Special case for empty graphs matching
        if self.target_graph.number_of_nodes() == 0 and actual_graph.number_of_nodes() == 0:
            return 1.0  # Perfect match for empty graphs
        
        # Check for graph isomorphism for perfectly identical graphs
        if (self.target_graph.number_of_nodes() == actual_graph.number_of_nodes() and
            self.target_graph.number_of_edges() == actual_graph.number_of_edges()):
            # Try faster tests first before full isomorphism check
            if sorted([d for _, d in self.target_graph.in_degree()]) == sorted([d for _, d in actual_graph.in_degree()]) and \
               sorted([d for _, d in self.target_graph.out_degree()]) == sorted([d for _, d in actual_graph.out_degree()]):
                # For smaller graphs, we can do full isomorphism checking
                # For larger graphs, this might be too expensive
                if self.target_graph.number_of_nodes() <= 20:  # Only do expensive check for reasonable sized graphs
                    try:
                        if nx.is_isomorphic(self.target_graph, actual_graph):
                            return 1.0  # Perfect match for isomorphic graphs
                    except:
                        pass  # Continue with normal evaluation if isomorphism check fails
        
        # Hierarchical evaluation of similarity levels
        size_similarity = self._evaluate_size_similarity(actual_graph)
        degree_similarity = self._evaluate_degree_similarity(actual_graph)
        motif_similarity = self._evaluate_motif_similarity(actual_graph)
        
        # Only compute structural similarity if needed (more expensive)
        if self.progression_weights['structure_weight'] > 0:
            structure_similarity = self._evaluate_structural_similarity(actual_graph)
        else:
            structure_similarity = 0
        
        # Weighted combination based on progression weights
        fitness = (
            self.progression_weights['size_weight'] * size_similarity +
            self.progression_weights['degree_weight'] * degree_similarity +
            self.progression_weights['motif_weight'] * motif_similarity +
            self.progression_weights['structure_weight'] * structure_similarity
        )
        
        # Progressive scaling - rewards achieving simpler objectives first
        # This helps create a smoother gradient in the fitness landscape
        size_threshold = 0.7  # Threshold for good size similarity
        degree_threshold = 0.6  # Threshold for good degree similarity
        
        if size_similarity < size_threshold:
            # If size is very different, focus mostly on size
            fitness = 0.8 * size_similarity + 0.2 * fitness
        elif degree_similarity < degree_threshold:
            # If size is ok but degree distribution is different, focus on that
            fitness = 0.3 * size_similarity + 0.5 * degree_similarity + 0.2 * fitness
        
        return fitness

class GymFitnessFunction(BaseFitnessFunction):
    """Evaluates fitness based on performance in Gymnasium environments"""
    
    # Environment-specific passing scores
    PASSING_SCORES = {
        'CartPole-v1': 195.0,  # OpenAI Gym standard
        'LunarLander-v3': 200.0,  # OpenAI Gym standard
        'Acrobot-v1': -100.0,  # OpenAI Gym standard
        'MountainCar-v0': -110.0,  # OpenAI Gym standard
        'MountainCarContinuous-v0': 90.0,  # OpenAI Gym standard
        'Pendulum-v1': -150.0,  # OpenAI Gym standard
    }
    
    def __init__(self, targets, penalize_morphogens=False, penalize_steps=False, penalize_dimensions=False, penalize_connections=False,
                 min_connection_fitness=0.8, max_unpenalized_connections=50, connection_half_decay=1000):
        """
        Initialize with environment specification.
        
        Args:
            targets: Dict containing:
                - env_name: Name of the Gymnasium environment
                - num_rollouts: Number of rollouts to average for fitness (default: 5)
            penalize_morphogens: Whether to penalize excess morphogens
            penalize_steps: Whether to penalize excess simulation steps
            penalize_dimensions: Whether to penalize excess grid dimensions
            penalize_connections: Whether to penalize excess connections
            min_connection_fitness: Asymptotic minimum fitness for connection penalty
            max_unpenalized_connections: Maximum number of connections before penalty starts
            connection_half_decay: Number of connections where penalty reaches (1+α)/2
        """
        super().__init__(targets, penalize_morphogens=penalize_morphogens, 
                        penalize_steps=penalize_steps, penalize_dimensions=penalize_dimensions,
                        penalize_connections=penalize_connections,
                        min_connection_fitness=min_connection_fitness,
                        max_unpenalized_connections=max_unpenalized_connections,
                        connection_half_decay=connection_half_decay)
        
        # Get environment name from targets
        self.env_name = targets.get('env_name')
        if not self.env_name:
            raise ValueError("env_name must be specified in targets dictionary")
            
        # Get number of rollouts from targets (default: 5)
        self.num_rollouts = targets.get('num_rollouts', 5)
        
        # Get passing score for environment
        self.passing_score = self.PASSING_SCORES.get(self.env_name)
        if self.passing_score is None:
            raise ValueError(f"Unknown environment {self.env_name}. Add passing score to PASSING_SCORES dict.")
        
        # Get seed from targets (default: None)
        self.seed = targets.get('seed')
        
        # Create environment instance to get dimensions
        self.env = None
        try:
            import gymnasium as gym
            self.env = gym.make(self.env_name, render_mode=None)
            self.input_dim, self.output_dim = self._get_network_dimensions()
            self.env.close()
            self.env = None
        except ImportError:
            raise ImportError("Gymnasium is required for GymFitnessFunction")
    
    def _get_network_dimensions(self) -> tuple:
        """Get required input and output dimensions for the environment."""
        import gymnasium as gym
        
        if self.env is None:
            return 0, 0
            
        # Handle observation space
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            input_dim = self.env.observation_space.n
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            # For Box spaces, use the flattened shape
            input_dim = int(np.prod(self.env.observation_space.shape))
        else:
            raise ValueError(f"Unsupported observation space type: {type(self.env.observation_space)}")
        
        # Handle action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            output_dim = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # For Box spaces, use the flattened shape
            output_dim = int(np.prod(self.env.action_space.shape))
        else:
            raise ValueError(f"Unsupported action space type: {type(self.env.action_space)}")
        
        return input_dim, output_dim
    
    def _evaluate(self, grid):
        """Evaluate fitness by running multiple rollouts in the environment."""
        import gymnasium as gym
        from .neural_propagation import NeuralPropagator
        
        # Create graph from grid's connectivity
        G = grid.get_graph()
        
        # Check if graph has enough nodes for input and output dimensions
        if G.number_of_nodes() < (self.input_dim + self.output_dim):
            return 0.0
        
        # Create propagator with correct dimensions
        propagator = NeuralPropagator(
            G=G,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            activation_function=NeuralPropagator.tanh_activation,
            extra_thinking_time=2,
            additive_update=False
        )
        
        # Create environment with seed if provided
        env = gym.make(self.env_name, render_mode=None)
        if self.seed is not None:
            env.reset(seed=self.seed)
            env.action_space.seed(self.seed)
        
        # Run multiple rollouts
        total_rewards = []
        for rollout in range(self.num_rollouts):
            # Reset network state between episodes so each rollout starts fresh
            # (the seed-paper evaluation carried state across rollouts).
            propagator.reset()
            # Use different seeds for each rollout if base seed is provided
            if self.seed is not None:
                rollout_seed = self.seed + rollout
                observation, _ = env.reset(seed=rollout_seed)
            else:
                observation, _ = env.reset()
                
            total_reward = 0
            done = False
            
            while not done:
                # Preprocess observation
                if isinstance(env.observation_space, gym.spaces.Discrete):
                    # One-hot encode discrete observations
                    obs = np.zeros(propagator.input_dim)
                    obs[observation] = 1
                else:
                    # For continuous observations, ensure correct shape
                    obs = np.array(observation).flatten()
                    if len(obs) != propagator.input_dim:
                        raise ValueError(f"Observation dimension mismatch. Expected {propagator.input_dim}, got {len(obs)}")
                
                # Propagate through network
                propagator.propagate(obs)
                
                # Get action from output neurons
                if isinstance(env.action_space, gym.spaces.Discrete):
                    output_values = propagator.get_output()
                    action = int(output_values.argmax().item())
                else:
                    action = propagator.get_output().cpu().numpy()
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # Take step in environment
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            total_rewards.append(total_reward)
        
        env.close()
        
        # Calculate average reward
        avg_reward = np.mean(total_rewards)
        
        # Calculate fitness using sigmoid function
        fitness = 1.0 / (1.0 + np.exp(-(avg_reward - self.passing_score) / 10.0))
        
        return fitness