import numpy as np
import networkx as nx
import gymnasium as gym
from typing import Callable, Tuple, Union, List, Dict
import torch

class NeuralPropagator:
    def __init__(
        self,
        G: nx.DiGraph,
        input_dim: int,
        output_dim: int,
        activation_function: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_thinking_time: int = 0,
        additive_update: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the neural propagator with a given network structure.
        
        Args:
            G: NetworkX DiGraph representing the neural network
            input_dim: Number of input neurons
            output_dim: Number of output neurons
            activation_function: Function to apply to neuron outputs (default: tanh)
            extra_thinking_time: Additional thinking time beyond graph diameter
            additive_update: Whether to add or replace neuron states
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Create a mapping from original node indices to 0-based indices
        self.node_mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
        
        # Reorder nodes by in-degree (ascending)
        self.node_order = sorted(G.nodes(), key=lambda x: G.in_degree(x))
        self.G = nx.DiGraph()
        
        # Create new graph with reordered nodes using 0-based indices
        for node in self.node_order:
            self.G.add_node(self.node_mapping[node])
        
        # Add edges with original weights, using mapped indices
        for u, v, data in G.edges(data=True):
            self.G.add_edge(self.node_mapping[u], self.node_mapping[v], weight=data['weight'])
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function or self.tanh_activation
        
        # Calculate graph diameter and set thinking time
        try:
            self.graph_diameter = nx.diameter(self.G)
        except nx.NetworkXError:
            # If graph is not strongly connected, use the maximum shortest path length
            max_length = 0
            for source in self.G.nodes():
                lengths = nx.shortest_path_length(self.G, source)
                max_length = max(max_length, max(lengths.values()))
            self.graph_diameter = max_length
        
        self.network_thinking_time = self.graph_diameter + extra_thinking_time
        self.additive_update = additive_update
        
        # Input nodes are now simply the first input_dim nodes
        self.input_nodes = [self.node_mapping[node] for node in self.node_order[:input_dim]]
        
        # Convert graph to weight matrix
        self.W = self._get_weight_matrix()
        
        # Initialize network state
        self.network_state = torch.zeros(len(G), device=self.device)
    
    def _select_input_nodes(self) -> List[int]:
        """
        Get the input nodes for the network.
        Since nodes are ordered by in-degree, input nodes are simply the first input_dim nodes.
        
        Returns:
            List of node indices to be used as input nodes
        """
        return self.node_order[:self.input_dim]
    
    @staticmethod
    def tanh_activation(x: torch.Tensor) -> torch.Tensor:
        """Tanh activation function."""
        return torch.tanh(x)
    
    @staticmethod
    def relu_activation(x: torch.Tensor) -> torch.Tensor:
        """ReLU activation function."""
        return torch.relu(x)
    
    def _get_weight_matrix(self) -> torch.Tensor:
        """Convert NetworkX graph to weight matrix."""
        num_neurons = len(self.G)
        W = torch.zeros((num_neurons, num_neurons), device=self.device)
        
        # Create a mapping from node indices to matrix indices
        node_to_idx = {node: idx for idx, node in enumerate(sorted(self.G.nodes()))}
        
        for i, j, data in self.G.edges(data=True):
            # W[post, pre]: rows index the postsynaptic neuron (edge target),
            # so matmul(W, state) sums each neuron's presynaptic inputs (forward flow).
            pre_idx = node_to_idx[i]   # edge source
            post_idx = node_to_idx[j]  # edge target
            W[post_idx, pre_idx] = data['weight']
        
        return W
    
    def propagate(self, input_values: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Propagate input values through the network.
        
        Args:
            input_values: Input values to propagate (shape: [input_dim])
            
        Returns:
            Network state after propagation
        """
        if isinstance(input_values, np.ndarray):
            input_values = torch.from_numpy(input_values).to(self.device)
            
        if len(input_values) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {len(input_values)}")
        
        # Set input neurons (using selected input nodes)
        for i, node_idx in enumerate(self.input_nodes):
            self.network_state[node_idx] = input_values[i]
        
        # Propagate through network
        current_state = self.network_state.clone()
        
        for _ in range(self.network_thinking_time):
            # Calculate new state
            new_state = torch.matmul(self.W, current_state)
            
            # Apply activation function
            new_state = self.activation_function(new_state)
            
            # Update state
            if self.additive_update:
                current_state += new_state
            else:
                current_state = new_state
        
        self.network_state = current_state
        return current_state
    
    def get_output(self) -> torch.Tensor:
        """Get the current output values from the network."""
        return self.network_state[-self.output_dim:]
    
    def reset(self):
        """Reset the network state to zeros."""
        self.network_state = torch.zeros(len(self.G), device=self.device)
    
    def get_input_nodes_info(self) -> dict:
        """
        Get information about the selected input nodes.
        
        Returns:
            Dictionary containing:
            - total_input_nodes: Total number of input nodes
            - zero_in_degree_nodes: Number of nodes without inbound edges
            - selected_nodes: List of selected input node indices
            - node_in_degrees: Dictionary mapping node indices to their in-degrees
        """
        zero_in_degree_nodes = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
        return {
            'total_input_nodes': len(self.input_nodes),
            'zero_in_degree_nodes': len(zero_in_degree_nodes),
            'selected_nodes': self.input_nodes,
            'node_in_degrees': {node: self.G.in_degree(node) for node in self.input_nodes}
        }

class GymEnvironment:
    def __init__(self, env_name: str):
        """
        Initialize the neural environment interface.
        
        Args:
            env_name: Name of the Gymnasium environment
        """
        self.env_name = env_name
        self.input_dim, self.output_dim = self._get_network_dimensions()
        self.env = None
    
    def _get_network_dimensions(self) -> Tuple[int, int]:
        """
        Get the required input and output dimensions for the environment.
        
        Returns:
            Tuple of (input_dim, output_dim)
        """
        env = gym.make(self.env_name)
        
        # Handle observation space
        if isinstance(env.observation_space, gym.spaces.Discrete):
            input_dim = env.observation_space.n
        elif isinstance(env.observation_space, gym.spaces.Box):
            input_dim = np.prod(env.observation_space.shape)
        else:
            raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")
        
        # Handle action space
        if isinstance(env.action_space, gym.spaces.Discrete):
            output_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            output_dim = np.prod(env.action_space.shape)
        else:
            raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
        
        env.close()
        return input_dim, output_dim
    
    def rollout(
        self,
        propagator: NeuralPropagator,
        render: bool = False,
        seed: int = None
    ) -> float:
        """
        Run an episode in the environment using the neural network for decision making.
        
        Args:
            propagator: NeuralPropagator instance
            render: Whether to render the environment
            seed: Random seed for environment (default: None for random)
            
        Returns:
            Total reward for the episode
        """
        self.env = gym.make(self.env_name, render_mode="human" if render else None)
        
        # Set seed if provided
        if seed is not None:
            self.env.reset(seed=seed)
        
        # Run episode
        observation, _ = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Preprocess observation
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                # One-hot encode discrete observations
                obs = torch.zeros(propagator.input_dim, device=propagator.device)
                obs[observation] = 1
            else:
                obs = torch.from_numpy(observation.flatten()).to(propagator.device)
            
            # Propagate through network
            propagator.propagate(obs)
            
            # Get action from output neurons
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # For discrete action spaces, we need to ensure the output dimension matches the number of actions
                if propagator.output_dim != self.env.action_space.n:
                    raise ValueError(f"Output dimension ({propagator.output_dim}) must match number of actions ({self.env.action_space.n})")
                
                # Get the output values and select the action with highest activation
                output_values = propagator.get_output()
                action = int(output_values.argmax().item())  # Convert to int for gym
                
                if render:
                    print(f"Output values: {output_values.cpu().numpy()}, Selected action: {action}")
            else:
                action = propagator.get_output().cpu().numpy()
                # Clip actions to valid range
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            # Take step in environment
            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        self.env.close()
        self.env = None
        return total_reward
    
    def visualize_network(self, G: nx.DiGraph, propagator: NeuralPropagator) -> None:
        """
        Visualize the neural network structure.
        
        Args:
            G: NetworkX DiGraph
            propagator: NeuralPropagator instance to get input nodes information
        """
        import matplotlib.pyplot as plt
        
        # Check if graph is empty
        if len(G.nodes()) == 0:
            print("Warning: Cannot visualize empty graph")
            return
            
        # Check if graph has edges
        if len(G.edges()) == 0:
            print("Warning: Graph has no edges, adding self-loops for visualization")
            # Add self-loops to ensure graph is connected
            for node in G.nodes():
                G.add_edge(node, node, weight=0.1)
        
        # Create a copy of the graph with sequential node indices
        G_viz = nx.DiGraph()
        node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(sorted(G.nodes()))}
        
        # Add nodes with new indices
        for old_node in G.nodes():
            G_viz.add_node(node_mapping[old_node])
        
        # Add edges with new indices
        for u, v, data in G.edges(data=True):
            G_viz.add_edge(node_mapping[u], node_mapping[v], weight=data['weight'])
        
        try:
            # Use a more stable layout algorithm
            pos = nx.kamada_kawai_layout(G_viz)
        except nx.NetworkXError:
            try:
                # Fallback to spring layout with more iterations
                pos = nx.spring_layout(G_viz, k=1, iterations=50)
            except nx.NetworkXError:
                print("Error: Could not compute node positions. Graph may be disconnected.")
                return
        
        plt.figure(figsize=(10, 8))
        
        # Get the mapped indices for input, hidden, and output nodes
        input_nodes = propagator.input_nodes
        hidden_nodes = [node for node in G_viz.nodes() 
                       if node not in input_nodes and node < len(G)-propagator.output_dim]
        output_nodes = [node for node in G_viz.nodes() 
                       if node >= len(G)-propagator.output_dim]
        
        # Draw edges first (so they appear behind nodes)
        nx.draw_networkx_edges(G_viz, pos, edge_color='gray', alpha=0.5, arrows=True)
        
        # Draw nodes with different colors
        if input_nodes:
            nx.draw_networkx_nodes(G_viz, pos, nodelist=input_nodes, node_color='blue', label='Input')
        if hidden_nodes:
            nx.draw_networkx_nodes(G_viz, pos, nodelist=hidden_nodes, node_color='green', label='Hidden')
        if output_nodes:
            nx.draw_networkx_nodes(G_viz, pos, nodelist=output_nodes, node_color='red', label='Output')
        
        # Add node labels showing original node indices
        labels = {new_idx: str(old_node) for old_node, new_idx in node_mapping.items()}
        nx.draw_networkx_labels(G_viz, pos, labels=labels)
        
        plt.legend()
        plt.title("Neural Network Structure")
        plt.axis('off')  # Hide axes
        plt.show() 