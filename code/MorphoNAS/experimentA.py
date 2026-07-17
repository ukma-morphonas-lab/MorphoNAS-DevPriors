import json
import os
import random
import networkx as nx
from pathlib import Path
import argparse
import numpy as np
import glob
from .experiment_runner import ExperimentRunner
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'

def generate_configs(stub_path: str, num_configs: int = 10, max_retries: int = 10, seed: int = None, include_degrees: bool = False):
    """
    Generate multiple configurations based on a stub config file.
    The new configs will have different graph properties sampled from random directed graphs.
    
    Args:
        stub_path: Path to the STUB.json file
        num_configs: Number of configs to generate (default: 10)
        max_retries: Maximum number of retries if graph has no source nodes (default: 10)
        seed: Random seed for reproducibility (default: None)
        include_degrees: Whether to include indegree and outdegree vectors in fitness_targets (default: False)
    """
    # Set random seeds if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Read the stub configuration
    with open(stub_path, 'r') as f:
        stub_config = json.load(f)
    
    # Get the directory of the stub file
    config_dir = os.path.dirname(stub_path)
    
    # Define the full range of possible network sizes
    min_neurons = 8
    max_neurons = 32
    
    # Generate configurations
    for i in range(1, num_configs + 1):
        # Uniformly select number of neurons across the full range
        num_neurons = random.randint(min_neurons, max_neurons)
        
        # Calculate proportional number of connections using density
        # Using much lower density to increase probability of source nodes
        # Minimum density is set to ensure average outdegree >= 1.5
        min_density = 1.5 / (num_neurons - 1)  # This ensures num_connections >= 1.5 * num_neurons
        max_density = 0.5  # 50% maximum edge density
        density = random.uniform(min_density, max_density)
        
        # Calculate max possible connections for the given number of neurons
        max_possible_connections = num_neurons * (num_neurons - 1)  # For directed graph
        
        # Calculate number of connections based on density
        num_connections = int(density * max_possible_connections)
        
        # Ensure minimum number of connections for connectivity and average outdegree
        min_connections = int(1.5 * num_neurons)  # Minimum edges needed for avg outdegree >= 1.5
        num_connections = max(num_connections, min_connections)
        
        # Try to generate a graph with at least one source node and ensure it's weakly connected
        for attempt in range(max_retries):
            # Create a random directed graph with the randomized values
            G = nx.gnm_random_graph(num_neurons, num_connections, directed=True)
            
            # Calculate graph statistics
            no_incoming = sum(1 for node in G.nodes() if G.in_degree(node) == 0)
            
            # Check if graph is weakly connected and has at least one source node
            if no_incoming > 0 and nx.is_weakly_connected(G):
                break
        
        avg_degree = (2 * num_connections) / num_neurons  # For directed graph, total degree is 2 * num_edges / num_nodes
        
        # Create new config by copying the stub
        new_config = stub_config.copy()
        
        # Update the fitness targets with the randomized values
        new_config['fitness_targets']['neurons'] = num_neurons
        new_config['fitness_targets']['connections'] = num_connections
        new_config['fitness_targets']['no_incoming'] = no_incoming
        
        # Add degree vectors if requested
        if include_degrees:
            # Extract indegree and outdegree for each node
            indegrees = [G.in_degree(node) for node in sorted(G.nodes())]
            outdegrees = [G.out_degree(node) for node in sorted(G.nodes())]
            
            new_config['fitness_targets']['indegrees'] = indegrees
            new_config['fitness_targets']['outdegrees'] = outdegrees
        
        # Generate the output filename (K01.json, K02.json, etc.)
        output_filename = f"K{i:02d}.json"
        output_path = os.path.join(config_dir, output_filename)
        
        # Save the new configuration
        with open(output_path, 'w') as f:
            json.dump(new_config, f, indent=4)
        
        degree_info = " (with degree vectors)" if include_degrees else ""
        print(f"Generated config {i}: neurons={num_neurons}, connections={num_connections}, "
              f"no_incoming={no_incoming}, density={density:.3f}, avg_degree={avg_degree:.1f}{degree_info}")

def run_experiments(config_dir: str, num_runs: int = 20, experiment_seed: int = None):
    """
    Run multiple experiments for each K*.json config with different seeds.
    
    Args:
        config_dir: Directory containing K*.json configs
        num_runs: Number of runs per config (default: 20)
        experiment_seed: Seed for generating experiment seeds (default: None)
    """
    # Set random seed for experiment seeds if provided
    if experiment_seed is not None:
        random.seed(experiment_seed)
    
    # Generate experiment seeds
    experiment_seeds = [random.randint(1, 1000000) for _ in range(num_runs)]
    
    # Get all K*.json configs
    config_files = sorted(glob.glob(os.path.join(config_dir, "K*.json")))
    
    if not config_files:
        print(f"No K*.json configs found in {config_dir}")
        return
    
    print(f"Found {len(config_files)} config files")
    print(f"Will run {num_runs} experiments for each config")
    print(f"Experiment seeds: {experiment_seeds}")
    
    # Run experiments for each config
    for config_file in config_files:
        # Extract base config name (e.g., "K01" from "K01.json" or "K01_R01.json")
        base_name = os.path.splitext(os.path.basename(config_file))[0]
        if "_R" in base_name:
            base_name = base_name.split("_R")[0]
        
        print(f"\nProcessing {base_name}...")
        
        # Load the config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Run experiments with different seeds
        for run_idx, seed in enumerate(experiment_seeds, 1):
            # Create a copy of the config with the new seed
            run_config = config.copy()
            run_config['ga_params']['seed'] = seed
            
            # Create a config file for this run
            run_config_name = f"{base_name}_R{run_idx:02d}"
            run_config_path = os.path.join(config_dir, f"{run_config_name}.json")
            
            # Save the run config
            with open(run_config_path, 'w') as f:
                json.dump(run_config, f, indent=4)
            
            print(f"\nRunning experiment {run_idx}/{num_runs} for {base_name} with seed {seed}")
            
            # Run the experiment
            try:
                runner = ExperimentRunner(run_config_path)
                runner.run()
            except Exception as e:
                print(f"Error running experiment: {e}")

def main():
    parser = argparse.ArgumentParser(description='Experiment A configuration generator and runner')
    parser.add_argument('--generate-configs', action='store_true',
                      help='Generate configurations from STUB.json')
    parser.add_argument('--run-experiments', action='store_true',
                      help='Run experiments for generated configs')
    parser.add_argument('--stub-path', type=str,
                      default='experiments/_ExpA_graph_properties/configs/STUB.json',
                      help='Path to the STUB.json file')
    parser.add_argument('--num-configs', type=int, default=20,
                      help='Number of configurations to generate')
    parser.add_argument('--max-retries', type=int, default=10,
                      help='Maximum number of retries if graph has no source nodes')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-runs', type=int, default=20,
                      help='Number of runs per config')
    parser.add_argument('--experiment-seed', type=int, default=None,
                      help='Seed for generating experiment seeds')
    parser.add_argument('--include-degrees', action='store_true',
                      help='Include indegree and outdegree vectors in fitness_targets')
    
    args = parser.parse_args()
    
    if args.generate_configs:
        generate_configs(args.stub_path, args.num_configs, args.max_retries, args.seed, args.include_degrees)
        print(f"Generated {args.num_configs} configurations in {os.path.dirname(args.stub_path)}")
    
    if args.run_experiments:
        config_dir = os.path.dirname(args.stub_path)
        run_experiments(config_dir, args.num_runs, args.experiment_seed)
    
    if not (args.generate_configs or args.run_experiments):
        parser.print_help()

if __name__ == "__main__":
    main()
