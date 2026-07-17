import os
import time
import json
import zlib
import numpy as np
import networkx as nx
import select
import sys
from .genetic_algorithm import GeneticAlgorithm
from .cma_optimizer import CMAESOptimizer
from .genome import Genome
from .grid import Grid
from .fitness_functions import NetworkFitnessFunction, TargetGraphFitnessFunction, StructuralGraphFitnessFunction, HierarchicalGraphFitnessFunction, GymFitnessFunction
import matplotlib.pyplot as plt
import traceback
from .genome_strategies import (
    DefaultMutationStrategy, AggressiveMutationStrategy,
    DefaultMetaParametersStrategy, ExtendedMatrixMutationStrategy,
    DefaultCrossoverStrategy, BlockPreservationCrossoverStrategy,
    AdaptiveMutationStrategy
)
from .convergence_strategies import DefaultConvergenceStrategy, DiversityMaintenanceStrategy
from .selection_strategies import TopPercentageSelectionStrategy, TournamentSelectionStrategy, TopFitnessSelectionStrategy, TopDistinctFitnessSelectionStrategy
from .selection_strategies import CombinedSelectionStrategy

class ExperimentRunner:
    """Manages genetic algorithm experiments with configurable parameters."""
    
    def __init__(self, config_path, force=False, rerun_on_max_workers_change=False):
        """
        Initialize experiment runner with configuration.
        
        Args:
            config_path (str): Path to JSON configuration file
            force (bool): If True, run experiment even if config unchanged
            rerun_on_max_workers_change (bool): If True, re-run when max_workers changes (default: False)
        """
        self.config = self._load_config(config_path)
        self.config_dir = os.path.dirname(config_path)
        self.start_time = None
        self.experiment_name = os.path.splitext(os.path.basename(config_path))[0]
        self.generation_stats = []
        self.force = force
        self.rerun_on_max_workers_change = rerun_on_max_workers_change
        
        # Get fitness threshold from config or use default of 1.0
        self.fitness_threshold = self.config.get('fitness_threshold', 1.0)
        
        # Get optimizer type from config
        self.optimizer_type = self.config.get('optimizer_type', 'genetic_algorithm')
        
        # Initialize fitness function
        fitness_class = globals()[self.config.get('fitness_function', 'NetworkFitnessFunction')]
        
        # Prepare common parameters
        fitness_params = {
            'targets': {
                **self.config['fitness_targets'],
                'seed': self.config['ga_params'].get('seed')  # Add seed from config
            },
            'penalize_morphogens': self.config.get('penalize_morphogens', False),
            'penalize_steps': self.config.get('penalize_steps', False),
            'penalize_dimensions': self.config.get('penalize_dimensions', False),
            'penalize_connections': self.config.get('penalize_connections', False)
        }
        
        # Add GymFitnessFunction-specific parameters only if using that function
        if self.config.get('fitness_function') == 'GymFitnessFunction':
            fitness_params.update({
                'min_connection_fitness': self.config.get('min_connection_fitness', 0.8),
                'max_unpenalized_connections': self.config.get('max_unpenalized_connections', 50),
                'connection_half_decay': self.config.get('connection_half_decay', 1000)
            })
        
        self.fitness_function = fitness_class(**fitness_params)
        
        # Initialize convergence strategy
        convergence_config = self.config.get('convergence_strategy', {})
        convergence_type = convergence_config.get('type', 'DefaultConvergenceStrategy')
        
        if convergence_type == 'DefaultConvergenceStrategy':
            convergence_threshold = convergence_config.get('convergence_threshold', 0.95)
            self.convergence_strategy = DefaultConvergenceStrategy(convergence_threshold)
        elif convergence_type == 'DiversityMaintenanceStrategy':
            convergence_threshold = convergence_config.get('convergence_threshold', 0.95)
            replacement_fraction = convergence_config.get('replacement_fraction', 0.1)
            bottom_fraction = convergence_config.get('bottom_fraction', 0.5)
            replacement_threshold = convergence_config.get('replacement_threshold', 0.9)
            self.convergence_strategy = DiversityMaintenanceStrategy(
                convergence_threshold=convergence_threshold,
                replacement_fraction=replacement_fraction,
                bottom_fraction=bottom_fraction,
                replacement_threshold=replacement_threshold
            )
        else:
            raise ValueError(f"Unknown convergence strategy type: {convergence_type}")
        
        # Initialize strategies for genetic algorithm
        if self.optimizer_type == 'genetic_algorithm':
            self.mutation_strategy = self._get_strategy_instance(
                self.config.get('mutation_strategy'),
                default=DefaultMutationStrategy,
                strategies={
                    'DefaultMutationStrategy': DefaultMutationStrategy,
                    'AggressiveMutationStrategy': AggressiveMutationStrategy
                }
            )
            
            self.meta_parameters_strategy = self._get_strategy_instance(
                self.config.get('meta_parameters_strategy'),
                default=DefaultMetaParametersStrategy,
                strategies={
                    'DefaultMetaParametersStrategy': DefaultMetaParametersStrategy,
                    'ExtendedMatrixMutationStrategy': ExtendedMatrixMutationStrategy
                }
            )
            
            self.crossover_strategy = self._get_strategy_instance(
                self.config.get('crossover_strategy'),
                default=DefaultCrossoverStrategy,
                strategies={
                    'DefaultCrossoverStrategy': DefaultCrossoverStrategy,
                    'BlockPreservationCrossoverStrategy': BlockPreservationCrossoverStrategy
                }
            )
    
    def _load_config(self, config_path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _should_run_experiment(self):
        """
        Check if the experiment configuration has changed since the last run.
        
        Returns:
            tuple: (should_run, stats_path, genome_path, population_state_path)
                - should_run: True if the experiment should run, False otherwise
                - stats_path: Path to the stats file if it exists, None otherwise
                - genome_path: Path to the best genome file if it exists, None otherwise
                - population_state_path: Path to the population state file if it exists, None otherwise
        """
        # Get path to previously saved stats and genome
        output_dir = os.path.join(self.config_dir, '..', 'results', self.experiment_name)
        stats_path = os.path.join(output_dir, 'stats.json')
        genome_path = os.path.join(output_dir, 'best_genome.json')
        population_state_path = os.path.join(output_dir, 'population_state.json')
        
        # If force is True, always run the experiment
        if self.force:
            print(f"Force option enabled - running experiment '{self.experiment_name}'")
            return True, None, None, None
        
        # Check if we have a population state but no stats (interrupted experiment)
        if os.path.exists(population_state_path) and not os.path.exists(stats_path):
            print(f"Found interrupted experiment state for '{self.experiment_name}'")
            print(f"Resuming from saved population state...")
            return True, None, None, population_state_path
        
        # If either file doesn't exist, we should run the experiment
        if not os.path.exists(stats_path) or not os.path.exists(genome_path):
            return True, None, None, None
            
        # Load previous configuration
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                previous_config = stats.get('configuration', {})
                
            # Compare current and previous configurations
            if self._configs_are_equivalent(previous_config, self.config):
                print(f"Configuration unchanged since last run for experiment '{self.experiment_name}'")
                print(f"Skipping experiment. Previous results available at: {output_dir}")
                return False, stats_path, genome_path, None
            else:
                print(f"Configuration changed since last run for experiment '{self.experiment_name}'")
                # Check if there's a population state for the new configuration
                if os.path.exists(population_state_path):
                    print(f"Found population state for new configuration, resuming...")
                    return True, None, None, population_state_path
                else:
                    print(f"Starting fresh with new configuration...")
                    return True, None, None, None
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error checking previous configuration: {e}")
            # In case of any error reading the previous config, run the experiment to be safe
            return True, None, None, None

    def _configs_are_equivalent(self, config1, config2):
        """
        Compare two configurations, optionally ignoring max_workers changes.
        
        Args:
            config1: First configuration dictionary
            config2: Second configuration dictionary
            
        Returns:
            bool: True if configurations are equivalent, False otherwise
        """
        # If we should ignore max_workers changes, create copies and remove max_workers
        if not self.rerun_on_max_workers_change:
            config1_copy = self._deep_copy_without_max_workers(config1)
            config2_copy = self._deep_copy_without_max_workers(config2)
            return config1_copy == config2_copy
        else:
            # Normal comparison including max_workers
            return config1 == config2

    def _deep_copy_without_max_workers(self, config):
        """
        Create a deep copy of a configuration with max_workers removed.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            dict: Deep copy of config with max_workers removed
        """
        import copy
        config_copy = copy.deepcopy(config)
        
        # Remove max_workers from ga_params if it exists
        if 'ga_params' in config_copy and 'max_workers' in config_copy['ga_params']:
            del config_copy['ga_params']['max_workers']
        
        # Remove max_workers from cma_params if it exists
        if 'cma_params' in config_copy and 'max_workers' in config_copy['cma_params']:
            del config_copy['cma_params']['max_workers']
        
        return config_copy

    def _save_population_state(self, optimizer, generation):
        """Save the current population state to a temporary file."""
        output_dir = os.path.join(self.config_dir, '..', 'results', self.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        population_state_path = os.path.join(output_dir, 'population_state.json')
        
        # Save population state
        state = {
            'generation': generation,
            'population': [genome.to_dict() for genome in optimizer.current_population],
            'fitness_scores': optimizer.current_fitness_scores if isinstance(optimizer.current_fitness_scores, list) else optimizer.current_fitness_scores.tolist(),
            'best_genome': optimizer.best_solution.to_dict(),
            'best_fitness': float(optimizer.best_fitness),
            'evaluation_count': optimizer.evaluation_count,
            'configuration': self.config,
            'fitness_cache': {genome.to_bytes().hex(): float(fitness) for genome, fitness in zip(optimizer.current_population, optimizer.current_fitness_scores)}
        }
        
        with open(population_state_path, 'w') as f:
            json.dump(state, f, indent=4)
        
        print(f"Saved population state (generation {generation}) to: {population_state_path}")

    def _load_population_state(self, optimizer, population_state_path):
        """Load population state from file and restore optimizer state."""
        try:
            with open(population_state_path, 'r') as f:
                state = json.load(f)
            
            # Restore population
            population = []
            for genome_dict in state['population']:
                genome = Genome.from_dict(
                    genome_dict,
                    meta_parameters_strategy=self.meta_parameters_strategy,
                    mutation_strategy=self.mutation_strategy,
                    crossover_strategy=self.crossover_strategy
                )
                population.append(genome)
            
            # Restore optimizer state
            optimizer.population = population
            optimizer.fitness_scores = np.array(state['fitness_scores'])
            optimizer._best_genome = Genome.from_dict(
                state['best_genome'],
                meta_parameters_strategy=self.meta_parameters_strategy,
                mutation_strategy=self.mutation_strategy,
                crossover_strategy=self.crossover_strategy
            )
            optimizer._best_fitness = state['best_fitness']
            optimizer._eval_count = state['evaluation_count']
            optimizer.generation = state['generation']
            
            # Restore fitness cache if available
            if 'fitness_cache' in state:
                # Convert hex keys back to bytes for the cache
                optimizer.fitness_cache = {bytes.fromhex(hex_key): fitness for hex_key, fitness in state['fitness_cache'].items()}
                print(f"Restored fitness cache with {len(optimizer.fitness_cache)} entries")
            else:
                print("No fitness cache found in population state")
            
            print(f"Loaded population state from generation {state['generation']}")
            print(f"Population size: {len(population)}")
            print(f"Best fitness: {state['best_fitness']}")
            print(f"Evaluation count: {state['evaluation_count']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading population state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup_population_state(self):
        """Remove the population state file after successful completion."""
        output_dir = os.path.join(self.config_dir, '..', 'results', self.experiment_name)
        population_state_path = os.path.join(output_dir, 'population_state.json')
        
        if os.path.exists(population_state_path):
            try:
                os.remove(population_state_path)
                print(f"Cleaned up population state file: {population_state_path}")
            except Exception as e:
                print(f"Warning: Could not remove population state file: {e}")
    
    def fitness_function_wrapper(self, genome):
        """Wrapper that runs simulation and evaluates fitness"""
        grid = self._run_simulation(genome)
        return self.fitness_function.evaluate(grid)
    
    def _run_simulation(self, genome, verbose=False):
        """Run simulation with given genome."""
        grid = Grid(genome)
        return grid.run_simulation(verbose=verbose)
    
    def _progress_callback(self, optimizer, generation, best_fitness, avg_fitness):
        """Display progress of evolution and check for user input."""
        elapsed = time.time() - self.start_time
        
        # Update convergence ratio for adaptive mutation strategy if used
        effective_mutation_rate = optimizer.get_mutation_rate()
        if isinstance(optimizer, GeneticAlgorithm) and isinstance(self.mutation_strategy, AdaptiveMutationStrategy):
            self.mutation_strategy.update_convergence(best_fitness, avg_fitness)
            effective_mutation_rate *= self.mutation_strategy.get_mutation_rate_multiplier()
        
        # Get stats about best solution
        best_grid = self._run_simulation(optimizer.best_solution)
        source_indices, target_indices = best_grid.neuron_connections.nonzero()
        
        # Create graph for analysis
        G = best_grid.get_graph();
        nodes_no_incoming = len([node for node in G.nodes() if G.in_degree(node) == 0])
        
        # Count individuals by number of morphogens
        morphogen_counts = {}
        for individual in optimizer.current_population:
            m_count = Grid(individual).num_morphogens
            morphogen_counts[m_count] = morphogen_counts.get(m_count, 0) + 1
        
        # Format morphogen distribution string
        morphogen_dist = ", ".join(f"M{m}: {count}" for m, count in sorted(morphogen_counts.items()))
        
        # Calculate distinct fitness scores and genomes
        distinct_fitness_scores = len(set(optimizer.current_fitness_scores))
        distinct_genomes = len(set(genome.to_bytes() for genome in optimizer.current_population))
        
        # Store generation stats - convert numpy types to native Python types
        gen_stats = {
            'generation': int(generation),
            'best_fitness': float(best_fitness),
            'avg_fitness': float(avg_fitness),
            'elapsed_time': float(elapsed),
            'evaluations': int(optimizer.evaluation_count),
            'distinct_fitness_scores': int(distinct_fitness_scores),
            'distinct_genomes': int(distinct_genomes),
            'dimensions': [int(best_grid.size_x), int(best_grid.size_y)],
            'max_steps': int(best_grid.max_growth_steps),
            'morphogens': int(best_grid.num_morphogens),
            'neurons': int(best_grid.neuron_count()),
            'inputs': int(nodes_no_incoming),
            'connections': int(len(source_indices)),
            'morphogen_distribution': {str(k): int(v) for k, v in morphogen_counts.items()}
        }
        
        # Add mutation rate info if using AdaptiveMutationStrategy
        if isinstance(optimizer, GeneticAlgorithm) and isinstance(self.mutation_strategy, AdaptiveMutationStrategy):
            gen_stats['mutation_rate'] = float(effective_mutation_rate)
            gen_stats['mutation_rate_multiplier'] = float(self.mutation_strategy.get_mutation_rate_multiplier())
        
        self.generation_stats.append(gen_stats)
        
        # Print progress with mutation rate info
        mutation_rate_info = f"MR: {effective_mutation_rate:.2f}" if isinstance(optimizer, GeneticAlgorithm) and isinstance(self.mutation_strategy, AdaptiveMutationStrategy) else ""
        print(f"Generation {generation:3d}: Best Fitness = {best_fitness:.2e}, "
              f"Avg Fitness = {avg_fitness:.2e}, Time = {elapsed:.1f}s, "
              f"Evals = {optimizer.evaluation_count}, "
              f"Fitnesses = {distinct_fitness_scores}, "
              f"Genomes = {distinct_genomes}, "
              f"Dims: ({best_grid.size_x}, {best_grid.size_y}), "
              f"Max Steps: {best_grid.max_growth_steps}, "
              f"M: {best_grid.num_morphogens}, "
              f"N: {best_grid.neuron_count()}, "
              f"INP: {nodes_no_incoming}, "
              f"CON: {len(source_indices)}, "
              f"[{morphogen_dist}] "
              f"{mutation_rate_info}")
        
        # Stop if fitness threshold is achieved
        if best_fitness >= self.fitness_threshold:
            print(f"\nFitness threshold of {self.fitness_threshold} achieved!")
            return False
            
        # Check for user input
        if select.select([sys.stdin], [], [], 0.1)[0]:
            sys.stdin.readline()
            return False
            
        return True
    
    def save_results(self, best_genome, stats):
        """Save experiment results to output directory."""
        output_dir = os.path.join(self.config_dir, '..', 'results', self.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best genome with validation
        genome_path = os.path.join(output_dir, 'best_genome.json')
        try:
            # Validate the genome can be serialized to JSON before saving
            genome_json = best_genome.to_json()
            
            # Write to file
            with open(genome_path, 'w') as f:
                f.write(genome_json)
            
            # Verify the file was written correctly
            if os.path.getsize(genome_path) == 0:
                print(f"Warning: Genome file {genome_path} was saved but is empty")
        except Exception as e:
            print(f"Error saving genome to {genome_path}: {e}")
            import traceback
            traceback.print_exc()
        
        # Add generation stats to overall stats
        stats['generations_data'] = self.generation_stats
        
        # Also save genome data as part of stats as a backup
        try:
            stats['best_genome_data'] = best_genome.to_dict()
        except Exception as e:
            print(f"Warning: Could not add genome data to stats: {e}")
            import traceback
            traceback.print_exc()
        
        # Save statistics
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Verify both files exist and have content
        print(f"Saved best genome to: {genome_path} ({os.path.getsize(genome_path)} bytes)")
        print(f"Saved stats to: {stats_path} ({os.path.getsize(stats_path)} bytes)")
    
    def print_hex_chunks(self, hex_string, description, chunk_size=128, print_size=True, print_bytes=False):
        """Print a hex string in chunks for better readability."""
        size_info = f" ({len(hex_string)//2} bytes)" if print_size else ""
        print(f"{description}{size_info}")
        if print_bytes:
            for i in range(0, len(hex_string), chunk_size):
                print(hex_string[i:i+chunk_size])

    def _print_final_analysis(self, best_genome, final_grid):
        """Print detailed analysis of the best solution."""
        # Print best genome as JSON
        print("\nBest Genome (JSON):")
        print(best_genome.to_json())
        
        # Print genome bytes in hex format
        genome_bytes = best_genome.to_bytes()
        self.print_hex_chunks(genome_bytes.hex(), "Best Genome (hex)")
        
        # Print zipped genome bytes in hex format
        zipped_bytes = zlib.compress(genome_bytes)
        self.print_hex_chunks(zipped_bytes.hex(), "Zipped Genome (hex)")
        print(f"Compression ratio: {len(zipped_bytes)/len(genome_bytes):.2f}")
        
        # Create bytes representation of weights
        weight_bytes = bytearray()
        source_indices, target_indices = final_grid.neuron_connections.nonzero()
        weights = final_grid.neuron_connections[source_indices, target_indices].toarray().flatten()
        
        for source, target, weight in zip(source_indices + 1, target_indices + 1, weights):
            source_int = int(source)
            target_int = int(target)
            weight_bytes.extend(source_int.to_bytes(2, byteorder='big'))
            weight_bytes.extend(target_int.to_bytes(2, byteorder='big'))
            weight_bytes.extend(np.float32(weight).tobytes())
        
        # Print weight bytes in hex format
        self.print_hex_chunks(weight_bytes.hex(), "Weight Data (hex)")
        
        # Print zipped weight bytes in hex format
        zipped_weights = zlib.compress(weight_bytes)
        self.print_hex_chunks(zipped_weights.hex(), "Zipped Weights (hex)")
        print(f"Weight compression ratio: {len(zipped_weights)/len(weight_bytes):.2f}")
        print(f"Number of connections: {len(weights)}")
        
        # Create bytes representation of connections (without weights)
        connection_bytes = bytearray()
        for source, target in zip(source_indices + 1, target_indices + 1):
            source_int = int(source)
            target_int = int(target)
            connection_bytes.extend(source_int.to_bytes(2, byteorder='big'))
            connection_bytes.extend(target_int.to_bytes(2, byteorder='big'))
        
        # Print connection bytes in hex format
        self.print_hex_chunks(connection_bytes.hex(), "Connection Data (hex, no weights)")
        
        # Print zipped connection bytes in hex format
        zipped_connections = zlib.compress(connection_bytes)
        self.print_hex_chunks(zipped_connections.hex(), "Zipped Connections (hex)")
        print(f"Connection compression ratio: {len(zipped_connections)/len(connection_bytes):.2f}")
        
        print(f"Final neuron count: {final_grid.neuron_count()}")

    def _get_strategy_instance(self, strategy_name, default, strategies, **kwargs):
        """
        Get strategy instance from name.
        
        Args:
            strategy_name (str): Name of strategy class
            default (class): Default strategy class
            strategies (dict): Dictionary mapping strategy names to classes
            **kwargs: Additional arguments to pass to strategy constructor
            
        Returns:
            Strategy instance
        """
        if not strategy_name:
            return default(**kwargs)
            
        try:
            # Special handling for AdaptiveMutationStrategy
            if strategy_name == 'AdaptiveMutationStrategy':
                # Get base strategy from config
                base_strategy_name = self.config.get('base_mutation_strategy', 'DefaultMutationStrategy')
                base_strategy = strategies[base_strategy_name]()
                
                # Get adaptive parameters from config
                min_multiplier = self.config.get('adaptive_min_multiplier', 0.5)
                max_multiplier = self.config.get('adaptive_max_multiplier', 2.0)
                
                return AdaptiveMutationStrategy(
                    base_strategy=base_strategy,
                    min_multiplier=min_multiplier,
                    max_multiplier=max_multiplier
                )
            
            strategy_class = strategies[strategy_name]
            return strategy_class(**kwargs)
        except KeyError:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {list(strategies.keys())}")

    def run(self):
        """Run the experiment according to configuration."""
        # Check if the experiment should be run based on configuration changes
        should_run, stats_path, genome_path, population_state_path = self._should_run_experiment()
        
        if not should_run:
            # Load and return previous results instead of running the experiment again
            try:
                # Load stats
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                
                # Load best genome - add error handling for the specific file
                try:
                    # First check if file has content and print its size
                    file_size = os.path.getsize(genome_path)
                    if file_size == 0:
                        raise ValueError("Best genome file is empty")
                    
                    # Try to read raw content first to debug
                    with open(genome_path, 'r') as f:
                        file_content = f.read(min(1000, file_size))  # Read at most 1000 chars for debug
                    
                    # Try to load the genome - use named parameter 'filepath' like in main code
                    try:
                        best_genome = Genome.from_json(filepath=genome_path)
                    except Exception as e:
                        print("Full stack trace:")
                        traceback.print_exc()
                        raise e
                    
                except Exception as genome_error:
                    print(f"Error loading best genome file: {genome_error}")
                    print(f"Checking if genome data is in stats file...")
                    
                    # Try to reconstruct from stats if possible
                    if 'best_genome_data' in stats:
                        print("Found genome data in stats file, reconstructing...")
                        try:
                            best_genome = Genome.from_dict(stats['best_genome_data'])
                        except Exception as dict_error:
                            print("Error reconstructing from stats file:")
                            traceback.print_exc()
                            raise dict_error
                    else:
                        print("Cannot recover genome data, running experiment instead...")
                        raise ValueError("Cannot recover genome data")
                    
                print(f"Loaded previous results with best fitness: {stats.get('best_fitness', 'unknown')}")
                return best_genome, stats
                
            except Exception as e:
                print(f"Error loading previous results: {e}")
                print("Full stack trace:")
                traceback.print_exc()
                print("Running experiment instead...")
                # If we can't load previous results, run the experiment
        
        print("Initializing optimizer...")
        print("Press Enter at any time to skip to final results...")
        
        self.start_time = time.time()
        
        # Create optimizer based on type
        if self.optimizer_type == 'genetic_algorithm':
            # Get GA parameters from config
            ga_params = self.config.get('ga_params', {}).copy()
            
            # Remove convergence_strategy from ga_params if it exists to prevent duplicate parameter
            if 'convergence_strategy' in ga_params:
                del ga_params['convergence_strategy']
            
            # Get elitism strategy if specified
            elitism_strategy = self._get_strategy_instance(
                self.config.get('elitism_strategy'),
                default=TopFitnessSelectionStrategy,
                strategies={
                    'TopFitnessSelectionStrategy': TopFitnessSelectionStrategy,
                    'TopDistinctFitnessSelectionStrategy': TopDistinctFitnessSelectionStrategy
                },
                selection_size=ga_params.get('selection_size', 2)
            )
            
            # Get selection strategy if specified
            parents_selection_strategy = None
            if 'parents_selection_strategy' in self.config:
                selection_config = self.config['parents_selection_strategy']
                selection_type = selection_config.get('type', 'TopPercentageSelectionStrategy')
                
                # Get selection pressure from config or use default
                selection_pressure = selection_config.get('selection_pressure', 0.2)
                
                if selection_type == 'TopPercentageSelectionStrategy':
                    parents_selection_strategy = TopPercentageSelectionStrategy(selection_pressure=selection_pressure)
                elif selection_type == 'TournamentSelectionStrategy':
                    tournament_size = selection_config.get('tournament_size', 3)
                    parents_selection_strategy = TournamentSelectionStrategy(
                        tournament_size=tournament_size,
                        selection_pressure=selection_pressure
                    )
                else:
                    raise ValueError(f"Unknown selection strategy type: {selection_type}")
            else:
                # Backward compatibility: use old style configuration
                use_tournament = ga_params.get('use_tournament', False)
                tournament_size = ga_params.get('tournament_size', 3)
                selection_pressure = ga_params.get('selection_pressure', 0.2)
                
                if use_tournament:
                    parents_selection_strategy = TournamentSelectionStrategy(
                        tournament_size=tournament_size,
                        selection_pressure=selection_pressure
                    )
                else:
                    parents_selection_strategy = TopPercentageSelectionStrategy(
                        selection_pressure=selection_pressure
                    )
                
                # Remove old style parameters from ga_params to prevent confusion
                if 'use_tournament' in ga_params:
                    del ga_params['use_tournament']
                if 'tournament_size' in ga_params:
                    del ga_params['tournament_size']
                if 'selection_pressure' in ga_params:
                    del ga_params['selection_pressure']
            
            # Get steady state strategy if specified
            steady_state_strategy = None
            if 'steady_state_strategy' in self.config:
                steady_state_config = self.config['steady_state_strategy']
                steady_state_type = steady_state_config.get('type', 'CombinedSelectionStrategy')
                
                if steady_state_type == 'CombinedSelectionStrategy':
                    # Get strategies configuration
                    strategies_config = steady_state_config.get('strategies', [])
                    strategies = []
                    
                    for strategy_config in strategies_config:
                        strategy_type = strategy_config.get('type')
                        if strategy_type == 'TopFitnessSelectionStrategy':
                            strategies.append(TopFitnessSelectionStrategy(
                                selection_size=strategy_config.get('selection_size', 2),
                                rng=None  # Will be set later
                            ))
                        elif strategy_type == 'TopDistinctFitnessSelectionStrategy':
                            strategies.append(TopDistinctFitnessSelectionStrategy(
                                selection_size=strategy_config.get('selection_size', 2),
                                rng=None  # Will be set later
                            ))
                        else:
                            raise ValueError(f"Unknown strategy type in steady state: {strategy_type}")
                    
                    steady_state_strategy = CombinedSelectionStrategy(
                        strategies=strategies,
                        exclude_previous=steady_state_config.get('exclude_previous', True)
                    )
                else:
                    raise ValueError(f"Unknown steady state strategy type: {steady_state_type}")
            
            # Create GA instance with base parameters
            optimizer = GeneticAlgorithm(
                fitness_fn=self.fitness_function_wrapper,
                convergence_strategy=self.convergence_strategy,
                elitism_strategy=elitism_strategy,
                parents_selection_strategy=parents_selection_strategy,
                steady_state_strategy=steady_state_strategy,
                **ga_params
            )
            
            # If using tournament selection, update its RNG to use GA's RNG
            if isinstance(parents_selection_strategy, TournamentSelectionStrategy):
                parents_selection_strategy.rng = optimizer.rng
            
            # Update elitism strategy's RNG to use GA's RNG
            if isinstance(elitism_strategy, (TopFitnessSelectionStrategy, TopDistinctFitnessSelectionStrategy)):
                elitism_strategy.rng = optimizer.rng
            
            # Replace the initial population with genomes using our configured strategies
            optimizer.population = [
                Genome.random(
                    optimizer.rng,  # Use GA's RNG for consistency
                    size_x=optimizer.grid_size_x,
                    size_y=optimizer.grid_size_y,
                    num_morphogens=optimizer.num_morphogens,
                    max_growth_steps=optimizer.max_growth_steps,
                    meta_parameters_strategy=self.meta_parameters_strategy,
                    mutation_strategy=self.mutation_strategy,
                    crossover_strategy=self.crossover_strategy
                ) for _ in range(optimizer.population_size)
            ]
            
            # Re-evaluate the new population
            optimizer._evaluate_population()
            
        elif self.optimizer_type == 'cma_es':
            # Get CMA-ES parameters from config
            cma_params = self.config.get('cma_params', {})
            
            # Create CMA-ES instance
            optimizer = CMAESOptimizer(
                fitness_fn=self.fitness_function_wrapper,
                initial_sigma=cma_params.get('initial_sigma', 0.5),
                population_size=cma_params.get('population_size', 100),
                seed=self.config['ga_params'].get('seed'),
                max_workers=cma_params.get('max_workers'),
                convergence_strategy=self.convergence_strategy,
                num_morphogens=self.config.get('ga_params', {}).get('num_morphogens')
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # Load population state if resuming from interrupted experiment
        if population_state_path is not None:
            if not self._load_population_state(optimizer, population_state_path):
                print("Failed to load population state, starting fresh...")
        
        # Print initial generation stats
        if not self._progress_callback(
            optimizer,
            optimizer.generation,
            optimizer.best_fitness,
            np.mean(optimizer.current_fitness_scores)
        ):
            print("\nFitness threshold achieved in initial population!")
            stats = {
                'generations': 0,
                'evaluations': optimizer.evaluation_count,
                'best_fitness': float(optimizer.best_fitness),
                'elapsed_time': time.time() - self.start_time,
                'configuration': self.config
            }
            self.save_results(optimizer.best_solution, stats)
            self._cleanup_population_state()
            return optimizer.best_solution, stats
        
        # Run optimization with population state saving
        print("Starting optimization...")
        max_generations = self.config.get('max_generations', 1000)
        
        # Custom callback that saves population state after each generation
        def progress_callback_with_saving(generation, best_fitness, avg_fitness):
            # Save population state after each generation
            self._save_population_state(optimizer, generation)
            # Call the original progress callback
            return self._progress_callback(optimizer, generation, best_fitness, avg_fitness)
        
        optimizer.run(max_generations, callback=progress_callback_with_saving)
        
        # Collect statistics
        stats = {
            'generations': optimizer.generation,
            'evaluations': optimizer.evaluation_count,
            'best_fitness': float(optimizer.best_fitness),
            'elapsed_time': time.time() - self.start_time,
            'configuration': self.config
        }
        
        # Save results
        self.save_results(optimizer.best_solution, stats)
        
        # Clean up population state file after successful completion
        self._cleanup_population_state()
        
        print("\nOptimization complete!")
        print(f"Best fitness achieved: {optimizer.best_fitness:.2f}")
        print(f"Total evaluations: {optimizer.evaluation_count}")
        
        # Run final simulation and print analysis
        print("\nRunning simulation with best genome...")
        final_grid = self._run_simulation(optimizer.best_solution, verbose=True)
        self._print_final_analysis(optimizer.best_solution, final_grid)
        
        print(f"\nResults saved to: experiments/results/{self.experiment_name}/")
        
        return optimizer.best_solution, stats

    def plot_fitness_history(self, stats_path):
        """Plot fitness history and neuron count from a stats.json file."""
        # Load stats file
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        generations_data = stats['generations_data']
        generations = [g['generation'] for g in generations_data]
        best_fitness = [g['best_fitness'] for g in generations_data]
        neuron_counts = [g['neurons'] for g in generations_data]
        no_incoming_counts = [g['inputs'] for g in generations_data]
        connection_counts = [g['connections'] for g in generations_data]
        
        # Target values from configuration
        target_neurons = self.config['fitness_targets']['neurons']
        target_no_incoming = self.config['fitness_targets']['no_incoming']
        target_connections = self.config['fitness_targets']['connections']
        output_dir = os.path.join(self.config_dir, '..', 'results', self.experiment_name)
        
        # Plot 1: Best Fitness
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Best Fitness Over Generations')
        plt.grid(True)
        fitness_plot_path = os.path.join(output_dir, 'fitness_plot.png')
        plt.savefig(fitness_plot_path)
        plt.close()
        
        # Plot 2: Neuron Count, Connections, and No Incoming
        fig, ax1 = plt.subplots(figsize=(12, 6))  # Increased figure width to accommodate legend
        
        # Primary y-axis (neurons and connections)
        color1 = 'g'
        color3 = 'b'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Neurons / Connections', color='black')
        line1 = ax1.plot(generations, neuron_counts, color=color1, label='Actual Neurons')
        line3 = ax1.plot(generations, connection_counts, color=color3, label='Connections')
        target1 = ax1.axhline(y=target_neurons, color=color1, linestyle='--', label='Target Neurons')
        target3 = ax1.axhline(y=target_connections, color=color3, linestyle='--', label='Target Connections')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Secondary y-axis (no incoming)
        ax2 = ax1.twinx()
        color2 = 'r'
        ax2.set_ylabel('Neurons with No Incoming', color=color2)
        line2 = ax2.plot(generations, no_incoming_counts, color=color2, label='No Incoming')
        
        # Calculate the vertical position for the target line (in the middle of the chart)
        y_min, y_max = ax2.get_ylim()
        ax2.set_ylim(y_min, max(y_max, target_no_incoming * 2))  # Ensure target is visible
        
        target2 = ax2.axhline(y=target_no_incoming, color=color2, linestyle='--', label='Target No Incoming')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends from both axes and place outside
        lines = line1 + line2 + line3 + [target1, target2, target3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, bbox_to_anchor=(1.15, 1), loc='upper left')
        
        plt.title('Neuron Metrics Over Generations')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent legend cutoff
        neurons_plot_path = os.path.join(output_dir, 'neurons_plot.png')
        plt.savefig(neurons_plot_path, bbox_inches='tight')  # Ensure legend is included in saved figure
        plt.close()
        
        print(f"Fitness plot saved to: {fitness_plot_path}")
        print(f"Neurons plot saved to: {neurons_plot_path}") 