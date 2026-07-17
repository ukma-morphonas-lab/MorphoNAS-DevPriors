import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .genome import Genome

class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, genome: 'Genome', rng) -> 'Genome':
        """
        Mutate the genome according to the strategy.
        
        Args:
            genome: Genome instance to mutate
            rng: Random number generator instance
            
        Returns:
            New mutated Genome instance
        """
        pass

class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: 'Genome', parent2: 'Genome', rng) -> 'Genome':
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent Genome instance
            parent2: Second parent Genome instance
            rng: Random number generator instance
            
        Returns:
            New Genome instance combining traits from both parents
        """
        pass

class MetaParametersStrategy(ABC):
    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Get the meta-parameters according to the strategy.
        
        Returns:
            Dictionary of meta-parameters
        """
        pass

class MorphogenRegenerationMixin:
    """Mixin class providing morphogen regeneration functionality for mutation strategies."""
    
    def _regenerate_morphogen_structures(self, genome, params, new_num, rng):
        """Helper method to regenerate morphogen-dependent structures."""
        params['num_morphogens'] = int(new_num)
        
        # Generate new secretion rates and inhibition matrix
        (prog_rates, neuron_rates, inhibition_matrix) = genome._generate_random_morphogen_matrices(
            rng, new_num)
        
        params['progenitor_secretion_rates'] = prog_rates.tolist()
        params['neuron_secretion_rates'] = neuron_rates.tolist()
        params['inhibition_matrix'] = inhibition_matrix.tolist()
        params['diffusion_patterns'] = [
            genome._generate_random_diffusion_pattern(rng).tolist()
            for _ in range(new_num)
        ]
        
        return params

class DefaultMutationStrategy(MutationStrategy, MorphogenRegenerationMixin):
    def mutate(self, genome: 'Genome', rng):
        """Original mutation strategy from Genome class."""
        params = genome.to_dict()
        
        mutation_types = [
            'size',
            'steps',
            'parameter',
            'morphogen_num',
            'matrices'
        ]
        mutation_weights = [
            genome.meta_params.MUTATION_PROB_SIZE,
            genome.meta_params.MUTATION_PROB_STEPS,
            genome.meta_params.MUTATION_PROB_PARAMETER,
            genome.meta_params.MUTATION_PROB_MORPHOGEN,
            genome.meta_params.MUTATION_PROB_MATRICES
        ]
        
        # Normalize weights
        mutation_weights = np.array(mutation_weights) / sum(mutation_weights)
        mutation_type = rng.choice(mutation_types, p=mutation_weights)
        
        if mutation_type == 'matrices':
            # Copy existing matrix mutation logic from original mutate method
            num_mutations = rng.integers(1, genome.meta_params.MAX_MATRIX_MUTATIONS + 1)
            
            for _ in range(num_mutations):
                matrix_type = rng.choice(['inhibition_matrix', 'diffusion_patterns'])
                
                if matrix_type == 'inhibition_matrix':
                    matrix = np.array(params['inhibition_matrix'])
                    row = rng.integers(0, matrix.shape[0])
                    col = rng.integers(0, matrix.shape[1])
                    new_value = rng.uniform(genome.meta_params.RANDOM_INHIBITION_RANGE[0],
                                          genome.meta_params.RANDOM_INHIBITION_RANGE[1])
                    matrix[row, col] = new_value
                    params['inhibition_matrix'] = matrix.tolist()
                else:
                    pattern_idx = rng.integers(0, len(params['diffusion_patterns']))
                    pattern = np.array(params['diffusion_patterns'][pattern_idx])
                    row = rng.integers(0, pattern.shape[0])
                    col = rng.integers(0, pattern.shape[1])
                    new_value = rng.uniform(genome.meta_params.RANDOM_DIFFUSION_PATTERN_RANGE[0],
                                          genome.meta_params.RANDOM_DIFFUSION_PATTERN_RANGE[1])
                    pattern[row, col] = new_value
                    pattern = pattern / pattern.sum()
                    params['diffusion_patterns'][pattern_idx] = pattern.tolist()
        
        elif mutation_type == 'size':
            dim = rng.choice(['size_x', 'size_y'])
            delta = int(rng.choice([-1, 1]))  # Restore original random choice
            new_size = max(genome.meta_params.MIN_GRID_SIZE, params[dim] + delta)
            params[dim] = int(new_size)
        
        elif mutation_type == 'steps':
            delta = int(rng.integers(-genome.meta_params.MAX_GROWTH_STEPS_DELTA, 
                                   genome.meta_params.MAX_GROWTH_STEPS_DELTA + 1))
            new_steps = max(genome.meta_params.MIN_GROWTH_STEPS, params['max_growth_steps'] + delta)
            params['max_growth_steps'] = int(new_steps)
        
        elif mutation_type == 'parameter':
            float_params = [
                'diffusion_rate',
                'division_threshold',
                'cell_differentiation_threshold',
                'axon_growth_threshold',
                'axon_connect_threshold',
                'self_connect_isolated_neurons_fraction',
                'weight_adjustment_target',
                'weight_adjustment_rate'
            ]
            int_params = ['max_axon_length']
            
            if rng.random() < genome.meta_params.MUTATION_PROB_FLOAT_VS_INT:
                param = rng.choice(float_params)
                mutation_factor = rng.uniform(1.0/genome.meta_params.MUTATION_FACTOR_FLOAT, 
                                           genome.meta_params.MUTATION_FACTOR_FLOAT)
                new_value = params[param] * mutation_factor
                params[param] = np.clip(new_value, 0.0, 1.0)
            else:
                param = rng.choice(int_params)
                delta = int(rng.choice([-1, 1]))  # Restore original random choice
                params[param] = max(1, params[param] + delta)
        
        else:  # morphogen_num
            delta = int(rng.choice([-1, 1]))  # Restore original random choice
            new_num = max(genome.meta_params.MIN_MORPHOGENS, params['num_morphogens'] + delta)
            
            if new_num != params['num_morphogens']:
                params = self._regenerate_morphogen_structures(genome, params, new_num, rng)
        
        return genome.__class__.from_dict(params, meta_parameters_strategy=genome.meta_parameters_strategy, mutation_strategy=genome.mutation_strategy, crossover_strategy=genome.crossover_strategy)

class DefaultCrossoverStrategy(CrossoverStrategy):
    def crossover(self, parent1: 'Genome', parent2: 'Genome', rng):
        """Original crossover strategy from Genome class."""
        p1_dict = parent1.to_dict()
        p2_dict = parent2.to_dict()
        child_dict = {}

        # Always set num_morphogens first
        if parent1.num_morphogens == parent2.num_morphogens:
            child_dict['num_morphogens'] = parent1.num_morphogens
            
            # Simple parameters: randomly choose from either parent
            simple_params = [
                'max_growth_steps', 'size_x', 'size_y', 'diffusion_rate', 'division_threshold',
                'cell_differentiation_threshold', 'axon_growth_threshold',
                'max_axon_length', 'axon_connect_threshold',
                'self_connect_isolated_neurons_fraction',
                'weight_adjustment_target', 'weight_adjustment_rate'
            ]
            for param in simple_params:
                child_dict[param] = p1_dict[param] if rng.random() < 0.5 else p2_dict[param]

            # For array parameters, perform element-wise crossover
            array_params = [
                'progenitor_secretion_rates',
                'neuron_secretion_rates'
            ]
            for param in array_params:
                mask = rng.random(parent1.num_morphogens) < 0.5
                child_dict[param] = [
                    p1_dict[param][i] if mask[i] else p2_dict[param][i]
                    for i in range(parent1.num_morphogens)
                ]

            # For inhibition matrix, crossover row by row
            child_dict['inhibition_matrix'] = [
                p1_dict['inhibition_matrix'][i] if rng.random() < 0.5 
                else p2_dict['inhibition_matrix'][i]
                for i in range(parent1.num_morphogens)
            ]

            # For diffusion patterns, randomly select complete patterns
            child_dict['diffusion_patterns'] = [
                p1_dict['diffusion_patterns'][i] if rng.random() < 0.5 
                else p2_dict['diffusion_patterns'][i]
                for i in range(parent1.num_morphogens)
            ]
        else:
            # Randomly choose number of morphogens from either parent
            chosen_parent = p1_dict if rng.random() < 0.5 else p2_dict
            child_dict['num_morphogens'] = chosen_parent['num_morphogens']
            
            # Simple parameters: randomly choose from either parent
            simple_params = [
                'max_growth_steps', 'size_x', 'size_y', 'diffusion_rate', 'division_threshold',
                'cell_differentiation_threshold', 'axon_growth_threshold',
                'max_axon_length', 'axon_connect_threshold',
                'self_connect_isolated_neurons_fraction',
                'weight_adjustment_target', 'weight_adjustment_rate'
            ]
            for param in simple_params:
                child_dict[param] = p1_dict[param] if rng.random() < 0.5 else p2_dict[param]
            
            # For array parameters, take from the parent with matching morphogen count
            array_params = [
                'progenitor_secretion_rates',
                'neuron_secretion_rates',
                'inhibition_matrix',
                'diffusion_patterns'
            ]
            for param in array_params:
                child_dict[param] = chosen_parent[param]

        return parent1.__class__.from_dict(child_dict, meta_parameters_strategy=parent1.meta_parameters_strategy, mutation_strategy=parent1.mutation_strategy, crossover_strategy=parent1.crossover_strategy) 

class BlockPreservationCrossoverStrategy(CrossoverStrategy):
    """Crossover strategy that preserves functionally related parameter blocks."""
    
    def crossover(self, parent1: 'Genome', parent2: 'Genome', rng):
        """
        Perform crossover while preserving parameter blocks.
        
        Args:
            parent1: First parent Genome instance
            parent2: Second parent Genome instance
            rng: Random number generator instance
            
        Returns:
            New Genome instance combining traits from both parents
        """
        p1_dict = parent1.to_dict()
        p2_dict = parent2.to_dict()
        child_dict = {}
        
        # First determine the number of morphogens
        if parent1.num_morphogens == parent2.num_morphogens:
            child_dict['num_morphogens'] = parent1.num_morphogens
            
            # Block 1: Size and Growth
            # Keep size_x, size_y, and max_growth_steps together
            if rng.random() < 0.5:
                child_dict['size_x'] = p1_dict['size_x']
                child_dict['size_y'] = p1_dict['size_y']
                child_dict['max_growth_steps'] = p1_dict['max_growth_steps']
            else:
                child_dict['size_x'] = p2_dict['size_x']
                child_dict['size_y'] = p2_dict['size_y']
                child_dict['max_growth_steps'] = p2_dict['max_growth_steps']
                
            # Block 2: Morphogen Diffusion
            if rng.random() < 0.5:
                child_dict['diffusion_rate'] = p1_dict['diffusion_rate']
                child_dict['diffusion_patterns'] = p1_dict['diffusion_patterns'].copy()
            else:
                child_dict['diffusion_rate'] = p2_dict['diffusion_rate']
                child_dict['diffusion_patterns'] = p2_dict['diffusion_patterns'].copy()
                
            # Block 3: Cell Division and Differentiation
            if rng.random() < 0.5:
                child_dict['division_threshold'] = p1_dict['division_threshold']
                child_dict['cell_differentiation_threshold'] = p1_dict['cell_differentiation_threshold']
            else:
                child_dict['division_threshold'] = p2_dict['division_threshold']
                child_dict['cell_differentiation_threshold'] = p2_dict['cell_differentiation_threshold']
                
            # Block 4: Axon Growth and Connection
            if rng.random() < 0.5:
                child_dict['axon_growth_threshold'] = p1_dict['axon_growth_threshold']
                child_dict['max_axon_length'] = p1_dict['max_axon_length']
                child_dict['axon_connect_threshold'] = p1_dict['axon_connect_threshold']
            else:
                child_dict['axon_growth_threshold'] = p2_dict['axon_growth_threshold']
                child_dict['max_axon_length'] = p2_dict['max_axon_length']
                child_dict['axon_connect_threshold'] = p2_dict['axon_connect_threshold']
                
            # Block 5: Network Weight Adjustment
            if rng.random() < 0.5:
                child_dict['self_connect_isolated_neurons_fraction'] = p1_dict['self_connect_isolated_neurons_fraction']
                child_dict['weight_adjustment_target'] = p1_dict['weight_adjustment_target']
                child_dict['weight_adjustment_rate'] = p1_dict['weight_adjustment_rate']
            else:
                child_dict['self_connect_isolated_neurons_fraction'] = p2_dict['self_connect_isolated_neurons_fraction']
                child_dict['weight_adjustment_target'] = p2_dict['weight_adjustment_target']
                child_dict['weight_adjustment_rate'] = p2_dict['weight_adjustment_rate']
            
            # Block 6: Per-Morphogen Signaling (treat each morphogen as a unit)
            child_dict['progenitor_secretion_rates'] = []
            child_dict['neuron_secretion_rates'] = []
            child_dict['inhibition_matrix'] = []
            
            for i in range(child_dict['num_morphogens']):
                # For each morphogen, decide as a block which parent to take from
                if rng.random() < 0.5:
                    child_dict['progenitor_secretion_rates'].append(p1_dict['progenitor_secretion_rates'][i])
                    child_dict['neuron_secretion_rates'].append(p1_dict['neuron_secretion_rates'][i])
                    child_dict['inhibition_matrix'].append(p1_dict['inhibition_matrix'][i])
                    # Note: diffusion patterns already handled above as part of Block 2
                else:
                    child_dict['progenitor_secretion_rates'].append(p2_dict['progenitor_secretion_rates'][i])
                    child_dict['neuron_secretion_rates'].append(p2_dict['neuron_secretion_rates'][i])
                    child_dict['inhibition_matrix'].append(p2_dict['inhibition_matrix'][i])
                    # Note: diffusion patterns already handled above as part of Block 2
        else:
            # When morphogen counts differ, we need a different approach
            # Randomly select which parent's morphogen count to use
            if rng.random() < 0.5:
                donor_parent = parent1
                donor_dict = p1_dict
            else:
                donor_parent = parent2
                donor_dict = p2_dict
                
            child_dict['num_morphogens'] = donor_parent.num_morphogens
            
            # For morphogen-specific parameters, use the donor parent
            morphogen_params = [
                'progenitor_secretion_rates',
                'neuron_secretion_rates',
                'inhibition_matrix',
                'diffusion_patterns'
            ]
            for param in morphogen_params:
                child_dict[param] = donor_dict[param].copy()
            
            # For the other parameter blocks, we can still use block-based inheritance
            # Block 1: Size and Growth
            if rng.random() < 0.5:
                child_dict['size_x'] = p1_dict['size_x']
                child_dict['size_y'] = p1_dict['size_y']
                child_dict['max_growth_steps'] = p1_dict['max_growth_steps']
            else:
                child_dict['size_x'] = p2_dict['size_x']
                child_dict['size_y'] = p2_dict['size_y']
                child_dict['max_growth_steps'] = p2_dict['max_growth_steps']
            
            # Block 2: Diffusion Rate (only the rate, patterns are morphogen-specific)
            child_dict['diffusion_rate'] = p1_dict['diffusion_rate'] if rng.random() < 0.5 else p2_dict['diffusion_rate']
            
            # Block 3: Cell Division and Differentiation
            if rng.random() < 0.5:
                child_dict['division_threshold'] = p1_dict['division_threshold']
                child_dict['cell_differentiation_threshold'] = p1_dict['cell_differentiation_threshold']
            else:
                child_dict['division_threshold'] = p2_dict['division_threshold']
                child_dict['cell_differentiation_threshold'] = p2_dict['cell_differentiation_threshold']
            
            # Block 4: Axon Growth and Connection
            if rng.random() < 0.5:
                child_dict['axon_growth_threshold'] = p1_dict['axon_growth_threshold']
                child_dict['max_axon_length'] = p1_dict['max_axon_length']
                child_dict['axon_connect_threshold'] = p1_dict['axon_connect_threshold']
            else:
                child_dict['axon_growth_threshold'] = p2_dict['axon_growth_threshold']
                child_dict['max_axon_length'] = p2_dict['max_axon_length']
                child_dict['axon_connect_threshold'] = p2_dict['axon_connect_threshold']
            
            # Block 5: Network Weight Adjustment
            if rng.random() < 0.5:
                child_dict['self_connect_isolated_neurons_fraction'] = p1_dict['self_connect_isolated_neurons_fraction']
                child_dict['weight_adjustment_target'] = p1_dict['weight_adjustment_target']
                child_dict['weight_adjustment_rate'] = p1_dict['weight_adjustment_rate']
            else:
                child_dict['self_connect_isolated_neurons_fraction'] = p2_dict['self_connect_isolated_neurons_fraction']
                child_dict['weight_adjustment_target'] = p2_dict['weight_adjustment_target']
                child_dict['weight_adjustment_rate'] = p2_dict['weight_adjustment_rate']
            
        return parent1.__class__.from_dict(child_dict, meta_parameters_strategy=parent1.meta_parameters_strategy, mutation_strategy=parent1.mutation_strategy, crossover_strategy=parent1.crossover_strategy) 

class DefaultMetaParametersStrategy(MetaParametersStrategy):
    def get_parameters(self):
        """Original meta-parameters from Genome class."""
        return {
            # Mutation probability constants
            'MUTATION_PROB_SIZE': 0.15,            # 15% chance to mutate size
            'MUTATION_PROB_STEPS': 0.15,           # 15% chance to mutate steps
            'MUTATION_PROB_PARAMETER': 0.45,       # 45% chance to mutate a parameter
            'MUTATION_PROB_MORPHOGEN': 0.1,        # 10% chance to change number of morphogens
            'MUTATION_PROB_MATRICES': 0.15,        # 15% chance to mutate morphogen matrices
            'MUTATION_PROB_FLOAT_VS_INT': 0.9,     # 90% chance to mutate float vs int parameter
            'MUTATION_FACTOR_FLOAT': 2.0,          # Float params can change by factor of 1/2 to 2
            
            # Growth and size constraints
            'MIN_GROWTH_STEPS': 10,
            'MAX_GROWTH_STEPS_DELTA': 20,
            'MIN_GRID_SIZE': 1,
            'MIN_MORPHOGENS': 3,
            
            # Random generation ranges
            'RANDOM_GROWTH_STEPS_RANGE': (100, 700),
            'RANDOM_GRID_SIZE_RANGE': (10, 30),
            'RANDOM_DIFFUSION_RATE_RANGE': (0.05, 0.2),
            'RANDOM_DIVISION_THRESHOLD_RANGE': (0.3, 0.7),
            'RANDOM_DIFFERENTIATION_THRESHOLD_RANGE': (0.3, 0.7),
            'RANDOM_AXON_GROWTH_THRESHOLD_RANGE': (0.2, 0.5),
            'RANDOM_AXON_CONNECT_THRESHOLD_RANGE': (0.4, 1.0),
            'RANDOM_SELF_CONNECT_RANGE': (0, 0.3),
            'RANDOM_WEIGHT_TARGET_RANGE': (0.3, 0.7),
            'RANDOM_WEIGHT_RATE_RANGE': (0.005, 0.02),
            'RANDOM_AXON_LENGTH_RANGE': (2, 11),
            'RANDOM_SECRETION_RANGE': (0, 0.5),
            'RANDOM_INHIBITION_RANGE': (0, 0.5),
            'RANDOM_DIFFUSION_PATTERN_RANGE': (0, 0.3),
            
            # Matrix mutation constraints
            'MAX_MATRIX_MUTATIONS': 3,           # Back to original value
            
            # Morphogen distribution probabilities
            'MORPHOGEN_PROBS': {
                3: 0.7,  # 70% chance for 3 morphogens
                4: 0.1,  # 10% chance for 4 morphogens
                5: 0.1,  # 10% chance for 5 morphogens
                6: 0.1   # 10% chance for 6 morphogens
            }
        } 

class AggressiveMutationStrategy(MutationStrategy, MorphogenRegenerationMixin):
    def mutate(self, genome: 'Genome', rng):
        """Enhanced mutation strategy with more dramatic changes."""
        params = genome.to_dict()
        
        mutation_types = [
            'size',
            'steps',
            'parameter',
            'morphogen_num',
            'matrices'
        ]
        mutation_weights = [
            genome.meta_params.MUTATION_PROB_SIZE,
            genome.meta_params.MUTATION_PROB_STEPS,
            genome.meta_params.MUTATION_PROB_PARAMETER,
            genome.meta_params.MUTATION_PROB_MORPHOGEN,
            genome.meta_params.MUTATION_PROB_MATRICES
        ]
        
        # Normalize weights
        mutation_weights = np.array(mutation_weights) / sum(mutation_weights)
        mutation_type = rng.choice(mutation_types, p=mutation_weights)
        
        if mutation_type == 'matrices':
            # Copy existing matrix mutation logic but use updated MAX_MATRIX_MUTATIONS
            num_mutations = rng.integers(1, genome.meta_params.MAX_MATRIX_MUTATIONS + 1)
            
            for _ in range(num_mutations):
                matrix_type = rng.choice(['inhibition_matrix', 'diffusion_patterns'])
                
                if matrix_type == 'inhibition_matrix':
                    matrix = np.array(params['inhibition_matrix'])
                    row = rng.integers(0, matrix.shape[0])
                    col = rng.integers(0, matrix.shape[1])
                    new_value = rng.uniform(genome.meta_params.RANDOM_INHIBITION_RANGE[0],
                                          genome.meta_params.RANDOM_INHIBITION_RANGE[1])
                    matrix[row, col] = new_value
                    params['inhibition_matrix'] = matrix.tolist()
                else:
                    pattern_idx = rng.integers(0, len(params['diffusion_patterns']))
                    pattern = np.array(params['diffusion_patterns'][pattern_idx])
                    row = rng.integers(0, pattern.shape[0])
                    col = rng.integers(0, pattern.shape[1])
                    new_value = rng.uniform(genome.meta_params.RANDOM_DIFFUSION_PATTERN_RANGE[0],
                                          genome.meta_params.RANDOM_DIFFUSION_PATTERN_RANGE[1])
                    pattern[row, col] = new_value
                    pattern = pattern / pattern.sum()
                    params['diffusion_patterns'][pattern_idx] = pattern.tolist()
        
        elif mutation_type == 'size':
            dim = rng.choice(['size_x', 'size_y'])
            delta = int(rng.integers(-3, 3))  # Changed from [-1, 1] to [-3, 3]
            new_size = max(genome.meta_params.MIN_GRID_SIZE, params[dim] + delta)
            params[dim] = int(new_size)
        
        elif mutation_type == 'steps':
            delta = int(rng.integers(-genome.meta_params.MAX_GROWTH_STEPS_DELTA, 
                                   genome.meta_params.MAX_GROWTH_STEPS_DELTA + 1))
            new_steps = max(genome.meta_params.MIN_GROWTH_STEPS, params['max_growth_steps'] + delta)
            params['max_growth_steps'] = int(new_steps)
        
        elif mutation_type == 'parameter':
            float_params = [
                'diffusion_rate',
                'division_threshold',
                'cell_differentiation_threshold',
                'axon_growth_threshold',
                'axon_connect_threshold',
                'self_connect_isolated_neurons_fraction',
                'weight_adjustment_target',
                'weight_adjustment_rate'
            ]
            int_params = ['max_axon_length']
            
            if rng.random() < genome.meta_params.MUTATION_PROB_FLOAT_VS_INT:
                param = rng.choice(float_params)
                mutation_factor = rng.uniform(1.0/genome.meta_params.MUTATION_FACTOR_FLOAT, 
                                           genome.meta_params.MUTATION_FACTOR_FLOAT)
                new_value = params[param] * mutation_factor
                params[param] = np.clip(new_value, 0.0, 1.0)
            else:
                param = rng.choice(int_params)
                delta = int(rng.integers(-2, 2))  # Changed from [-1, 1] to [-2, 2]
                params[param] = max(1, params[param] + delta)
        
        else:  # morphogen_num
            delta = int(rng.choice([-2, -1, 1, 2]))  # Changed to allow bigger jumps
            new_num = max(genome.meta_params.MIN_MORPHOGENS, params['num_morphogens'] + delta)
            
            if new_num != params['num_morphogens']:
                params = self._regenerate_morphogen_structures(genome, params, new_num, rng)
        
        # Add chance for radical mutation
        if rng.random() < 0.05:  # 5% chance of radical mutation
            # For morphogens:
            new_num = rng.integers(genome.meta_params.MIN_MORPHOGENS, genome.meta_params.MIN_MORPHOGENS + 10)
            # For growth steps:
            new_steps = rng.integers(genome.meta_params.MIN_GROWTH_STEPS, 1000)
            
            params['max_growth_steps'] = int(new_steps)
            if new_num != params['num_morphogens']:
                params = self._regenerate_morphogen_structures(genome, params, new_num, rng)
        
        return genome.__class__.from_dict(params, meta_parameters_strategy=genome.meta_parameters_strategy, mutation_strategy=genome.mutation_strategy, crossover_strategy=genome.crossover_strategy)

class ExtendedMatrixMutationStrategy(DefaultMetaParametersStrategy):
    def get_parameters(self):
        """Enhanced meta-parameters with increased matrix mutations."""
        params = super().get_parameters()
        params['MAX_MATRIX_MUTATIONS'] = 5  # Increase from default 3 to 5
        return params 

class AdaptiveMutationStrategy(MutationStrategy):
    """Mutation strategy that adapts mutation rates based on population convergence."""
    
    def __init__(self, base_strategy, min_multiplier=0.5, max_multiplier=2.0):
        """
        Initialize adaptive mutation strategy.
        
        Args:
            base_strategy: Base mutation strategy to use
            min_multiplier: Minimum mutation rate multiplier (default: 0.5)
            max_multiplier: Maximum mutation rate multiplier (default: 2.0)
        """
        self.base_strategy = base_strategy
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.convergence_ratio = 1.0  # Start with no adjustment
    
    def update_convergence(self, best_fitness, avg_fitness):
        """
        Update convergence ratio based on current population state.
        
        Args:
            best_fitness: Best fitness in current population
            avg_fitness: Average fitness of current population
        """
        if best_fitness <= 0:
            self.convergence_ratio = 1.0
            return
            
        # Calculate convergence ratio (closer to 1 means more converged)
        self.convergence_ratio = avg_fitness / best_fitness
    
    def get_mutation_rate_multiplier(self):
        """
        Get the current mutation rate multiplier based on convergence.
        
        Returns:
            float: Mutation rate multiplier between min_multiplier and max_multiplier
        """
        # When convergence_ratio is close to 1 (highly converged), multiplier is high
        # When convergence_ratio is close to 0 (not converged), multiplier is close to 1
        return self.min_multiplier + (self.max_multiplier - self.min_multiplier) * self.convergence_ratio
    
    def mutate(self, genome: 'Genome', rng) -> 'Genome':
        """
        Mutate the genome using the base strategy.
        
        Args:
            genome: Genome instance to mutate
            rng: Random number generator instance
            
        Returns:
            New mutated Genome instance
        """
        # Simply delegate to base strategy
        return self.base_strategy.mutate(genome, rng)
