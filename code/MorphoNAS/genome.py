import numpy as np
import struct
try:
    from .genome_strategies import (
        DefaultMutationStrategy,
        DefaultCrossoverStrategy,
        DefaultMetaParametersStrategy,
        MetaParametersStrategy,
        MutationStrategy,
        CrossoverStrategy
    )
except ImportError:
    from .genome_strategies import (
        DefaultMutationStrategy,
        DefaultCrossoverStrategy,
        DefaultMetaParametersStrategy,
        MetaParametersStrategy,
        MutationStrategy,
        CrossoverStrategy
    )

class MetaParameters:
    """Container class for genome meta-parameters."""
    def __init__(self, params_dict):
        for key, value in params_dict.items():
            setattr(self, key, value)

class Genome:
    def __init__(self, 
                 max_growth_steps=500,
                 size_x=200,
                 size_y=200,
                 diffusion_rate=0.1,
                 num_morphogens=3,
                 division_threshold=0.5,
                 cell_differentiation_threshold=0.5,
                 axon_growth_threshold=0.3,
                 max_axon_length=5,
                 axon_connect_threshold=0.6,
                 self_connect_isolated_neurons_fraction=0.0,
                 weight_adjustment_target=0.5,
                 weight_adjustment_rate=0.01,
                 progenitor_secretion_rates=None,
                 neuron_secretion_rates=None,
                 inhibition_matrix=None,
                 diffusion_patterns=None,
                 meta_parameters_strategy=None,
                 mutation_strategy=None,
                 crossover_strategy=None):
        
        # Set up meta-parameters first
        if meta_parameters_strategy is None:
            meta_parameters_strategy = DefaultMetaParametersStrategy()
        assert isinstance(meta_parameters_strategy, MetaParametersStrategy), "meta_parameters_strategy must be an instance of MetaParametersStrategy"
        self.meta_params = MetaParameters(meta_parameters_strategy.get_parameters())
        self.meta_parameters_strategy = meta_parameters_strategy
        
        # Set up mutation and crossover strategies
        if mutation_strategy is None:
            mutation_strategy = DefaultMutationStrategy()
        assert isinstance(mutation_strategy, MutationStrategy), "mutation_strategy must be an instance of MutationStrategy"
        self.mutation_strategy = mutation_strategy
        
        if crossover_strategy is None:
            crossover_strategy = DefaultCrossoverStrategy()
        assert isinstance(crossover_strategy, CrossoverStrategy), "crossover_strategy must be an instance of CrossoverStrategy"
        self.crossover_strategy = crossover_strategy
        
        # Input validation
        assert isinstance(max_growth_steps, int) and max_growth_steps > 0, "max_growth_steps must be a positive integer"
        assert isinstance(size_x, int) and size_x > 0, "size_x must be a positive integer"
        assert isinstance(size_y, int) and size_y > 0, "size_y must be a positive integer"
        assert isinstance(num_morphogens, int) and num_morphogens > 0, "num_morphogens must be a positive integer"
        assert num_morphogens >= 3, "num_morphogens must be at least 3"
        assert isinstance(max_axon_length, int) and max_axon_length > 0, "max_axon_length must be a positive integer"
        
        # Validate float parameters are between 0 and 1
        for name, value in [
            ("diffusion_rate", diffusion_rate),
            ("division_threshold", division_threshold),
            ("cell_differentiation_threshold", cell_differentiation_threshold),
            ("axon_growth_threshold", axon_growth_threshold),
            ("axon_connect_threshold", axon_connect_threshold)
        ]:
            assert isinstance(value, (int, float)), f"{name} must be a number"
            assert 0 <= value <= 1, f"{name} must be between 0 and 1"
        
        # Add validation for new parameter
        assert isinstance(self_connect_isolated_neurons_fraction, (int, float)), "self_connect_isolated_neurons_fraction must be a number"
        assert 0 <= self_connect_isolated_neurons_fraction <= 1, "self_connect_isolated_neurons_fraction must be between 0 and 1"
        
        # Validate new weight adjustment parameters
        assert isinstance(weight_adjustment_target, (int, float)), "weight_adjustment_target must be a number"
        assert 0 <= weight_adjustment_target <= 1, "weight_adjustment_target must be between 0 and 1"
        assert isinstance(weight_adjustment_rate, (int, float)), "weight_adjustment_rate must be a number"
        assert 0 <= weight_adjustment_rate <= 1, "weight_adjustment_rate must be between 0 and 1"
        
        # Basic parameters
        self.max_growth_steps = max_growth_steps
        self.size_x = size_x
        self.size_y = size_y
        self.diffusion_rate = diffusion_rate
        self.num_morphogens = num_morphogens
        self.division_threshold = division_threshold
        self.cell_differentiation_threshold = cell_differentiation_threshold
        self.axon_growth_threshold = axon_growth_threshold
        self.max_axon_length = max_axon_length
        self.axon_connect_threshold = axon_connect_threshold
        self.self_connect_isolated_neurons_fraction = self_connect_isolated_neurons_fraction
        self.weight_adjustment_target = weight_adjustment_target
        self.weight_adjustment_rate = weight_adjustment_rate
        
        # Array parameters with defaults
        self.progenitor_secretion_rates = np.array(progenitor_secretion_rates if progenitor_secretion_rates is not None 
                                                  else np.zeros(num_morphogens))
        self.neuron_secretion_rates = np.array(neuron_secretion_rates if neuron_secretion_rates is not None 
                                             else np.zeros(num_morphogens))
        self.inhibition_matrix = np.array(inhibition_matrix if inhibition_matrix is not None 
                                        else np.zeros((num_morphogens, num_morphogens)))
        
        # Handle diffusion patterns differently for inhomogeneous shapes
        if diffusion_patterns is not None:
            # Convert each pattern to numpy array individually
            self.diffusion_patterns = [np.array(pattern) for pattern in diffusion_patterns]
        else:
            # Default patterns
            self.diffusion_patterns = [np.ones((3, 3)) / 9 for _ in range(num_morphogens)]
        
        # Validate array shapes
        assert self.progenitor_secretion_rates.shape == (num_morphogens,), \
            f"progenitor_secretion_rates must have shape ({num_morphogens},)"
        assert self.neuron_secretion_rates.shape == (num_morphogens,), \
            f"neuron_secretion_rates must have shape ({num_morphogens},)"
        assert self.inhibition_matrix.shape == (num_morphogens, num_morphogens), \
            f"inhibition_matrix must have shape ({num_morphogens}, {num_morphogens})"
        
        # Validate array values
        assert np.all((0 <= self.progenitor_secretion_rates) & (self.progenitor_secretion_rates <= 1)), \
            "progenitor_secretion_rates values must be between 0 and 1"
        assert np.all((0 <= self.neuron_secretion_rates) & (self.neuron_secretion_rates <= 1)), \
            "neuron_secretion_rates values must be between 0 and 1"
        assert np.all((0 <= self.inhibition_matrix) & (self.inhibition_matrix <= 1)), \
            "inhibition_matrix values must be between 0 and 1"
        
        # Validate diffusion patterns individually
        for i, pattern in enumerate(self.diffusion_patterns):
            assert np.all((0 <= pattern) & (pattern <= 1)), \
                f"diffusion_pattern {i} values must be between 0 and 1"

    def to_bytes(self):
        """Serialize the genome to bytes."""
        # Pack basic parameters
        basic_params = struct.pack('iiiffffffffi',
            self.max_growth_steps,   
            self.size_x,
            self.size_y,
            self.diffusion_rate,
            self.division_threshold,
            self.cell_differentiation_threshold,
            self.axon_growth_threshold,
            self.axon_connect_threshold,
            self.self_connect_isolated_neurons_fraction,
            self.weight_adjustment_target,
            self.weight_adjustment_rate,
            self.max_axon_length
        )
        
        # Pack numpy arrays
        arrays = [
            self.progenitor_secretion_rates.tobytes(),
            self.neuron_secretion_rates.tobytes(),
            self.inhibition_matrix.tobytes()
        ]
        
        # Pack diffusion patterns with their shapes
        for pattern in self.diffusion_patterns:
            # Add shape information before each pattern
            arrays.append(struct.pack('ii', *pattern.shape))
            arrays.append(pattern.tobytes())
        
        return basic_params + b''.join(arrays)

    @classmethod
    def from_bytes(cls, data):
        """Deserialize bytes into a Genome instance."""
        # Unpack basic parameters
        basic_size = struct.calcsize('iiiffffffffi')
        basic_params = struct.unpack('iiiffffffffi', data[:basic_size])
        
        # Create instance with basic parameters
        genome = cls(
            max_growth_steps=basic_params[0],
            size_x=basic_params[1],
            size_y=basic_params[2],
            diffusion_rate=basic_params[3],
            division_threshold=basic_params[4],
            cell_differentiation_threshold=basic_params[5],
            axon_growth_threshold=basic_params[6],
            axon_connect_threshold=basic_params[7],
            self_connect_isolated_neurons_fraction=basic_params[8],
            weight_adjustment_target=basic_params[9],
            weight_adjustment_rate=basic_params[10],
            max_axon_length=basic_params[11]
        )
        
        # Calculate sizes for numpy arrays
        num_morphogens = genome.num_morphogens
        offset = basic_size
        
        # Reconstruct fixed-size numpy arrays
        secretion_size = num_morphogens * 8  # 8 bytes per float64
        inhibition_size = num_morphogens * num_morphogens * 8
        
        genome.progenitor_secretion_rates = np.frombuffer(
            data[offset:offset + secretion_size]).reshape(num_morphogens)
        offset += secretion_size
        
        genome.neuron_secretion_rates = np.frombuffer(
            data[offset:offset + secretion_size]).reshape(num_morphogens)
        offset += secretion_size
        
        genome.inhibition_matrix = np.frombuffer(
            data[offset:offset + inhibition_size]).reshape(num_morphogens, num_morphogens)
        offset += inhibition_size
        
        # Reconstruct diffusion patterns
        patterns = []
        shape_size = struct.calcsize('ii')
        for _ in range(num_morphogens):
            # Read shape information
            height, width = struct.unpack('ii', data[offset:offset + shape_size])
            offset += shape_size
            
            # Read pattern data
            pattern_size = height * width * 8  # 8 bytes per float64
            pattern = np.frombuffer(
                data[offset:offset + pattern_size]).reshape(height, width)
            offset += pattern_size
            patterns.append(pattern)
        
        genome.diffusion_patterns = np.array(patterns)
        
        return genome 

    def to_dict(self):
        """Convert genome to a dictionary suitable for JSON serialization."""
        # Convert diffusion patterns to list if it's a numpy array
        patterns = (self.diffusion_patterns.tolist() 
                   if isinstance(self.diffusion_patterns, np.ndarray) 
                   else [p.tolist() if isinstance(p, np.ndarray) else p 
                         for p in self.diffusion_patterns])
        
        return {
            'max_growth_steps': self.max_growth_steps,
            'size_x': self.size_x,
            'size_y': self.size_y,
            'diffusion_rate': self.diffusion_rate,
            'num_morphogens': self.num_morphogens,
            'division_threshold': self.division_threshold,
            'cell_differentiation_threshold': self.cell_differentiation_threshold,
            'axon_growth_threshold': self.axon_growth_threshold,
            'max_axon_length': self.max_axon_length,
            'axon_connect_threshold': self.axon_connect_threshold,
            'self_connect_isolated_neurons_fraction': self.self_connect_isolated_neurons_fraction,
            'weight_adjustment_target': self.weight_adjustment_target,
            'weight_adjustment_rate': self.weight_adjustment_rate,
            'progenitor_secretion_rates': self.progenitor_secretion_rates.tolist(),
            'neuron_secretion_rates': self.neuron_secretion_rates.tolist(),
            'inhibition_matrix': self.inhibition_matrix.tolist(),
            'diffusion_patterns': patterns,
            # We don't serialize the strategy objects as they're replaced with defaults when deserializing
        }

    def to_json(self, filepath=None):
        """
        Convert genome to JSON string or save to file if filepath is provided.
        
        Args:
            filepath: Optional path to save JSON file
        
        Returns:
            JSON string if filepath is None, otherwise None
        """
        import json
        
        data = self.to_dict()
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            return json.dumps(data, indent=2)

    @classmethod
    def from_dict(cls, data, meta_parameters_strategy=None, mutation_strategy=None, crossover_strategy=None):
        """Create a Genome instance from a dictionary.
        
        Args:
            data: Dictionary containing genome parameters
            meta_parameters_strategy: Optional MetaParametersStrategy instance
            mutation_strategy: Optional MutationStrategy instance
            crossover_strategy: Optional CrossoverStrategy instance
        
        Returns:
            Genome instance
        """
        # Convert diffusion patterns individually to handle different sizes
        patterns = [np.array(pattern) for pattern in data['diffusion_patterns']]
        
        return cls(
            max_growth_steps=data['max_growth_steps'],
            size_x=data['size_x'],
            size_y=data['size_y'],
            diffusion_rate=data['diffusion_rate'],
            num_morphogens=data['num_morphogens'],
            division_threshold=data['division_threshold'],
            cell_differentiation_threshold=data['cell_differentiation_threshold'],
            axon_growth_threshold=data['axon_growth_threshold'],
            max_axon_length=data['max_axon_length'],
            axon_connect_threshold=data['axon_connect_threshold'],
            self_connect_isolated_neurons_fraction=data['self_connect_isolated_neurons_fraction'],
            weight_adjustment_target=data['weight_adjustment_target'],
            weight_adjustment_rate=data['weight_adjustment_rate'],
            progenitor_secretion_rates=np.array(data['progenitor_secretion_rates']),
            neuron_secretion_rates=np.array(data['neuron_secretion_rates']),
            inhibition_matrix=np.array(data['inhibition_matrix']),
            diffusion_patterns=patterns,
            meta_parameters_strategy=meta_parameters_strategy,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy
        )

    @classmethod
    def from_json(cls, json_str=None, filepath=None, meta_parameters_strategy=None, 
                  mutation_strategy=None, crossover_strategy=None):
        """
        Create a Genome instance from JSON string or file.
        
        Args:
            json_str: JSON string (mutually exclusive with filepath)
            filepath: Path to JSON file (mutually exclusive with json_str)
            meta_parameters_strategy: Optional MetaParametersStrategy instance
            mutation_strategy: Optional MutationStrategy instance
            crossover_strategy: Optional CrossoverStrategy instance
        
        Returns:
            Genome instance
        
        Raises:
            ValueError: If both json_str and filepath are provided
        """
        import json
        
        if json_str is not None and filepath is not None:
            raise ValueError("Provide either json_str or filepath, not both")
        
        if filepath:
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(json_str)
        
        return cls.from_dict(
            data,
            meta_parameters_strategy=meta_parameters_strategy,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy
        )

    @staticmethod
    def _generate_random_morphogen_matrices(rng, num_morphogens, meta_params=None):
        """Generate random secretion rates and inhibition matrix.
        
        Args:
            rng: Random number generator instance
            num_morphogens: Number of morphogens
            meta_params: Optional MetaParameters instance (uses default if None)
            
        Returns:
            Tuple of (progenitor_secretion_rates, neuron_secretion_rates, inhibition_matrix)
        """
        if meta_params is None:
            # Use default meta-parameters
            strategy = DefaultMetaParametersStrategy()
            meta_params = MetaParameters(strategy.get_parameters())
            
        progenitor_secretion_rates = rng.uniform(meta_params.RANDOM_SECRETION_RANGE[0], 
                                               meta_params.RANDOM_SECRETION_RANGE[1], 
                                               size=num_morphogens)
        neuron_secretion_rates = rng.uniform(meta_params.RANDOM_SECRETION_RANGE[0], 
                                           meta_params.RANDOM_SECRETION_RANGE[1], 
                                           size=num_morphogens)
        
        inhibition_matrix = rng.uniform(meta_params.RANDOM_INHIBITION_RANGE[0], 
                                      meta_params.RANDOM_INHIBITION_RANGE[1], 
                                      size=(num_morphogens, num_morphogens))
        np.fill_diagonal(inhibition_matrix, 0)  # No self-inhibition
        
        # Each row must have one to num_morphogens zero values
        for i in range(num_morphogens):
            num_zeros = rng.integers(1, num_morphogens + 1)
            zero_indices = rng.choice(num_morphogens, size=num_zeros, replace=False)
            inhibition_matrix[i, zero_indices] = 0
            
        return progenitor_secretion_rates, neuron_secretion_rates, inhibition_matrix

    @staticmethod
    def _generate_random_diffusion_pattern(rng, meta_params=None):
        """Generate a single random 3x3 diffusion pattern that sums to 1."""
        if meta_params is None:
            # Use default meta-parameters
            strategy = DefaultMetaParametersStrategy()
            meta_params = MetaParameters(strategy.get_parameters())
            
        pattern = rng.uniform(meta_params.RANDOM_DIFFUSION_PATTERN_RANGE[0], 
                            meta_params.RANDOM_DIFFUSION_PATTERN_RANGE[1], 
                            size=(3, 3))
        return pattern / pattern.sum()  # Normalize

    @staticmethod
    def random(rng, size_x=20, size_y=20, num_morphogens=3, max_growth_steps=500, 
               meta_parameters_strategy=None, mutation_strategy=None, crossover_strategy=None):
        """
        Generate a random genome with valid parameters.
        
        Args:
            rng: Random number generator instance (numpy.random.Generator)
            size_x: Grid width (default: 20) or array of possible values
            size_y: Grid height (default: 20) or array of possible values
            num_morphogens: Number of morphogens (default: A 3) or array of possible values
            max_growth_steps: Maximum number of growth steps (default: 500) or array of possible values
            meta_parameters_strategy: MetaParametersStrategy instance (optional, uses default if None)
            mutation_strategy: MutationStrategy instance (optional, uses default if None)
            crossover_strategy: CrossoverStrategy instance (optional, uses default if None)
            
        Returns:
            Genome instance with randomized parameters
        """
        if meta_parameters_strategy is None:
            meta_parameters_strategy = DefaultMetaParametersStrategy()
        meta_params = MetaParameters(meta_parameters_strategy.get_parameters())
        
        # Handle array parameters
        def choose_value(param, default):
            if param is None:
                return default
            elif isinstance(param, (list, tuple, np.ndarray)):
                return int(rng.choice(param))
            else:
                return int(param)
        
        chosen_size_x = choose_value(size_x, int(rng.integers(*meta_params.RANDOM_GRID_SIZE_RANGE)))
        chosen_size_y = choose_value(size_y, int(rng.integers(*meta_params.RANDOM_GRID_SIZE_RANGE)))
        chosen_steps = choose_value(max_growth_steps, int(rng.integers(*meta_params.RANDOM_GROWTH_STEPS_RANGE)))
        
        params = {
            'max_growth_steps': chosen_steps,
            'size_x': chosen_size_x,
            'size_y': chosen_size_y,
            'diffusion_rate': rng.uniform(*meta_params.RANDOM_DIFFUSION_RATE_RANGE),
            'division_threshold': rng.uniform(*meta_params.RANDOM_DIVISION_THRESHOLD_RANGE),
            'cell_differentiation_threshold': rng.uniform(*meta_params.RANDOM_DIFFERENTIATION_THRESHOLD_RANGE),
            'axon_growth_threshold': rng.uniform(*meta_params.RANDOM_AXON_GROWTH_THRESHOLD_RANGE),
            'axon_connect_threshold': rng.uniform(*meta_params.RANDOM_AXON_CONNECT_THRESHOLD_RANGE),
            'self_connect_isolated_neurons_fraction': rng.uniform(*meta_params.RANDOM_SELF_CONNECT_RANGE),
            'weight_adjustment_target': rng.uniform(*meta_params.RANDOM_WEIGHT_TARGET_RANGE),
            'weight_adjustment_rate': rng.uniform(*meta_params.RANDOM_WEIGHT_RATE_RANGE),
            'max_axon_length': int(rng.integers(*meta_params.RANDOM_AXON_LENGTH_RANGE))
        }
        
        # Handle num_morphogens selection
        if num_morphogens is None:
            # Use MORPHOGEN_PROBS for morphogen selection
            choices, probs = zip(*meta_params.MORPHOGEN_PROBS.items())
            num_morphogens = int(rng.choice(choices, p=probs))
        elif isinstance(num_morphogens, (list, tuple, np.ndarray)):
            # Choose randomly from provided values with equal probability
            num_morphogens = int(rng.choice(num_morphogens))
        else:
            # Use the provided integer value
            num_morphogens = int(num_morphogens)

        # Generate secretion rates and inhibition matrix
        (progenitor_secretion_rates, 
         neuron_secretion_rates, 
         inhibition_matrix) = Genome._generate_random_morphogen_matrices(rng, num_morphogens, meta_params)
        
        # Generate random diffusion patterns
        diffusion_patterns = [
            Genome._generate_random_diffusion_pattern(rng, meta_params) 
            for _ in range(num_morphogens)
        ]
        
        return Genome(
            max_growth_steps=params['max_growth_steps'],
            size_x=params['size_x'],
            size_y=params['size_y'],
            num_morphogens=num_morphogens,
            diffusion_rate=params['diffusion_rate'],
            division_threshold=params['division_threshold'],
            cell_differentiation_threshold=params['cell_differentiation_threshold'],
            axon_growth_threshold=params['axon_growth_threshold'],
            max_axon_length=params['max_axon_length'],
            axon_connect_threshold=params['axon_connect_threshold'],
            self_connect_isolated_neurons_fraction=params['self_connect_isolated_neurons_fraction'],
            weight_adjustment_target=params['weight_adjustment_target'],
            weight_adjustment_rate=params['weight_adjustment_rate'],
            progenitor_secretion_rates=progenitor_secretion_rates,
            neuron_secretion_rates=neuron_secretion_rates,
            inhibition_matrix=inhibition_matrix,
            diffusion_patterns=diffusion_patterns,
            meta_parameters_strategy=meta_parameters_strategy,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy
        )

    def mutate(self, rng):
        """
        Mutate the genome using the instance's mutation strategy.
        
        Args:
            rng: Random number generator instance
            
        Returns:
            New mutated Genome instance
        """
        return self.mutation_strategy.mutate(self, rng)

    @classmethod
    def crossover(cls, parent1, parent2, rng):
        """
        Perform crossover between two parent genomes using parent1's crossover strategy.
        
        Args:
            parent1: First parent Genome instance
            parent2: Second parent Genome instance
            rng: Random number generator instance
            
        Returns:
            New Genome instance combining traits from both parents
        """
        return parent1.crossover_strategy.crossover(parent1, parent2, rng)

    @staticmethod
    def _normalize_integer(value, min_val, mid_val, max_val):
        """
        Normalize an integer value to [0,1] range using a piecewise linear mapping.
        
        Args:
            value: Integer value to normalize
            min_val: Minimum value in the range
            mid_val: Middle value that maps to 0.9
            max_val: Maximum value in the range
            
        Returns:
            Normalized float value in [0,1] range
        """
        if value <= mid_val:
            # Map [min_val, mid_val] to [0, 0.9]
            return (value - min_val) * 0.9 / (mid_val - min_val)
        else:
            # Map [mid_val, max_val] to [0.9, 1.0]
            return 0.9 + (value - mid_val) * 0.1 / (max_val - mid_val)

    @staticmethod
    def _denormalize_integer(norm_value, min_val, mid_val, max_val):
        """
        Denormalize a value from [0,1] range back to integer range.
        
        Args:
            norm_value: Normalized value in [0,1] range
            min_val: Minimum value in the range
            mid_val: Middle value that maps to 0.9
            max_val: Maximum value in the range
            
        Returns:
            Denormalized integer value
        """
        # Ensure norm_value is within [0,1]
        norm_value = max(0.0, min(1.0, float(norm_value)))
        
        if np.isclose(norm_value, 1.0, rtol=1e-5):
            return max_val
        elif norm_value < 0.9:
            # Map [0, 0.9] back to [min_val, mid_val]
            value = min_val + norm_value * (mid_val - min_val) / 0.9
        else:
            # Map [0.9, 1.0] back to [mid_val, max_val]
            value = mid_val + (norm_value - 0.9) * (max_val - mid_val) / 0.1
        
        # Ensure we get a proper positive integer
        return max(1, min(max_val, int(np.round(value))))

    def flatten(self):
        """
        Convert the genome into a 1-dimensional array of floats in the 0-1 range.
        
        Returns:
            numpy.ndarray: 1D array containing all genome parameters normalized to [0,1]
        """
        # Normalize integer parameters
        max_growth_steps_norm = self._normalize_integer(self.max_growth_steps, 0, 200, 500)
        size_x_norm = self._normalize_integer(self.size_x, 10, 40, 200)
        size_y_norm = self._normalize_integer(self.size_y, 10, 40, 200)
        max_axon_length_norm = self._normalize_integer(self.max_axon_length, 1, 3, 10)
            
        # Collect all parameters in order
        params = [
            max_growth_steps_norm,
            size_x_norm,
            size_y_norm,
            self.diffusion_rate,  # Already in 0-1 range
            self.division_threshold,  # Already in 0-1 range
            self.cell_differentiation_threshold,  # Already in 0-1 range
            self.axon_growth_threshold,  # Already in 0-1 range
            max_axon_length_norm,
            self.axon_connect_threshold,  # Already in 0-1 range
            self.self_connect_isolated_neurons_fraction,  # Already in 0-1 range
            self.weight_adjustment_target,  # Already in 0-1 range
            self.weight_adjustment_rate,  # Already in 0-1 range
        ]
        
        # Add arrays
        params.extend(self.progenitor_secretion_rates)  # Already in 0-1 range
        params.extend(self.neuron_secretion_rates)  # Already in 0-1 range
        params.extend(self.inhibition_matrix.flatten())  # Already in 0-1 range
        
        # Add diffusion patterns
        for pattern in self.diffusion_patterns:
            params.extend(pattern.flatten())  # Already normalized to sum to 1
        
        return np.array(params, dtype=np.float64)

    @classmethod
    def from_flattened(cls, flattened_array, meta_parameters_strategy=None, 
                      mutation_strategy=None, crossover_strategy=None):
        """
        Create a Genome instance from its flattened representation.
        
        Args:
            flattened_array: 1D numpy array of floats in [0,1] range
            meta_parameters_strategy: Optional MetaParametersStrategy instance
            mutation_strategy: Optional MutationStrategy instance
            crossover_strategy: Optional CrossoverStrategy instance
            
        Returns:
            Genome instance
            
        Raises:
            ValueError: If the flattened array has invalid length or contains values outside [0,1]
        """
        # Validate input range
        if not np.all((flattened_array >= 0) & (flattened_array <= 1)):
            raise ValueError("All values in flattened array must be in [0,1] range")
            
        # Calculate number of morphogens from array length
        # The formula is based on the structure of the flattened array:
        # 12 basic parameters + 2*num_morphogens (secretion rates) + 
        # num_morphogens^2 (inhibition matrix) + 9*num_morphogens (diffusion patterns)
        # So: 12 + 2*n + n^2 + 9*n = flattened_array.size
        # Solving for n: n^2 + 11*n + 12 - flattened_array.size = 0
        a = 1
        b = 11
        c = 12 - flattened_array.size
        num_morphogens = int((-b + np.sqrt(b*b - 4*a*c)) / (2*a))
        
        if num_morphogens < 3:
            raise ValueError(f"Invalid flattened array size {flattened_array.size}. "
                           f"Calculated number of morphogens {num_morphogens} is less than minimum required 3.")
        
        # Extract and denormalize basic parameters
        idx = 0
        
        # Denormalize integer parameters
        max_growth_steps = cls._denormalize_integer(flattened_array[idx], 0, 200, 500); idx += 1
        size_x = cls._denormalize_integer(flattened_array[idx], 10, 40, 200); idx += 1
        size_y = cls._denormalize_integer(flattened_array[idx], 10, 40, 200); idx += 1
        diffusion_rate = flattened_array[idx]; idx += 1
        division_threshold = flattened_array[idx]; idx += 1
        cell_differentiation_threshold = flattened_array[idx]; idx += 1
        axon_growth_threshold = flattened_array[idx]; idx += 1
        max_axon_length = cls._denormalize_integer(flattened_array[idx], 1, 3, 10); idx += 1
        axon_connect_threshold = flattened_array[idx]; idx += 1
        self_connect_isolated_neurons_fraction = flattened_array[idx]; idx += 1
        weight_adjustment_target = flattened_array[idx]; idx += 1
        weight_adjustment_rate = flattened_array[idx]; idx += 1
        
        # Extract arrays
        progenitor_secretion_rates = flattened_array[idx:idx + num_morphogens]; idx += num_morphogens
        neuron_secretion_rates = flattened_array[idx:idx + num_morphogens]; idx += num_morphogens
        inhibition_matrix = flattened_array[idx:idx + num_morphogens*num_morphogens].reshape(num_morphogens, num_morphogens); idx += num_morphogens*num_morphogens
        
        # Extract and renormalize diffusion patterns
        diffusion_patterns = []
        for _ in range(num_morphogens):
            pattern = flattened_array[idx:idx + 9].reshape(3, 3)
            # Renormalize to sum to 1
            pattern = pattern / pattern.sum()
            diffusion_patterns.append(pattern)
            idx += 9
        
        # Create and return the genome
        return cls(
            max_growth_steps=max_growth_steps,
            size_x=size_x,
            size_y=size_y,
            diffusion_rate=diffusion_rate,
            num_morphogens=num_morphogens,
            division_threshold=division_threshold,
            cell_differentiation_threshold=cell_differentiation_threshold,
            axon_growth_threshold=axon_growth_threshold,
            max_axon_length=max_axon_length,
            axon_connect_threshold=axon_connect_threshold,
            self_connect_isolated_neurons_fraction=self_connect_isolated_neurons_fraction,
            weight_adjustment_target=weight_adjustment_target,
            weight_adjustment_rate=weight_adjustment_rate,
            progenitor_secretion_rates=progenitor_secretion_rates,
            neuron_secretion_rates=neuron_secretion_rates,
            inhibition_matrix=inhibition_matrix,
            diffusion_patterns=diffusion_patterns,
            meta_parameters_strategy=meta_parameters_strategy,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy
        ) 