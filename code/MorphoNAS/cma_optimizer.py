import numpy as np
import cma
from .genome import Genome
import multiprocessing
from .optimizer import Optimizer

# Set start method to 'spawn' for consistent behavior across platforms
multiprocessing.set_start_method('spawn', force=True)

class CMAESOptimizer(Optimizer):
    """
    A class to manage CMA-ES optimization of genomes.
    
    Args:
        fitness_fn (callable): Function that takes a Genome and returns its fitness score
        initial_sigma (float): Initial step size for CMA-ES
        population_size (int): Size of the population (lambda in CMA-ES)
        seed (int, optional): Random seed for reproducibility
        max_workers (int, optional): Maximum number of workers for parallel fitness evaluation
        convergence_strategy (ConvergenceStrategy, optional): Strategy for determining convergence
        num_morphogens (int, optional): Number of morphogens to use in genomes
    """
    
    def __init__(self,
                 fitness_fn,
                 initial_sigma=0.5,
                 population_size=100,
                 seed=None,
                 max_workers=None,
                 convergence_strategy=None,
                 num_morphogens=None):
        super().__init__(fitness_fn, seed, max_workers, convergence_strategy)
        self.initial_sigma = initial_sigma
        self.population_size = population_size
        self.num_morphogens = num_morphogens
        
        # Initialize CMA-ES
        self.es = None
        
        # Initialize with a random genome
        initial_genome = Genome.random(self.rng, num_morphogens=self.num_morphogens)
        self._initialize_optimizer(initial_genome)
        # Evaluate initial genome
        initial_fitness = self.fitness_fn(initial_genome)
        self._best_genome = initial_genome
        self._best_fitness = initial_fitness
        self._eval_count += 1
        self._current_fitness_scores = [initial_fitness]
    
    def _initialize_optimizer(self, initial_genome):
        """Initialize CMA-ES with the flattened genome."""
        # Get flattened genome as initial point
        x0 = initial_genome.flatten()
        
        # Get problem dimension
        dim = len(x0)
        
        # Set bounds for all parameters to [0,1]
        bounds = [0.0, 1.0]  # Single pair of bounds for all dimensions
        
        # Initialize CMA-ES
        self.es = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=self.initial_sigma,
            inopts={
                'popsize': self.population_size,
                'seed': self.rng.integers(0, 2**32) if self.rng else None,
                'CMA_diagonal': True,  # Use diagonal covariance matrix for efficiency
                'CMA_elitist': True,   # Use elitist selection
                'bounds': bounds,      # Set bounds for all parameters
            }
        )
    
    def step(self):
        """Perform one generation of CMA-ES optimization."""
        # Get new solutions from CMA-ES
        solutions = self.es.ask()
        
        # Evaluate solutions
        fitness_scores = self._evaluate_solutions(solutions)
        
        # Update CMA-ES with fitness scores
        self.es.tell(solutions, -fitness_scores)  # Negative because CMA-ES minimizes
        
        # Check convergence
        if self.convergence_strategy and self.convergence_strategy.should_converge(
            self.generation,
            self.best_fitness,
            np.mean(fitness_scores),
            self.es.countiter
        ):
            return False
        
        self.generation += 1
        return True
    
    @property
    def current_population(self):
        """Get the current population as genomes."""
        if self.es is None:
            return []
        solutions = self.es.ask()
        # Clip solutions to [0,1] range
        solutions = np.clip(solutions, 0.0, 1.0)
        return [Genome.from_flattened(x) for x in solutions]
    
    def _evaluate_solutions(self, solutions):
        """Evaluate a batch of solutions in parallel."""
        # Clip solutions to [0,1] range before converting to genomes
        solutions = np.clip(solutions, 0.0, 1.0)
        
        # Convert solutions to genomes
        genomes = [Genome.from_flattened(x) for x in solutions]
        
        # First check cache for all genomes
        fitness_scores = []
        uncached_genomes = []
        uncached_indices = []
        
        for i, genome in enumerate(genomes):
            genome_bytes = genome.to_bytes()
            if genome_bytes in self.fitness_cache:
                fitness_scores.append(self.fitness_cache[genome_bytes])
            else:
                uncached_genomes.append(genome)
                uncached_indices.append(i)
                fitness_scores.append(None)  # Placeholder
        
        # If there are uncached genomes, evaluate them in parallel
        if uncached_genomes:
            uncached_fitness_scores = self._evaluate_uncached_genomes(uncached_genomes)
            
            # Update cache and fitness scores
            for idx, genome, fitness in zip(uncached_indices, uncached_genomes, uncached_fitness_scores):
                genome_bytes = genome.to_bytes()
                self.fitness_cache[genome_bytes] = fitness
                fitness_scores[idx] = fitness
                
                # Update best solution if necessary
                if fitness > self._best_fitness:
                    self._best_fitness = fitness
                    self._best_genome = genome
        
        # Store current generation's fitness scores
        self._current_fitness_scores = fitness_scores
        return np.array(fitness_scores)
    
    @property
    def best_solution(self):
        """Get the best genome found so far."""
        return self._best_genome
    
    @property
    def current_fitness_scores(self):
        """Get fitness scores for current population."""
        return self._current_fitness_scores
    
    @property
    def evaluation_count(self):
        """Get the total number of fitness evaluations performed."""
        return self._eval_count 