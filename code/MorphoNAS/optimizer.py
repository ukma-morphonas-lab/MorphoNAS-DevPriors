from abc import ABC, abstractmethod
import numpy as np
from threading import Lock
from concurrent.futures import ProcessPoolExecutor

class Optimizer(ABC):
    """
    Base class for optimization algorithms.
    
    Args:
        fitness_fn (callable): Function that takes a Genome and returns its fitness score
        seed (int, optional): Random seed for reproducibility
        max_workers (int, optional): Maximum number of workers for parallel fitness evaluation
        convergence_strategy (ConvergenceStrategy, optional): Strategy for determining convergence
    """
    
    def __init__(self,
                 fitness_fn,
                 seed=None,
                 max_workers=None,
                 convergence_strategy=None):
        self.fitness_fn = fitness_fn
        self.max_workers = max_workers
        self.convergence_strategy = convergence_strategy
        
        # Initialize random number generator
        self.rng = np.random.default_rng(seed)
        
        # Initialize tracking variables
        self.generation = 0
        self._best_genome = None
        self._best_fitness = float('-inf')
        self._eval_count = 0
        self._current_fitness_scores = []  # Track current generation's fitness scores
        
        # Add fitness cache
        self.fitness_cache = {}
        
        # Add evaluation lock
        self._eval_lock = Lock()
    
    def _evaluate_uncached_genomes(self, genomes):
        """Evaluate fitness for multiple genomes in parallel."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.fitness_fn, genomes))
            
        # Update evaluation counter after batch processing
        with self._eval_lock:
            self._eval_count += len(genomes)
            
        return results
    
    def _evaluate_solutions(self, genomes):
        """Evaluate a batch of genomes in parallel."""
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
    
    @abstractmethod
    def step(self):
        """Perform one step of optimization."""
        pass
    
    def run(self, max_generations, callback=None):
        """
        Run the optimization process for max_generations.
        
        Args:
            max_generations (int): Maximum number of generations to run
            callback (callable, optional): Function called after each generation
                                        with (generation, best_fitness, avg_fitness)
                                        Returns True to continue, False to stop
        """
        while self.generation < max_generations:
            if not self.step():
                break
            
            if callback:
                avg_fitness = np.mean(self._current_fitness_scores)
                if not callback(self.generation, self.best_fitness, avg_fitness):
                    break
    
    def get_mutation_rate(self):
        """Get the current mutation rate. Returns 0.0 for optimizers that don't use mutation."""
        return 0.0
    
    @property
    def best_solution(self):
        """Get the best genome found so far."""
        return self._best_genome
    
    @property
    def best_fitness(self):
        """Get the best fitness score found so far."""
        return self._best_fitness
    
    @property
    def current_fitness_scores(self):
        """Get fitness scores for current population."""
        return self._current_fitness_scores
    
    @property
    def evaluation_count(self):
        """Get the total number of fitness evaluations performed."""
        return self._eval_count 