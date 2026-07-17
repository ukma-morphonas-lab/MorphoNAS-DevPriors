from abc import ABC, abstractmethod
import numpy as np

from .genome import Genome

class ConvergenceStrategy(ABC):
    """Base class for genetic algorithm convergence strategies."""
    
    @abstractmethod
    def should_converge(self, generation, best_fitness, avg_fitness, max_generations):
        """
        Determine if the genetic algorithm should converge.
        
        Args:
            generation (int): Current generation number
            best_fitness (float): Best fitness in current population
            avg_fitness (float): Average fitness of current population
            max_generations (int): Maximum number of generations allowed
            
        Returns:
            bool: True if algorithm should converge, False otherwise
        """
        pass
    
    def manage_population(self, population, fitness_scores, rng, grid_size_x, grid_size_y, num_morphogens, max_growth_steps):
        """
        Optionally modify the population to maintain diversity or improve convergence.
        
        Args:
            population (list): Current population of genomes
            fitness_scores (list): Fitness scores for each genome
            rng: Random number generator instance
            grid_size_x (int): Grid width for new genomes
            grid_size_y (int): Grid height for new genomes
            num_morphogens (int): Number of morphogens for new genomes
            max_growth_steps (int): Maximum growth steps for new genomes
            
        Returns:
            tuple: (modified_population, modified_fitness_scores)
        """
        return population, fitness_scores

class DefaultConvergenceStrategy(ConvergenceStrategy):
    """Default convergence strategy that stops when average fitness is >95% of best fitness."""
    
    def __init__(self, convergence_threshold=0.95):
        """
        Initialize the default convergence strategy.
        
        Args:
            convergence_threshold (float): Threshold for convergence as a fraction of best fitness.
                                         Defaults to 0.95 (95%)
        """
        self.convergence_threshold = convergence_threshold
    
    def should_converge(self, generation, best_fitness, avg_fitness, max_generations):
        """
        Check if the algorithm should converge based on fitness ratio.
        
        Args:
            generation (int): Current generation number
            best_fitness (float): Best fitness in current population
            avg_fitness (float): Average fitness of current population
            max_generations (int): Maximum number of generations allowed
            
        Returns:
            bool: True if algorithm should converge, False otherwise
        """
        # Don't converge if we haven't started or if fitness is 0
        if generation == 0 or best_fitness <= 0:
            return False
            
        # Calculate convergence ratio
        convergence_ratio = avg_fitness / best_fitness
        
        # Converge if ratio exceeds threshold
        return convergence_ratio >= self.convergence_threshold

class DiversityMaintenanceStrategy(DefaultConvergenceStrategy):
    """
    Convergence strategy that maintains diversity by replacing some of the worst individuals
    with random ones when the population starts to converge.
    """
    
    def __init__(self, 
                 convergence_threshold=0.95,
                 replacement_fraction=0.1,
                 bottom_fraction=0.5,
                 replacement_threshold=0.9):
        """
        Initialize the diversity maintenance strategy.
        
        Args:
            convergence_threshold (float): Threshold for convergence as a fraction of best fitness.
                                         Defaults to 0.95 (95%)
            replacement_fraction (float): Fraction of population to replace with random individuals.
                                        Defaults to 0.1 (10%)
            bottom_fraction (float): Fraction of worst individuals to consider for replacement.
                                   Defaults to 0.5 (50%)
            replacement_threshold (float): Threshold for when to start replacing individuals.
                                         Defaults to 0.9 (90%)
        """
        super().__init__(convergence_threshold)
        self.replacement_fraction = replacement_fraction
        self.bottom_fraction = bottom_fraction
        self.replacement_threshold = replacement_threshold
    
    def manage_population(self, population, fitness_scores, rng, grid_size_x, grid_size_y, num_morphogens, max_growth_steps):
        """
        Replace some of the worst individuals with random ones to maintain diversity.
        Only replaces individuals if convergence ratio exceeds replacement_threshold.
        
        Args:
            population (list): Current population of genomes
            fitness_scores (list): Fitness scores for each genome
            rng: Random number generator instance
            grid_size_x (int): Grid width for new genomes
            grid_size_y (int): Grid height for new genomes
            num_morphogens (int): Number of morphogens for new genomes
            max_growth_steps (int): Maximum growth steps for new genomes
            
        Returns:
            tuple: (modified_population, modified_fitness_scores)
        """
        # Calculate convergence ratio
        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        convergence_ratio = avg_fitness / best_fitness if best_fitness > 0 else 0
        
        # Only replace individuals if convergence ratio exceeds threshold
        if convergence_ratio >= self.replacement_threshold:
            # Calculate number of individuals to replace
            num_to_replace = max(1, int(len(population) * self.replacement_fraction))
            
            # Get indices of worst individuals (bottom fraction)
            sorted_indices = np.argsort(fitness_scores)
            bottom_indices = sorted_indices[:int(len(population) * self.bottom_fraction)]
            
            # Randomly select individuals to replace from bottom fraction
            replace_indices = rng.choice(bottom_indices, size=num_to_replace, replace=False)
            
            # Create new random individuals
            new_population = population.copy()
            for idx in replace_indices:
                new_population[idx] = Genome.random(
                    rng,
                    size_x=grid_size_x,
                    size_y=grid_size_y,
                    num_morphogens=num_morphogens,
                    max_growth_steps=max_growth_steps
                )
            
            return new_population, fitness_scores
        
        # If not replacing, use default behavior
        return super().manage_population(population, fitness_scores, rng, grid_size_x, grid_size_y, num_morphogens, max_growth_steps) 