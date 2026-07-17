from abc import ABC, abstractmethod
import numpy as np

class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, population, fitness_scores):
        """
        Select individuals from the population.
        
        Args:
            population (list): List of genomes
            fitness_scores (list): List of fitness scores corresponding to population
            
        Returns:
            list: Selected individuals
        """
        pass

class TopPercentageSelectionStrategy(SelectionStrategy):
    """Selects individuals based on top percentage of fitness scores."""
    
    def __init__(self, selection_pressure=0.2):
        """
        Initialize top percentage selection strategy.
        
        Args:
            selection_pressure (float): Fraction of population to select as parents. Defaults to 0.2
        """
        self.selection_pressure = selection_pressure
    
    def select(self, population, fitness_scores):
        """Select individuals by taking top percentage of population based on fitness."""
        num_individuals = int(len(population) * self.selection_pressure)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        return [population[idx] for idx in sorted_indices[:num_individuals]]

class TournamentSelectionStrategy(SelectionStrategy):
    """Selects individuals using tournament selection."""
    
    def __init__(self, tournament_size=3, selection_pressure=0.2, rng=None):
        """
        Initialize tournament selection strategy.
        
        Args:
            tournament_size (int): Number of individuals in each tournament
            selection_pressure (float): Fraction of population to select as parents. Defaults to 0.2
            rng (numpy.random.Generator): Random number generator
        """
        self.tournament_size = tournament_size
        self.selection_pressure = selection_pressure
        self.rng = rng
    
    def select(self, population, fitness_scores):
        """Select individuals using tournament selection."""
        num_individuals = int(len(population) * self.selection_pressure)
        individuals = []
        population_size = len(population)
        
        while len(individuals) < num_individuals:
            # Select random individuals for tournament
            tournament_idx = self.rng.choice(range(population_size), 
                                          size=self.tournament_size, 
                                          replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            individuals.append(population[winner_idx])
        
        return individuals 

class TopFitnessSelectionStrategy(SelectionStrategy):
    """Default elitism strategy that keeps the best selection_size individuals."""
    
    def __init__(self, selection_size=2, rng=None):
        self.selection_size = selection_size
        self.rng = rng
    
    def select(self, population, fitness_scores):
        """Select the best selection_size individuals based on fitness."""
        sorted_indices = np.argsort(fitness_scores)[::-1]
        return [population[idx] for idx in sorted_indices[:self.selection_size]]

class TopDistinctFitnessSelectionStrategy(SelectionStrategy):
    """Elitism strategy that keeps the best individual for each distinct fitness score."""
    
    def __init__(self, selection_size=2, rng=None):
        self.selection_size = selection_size
        self.rng = rng
    
    def select(self, population, fitness_scores):
        """Select the best individual for each distinct fitness score."""
        # Sort indices by fitness in descending order
        sorted_indices = np.argsort(fitness_scores)[::-1]
        seen_fitness = set()
        elite = []
        
        # Keep best individual for each distinct fitness score
        for idx in sorted_indices:
            if len(elite) >= self.selection_size:
                break
            fitness = fitness_scores[idx]
            if fitness not in seen_fitness:
                elite.append(population[idx])
                seen_fitness.add(fitness)
        
        return elite

class CombinedSelectionStrategy(SelectionStrategy):
    """Selection strategy that combines results from multiple selection strategies."""
    
    def __init__(self, strategies, exclude_previous=False):
        """
        Initialize combined selection strategy.
        
        Args:
            strategies (list): List of SelectionStrategy instances to apply
            exclude_previous (bool): If True, exclude individuals selected by previous strategies
        """
        self.strategies = strategies
        self.exclude_previous = exclude_previous
    
    def select(self, population, fitness_scores):
        """
        Apply each selection strategy and combine their results.
        
        Args:
            population (list): List of genomes
            fitness_scores (list): List of fitness scores corresponding to population
            
        Returns:
            list: Combined list of individuals selected by all strategies
        """
        selected_individuals = []
        excluded_indices = set()
        
        for strategy in self.strategies:
            if self.exclude_previous:
                # Create filtered population and fitness scores excluding previously selected individuals
                filtered_population = [genome for i, genome in enumerate(population) 
                                    if i not in excluded_indices]
                filtered_fitness = [score for i, score in enumerate(fitness_scores) 
                                  if i not in excluded_indices]
                
                # Apply strategy to filtered population
                strategy_selected = strategy.select(filtered_population, filtered_fitness)
                
                # Add selected individuals to result
                selected_individuals.extend(strategy_selected)
                
                # Update excluded indices
                for genome in strategy_selected:
                    idx = population.index(genome)
                    excluded_indices.add(idx)
            else:
                # Original behavior - just append results
                strategy_selected = strategy.select(population, fitness_scores)
                selected_individuals.extend(strategy_selected)
        
        return selected_individuals