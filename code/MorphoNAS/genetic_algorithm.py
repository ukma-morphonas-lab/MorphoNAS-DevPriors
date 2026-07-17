import numpy as np
from .genome import Genome
from .optimizer import Optimizer

try:
    from .convergence_strategies import DefaultConvergenceStrategy
    from .selection_strategies import SelectionStrategy, TopPercentageSelectionStrategy, TournamentSelectionStrategy, TopFitnessSelectionStrategy, CombinedSelectionStrategy
except ImportError:
    from .convergence_strategies import DefaultConvergenceStrategy
    from .selection_strategies import SelectionStrategy, TopPercentageSelectionStrategy, TournamentSelectionStrategy, TopFitnessSelectionStrategy, CombinedSelectionStrategy

class GeneticAlgorithm(Optimizer):
    """
    A class to manage evolutionary optimization of genomes.
    
    Args:
        population_size (int): Size of the population
        max_generations (int): Maximum number of generations to evolve
        fitness_fn (callable): Function that takes a Genome and returns its fitness score
        grid_size_x (int, optional): Grid width for genomes. Defaults to 20
        grid_size_y (int, optional): Grid height for genomes. Defaults to 20
        num_morphogens (int, optional): Number of morphogens. Defaults to 3
        selection_pressure (float, optional): Fraction of population to select as parents. Defaults to 0.2
        mutation_rate (float, optional): Probability of mutation for offspring. Defaults to 0.3
        seed (int, optional): Random seed for reproducibility. Defaults to None
        max_workers (int, optional): Maximum number of workers for parallel fitness evaluation. Defaults to None (use all available CPU cores)
        use_elitism (bool, optional): Whether to use elitism. Defaults to False
        use_steady_state (bool, optional): Whether to use steady state. Defaults to False
        use_tournament (bool, optional): Whether to use tournament selection. Defaults to False
        num_elite (int, optional): Number of elite individuals to keep. Defaults to 2
        steady_state_replace_rate (float, optional): Rate of replacement in steady state. Defaults to 0.1
        tournament_size (int, optional): Size of tournament for tournament selection. Defaults to 3
        convergence_strategy (ConvergenceStrategy, optional): Strategy for determining convergence. Defaults to DefaultConvergenceStrategy
        elitism_strategy (ElitismStrategy, optional): Strategy for selecting elite individuals. Defaults to TopFitnessSelectionStrategy
        parents_selection_strategy (SelectionStrategy, optional): Strategy for selecting parents. Defaults to TopPercentageSelectionStrategy
        steady_state_strategy (SelectionStrategy, optional): Strategy for managing steady state. Defaults to None
    """
    
    def __init__(self, 
                 population_size, 
                 max_generations, 
                 fitness_fn,
                 grid_size_x=20,
                 grid_size_y=20,
                 num_morphogens=None,
                 max_growth_steps=200,
                 selection_pressure=0.2,
                 mutation_rate=0.3,
                 seed=None,
                 max_workers=None,
                 use_elitism=False,
                 use_steady_state=False,
                 use_tournament=False,
                 num_elite=2,
                 steady_state_replace_rate=0.1,
                 tournament_size=3,
                 convergence_strategy=None,
                 elitism_strategy=None,
                 parents_selection_strategy=None,
                 steady_state_strategy=None):
        super().__init__(fitness_fn, seed, max_workers, convergence_strategy)
        self.population_size = population_size
        self.max_generations = max_generations
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_morphogens = num_morphogens
        self.max_growth_steps = max_growth_steps
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.use_elitism = use_elitism
        self.use_steady_state = use_steady_state
        self.use_tournament = use_tournament
        
        # Configure strategy parameters
        self.steady_state_replace_rate = steady_state_replace_rate if use_steady_state else 1.0
        self.num_elite = num_elite if use_elitism else 0
        
        # Set up strategies
        if use_elitism:
            self.elitism_strategy = elitism_strategy if elitism_strategy is not None else TopFitnessSelectionStrategy(selection_size=self.num_elite, rng=self.rng)
        else:
            self.elitism_strategy = None
        
        # Set up selection strategy
        if parents_selection_strategy is not None:
            self.parents_selection_strategy = parents_selection_strategy
        elif use_tournament:
            self.parents_selection_strategy = TournamentSelectionStrategy(
                tournament_size=tournament_size,
                selection_pressure=selection_pressure,
                rng=self.rng
            )
        else:
            self.parents_selection_strategy = TopPercentageSelectionStrategy(
                selection_pressure=selection_pressure
            )
        
        # Set up steady state strategy
        if steady_state_strategy is not None:
            self.steady_state_strategy = steady_state_strategy
        elif use_steady_state:
            # Calculate the number of individuals to keep (excluding elite)
            num_to_replace = max(2, int(self.population_size * self.steady_state_replace_rate))
            num_to_keep = self.population_size - num_to_replace
            if self.use_elitism:
                num_to_keep -= self.num_elite  # Adjust for elite individuals
            
            # Create steady state strategy using TopFitnessSelectionStrategy
            steady_state_strategy = TopFitnessSelectionStrategy(
                selection_size=num_to_keep,
                rng=self.rng
            )
            
            # Create combined strategy if elitism is enabled
            if self.use_elitism:
                self.steady_state_strategy = CombinedSelectionStrategy(
                    strategies=[self.elitism_strategy, steady_state_strategy],
                    exclude_previous=True
                )
            else:
                self.steady_state_strategy = steady_state_strategy
        else:
            self.steady_state_strategy = None
        
        # Initialize population
        self.population = []
        self.fitness_scores = []
    
    def _initialize_population(self):
        """Generate initial random population."""
        print(f"Initializing population of size {self.population_size}...")
        self.population = [
            Genome.random(self.rng, 
                         size_x=self.grid_size_x,
                         size_y=self.grid_size_y,
                         num_morphogens=self.num_morphogens,
                         max_growth_steps=self.max_growth_steps)
            for _ in range(self.population_size)
        ]
        
        # Evaluate initial population
        self._evaluate_population()
        print(f"Initial population evaluated. Best fitness: {self.best_fitness}; evaluations: {self.evaluation_count}")
    
    def _evaluate_population(self):
        """Evaluate fitness for all genomes in the population."""
        self.fitness_scores = self._evaluate_solutions(self.population)
    
    def _create_offspring(self, parents, target_size):
        """Create offspring through crossover and mutation until reaching target size."""
        offspring = []
        while len(offspring) < target_size:
            parent1, parent2 = self.rng.choice(parents, size=2, replace=False)
            child = Genome.crossover(parent1, parent2, self.rng)
            
            # Get effective mutation rate
            effective_mutation_rate = self.mutation_rate
            if hasattr(child.mutation_strategy, 'get_mutation_rate_multiplier'):
                effective_mutation_rate *= child.mutation_strategy.get_mutation_rate_multiplier()
            
            if self.rng.random() < effective_mutation_rate:
                child = child.mutate(self.rng)
            offspring.append(child)
            
        return offspring

    def _create_next_generation(self):
        """Create the next generation combining all enabled strategies."""
        # Select parents using the selection strategy
        parents = self.parents_selection_strategy.select(
            self.population,
            self.fitness_scores
        )
        
        # Get fixed part of population using steady state strategy
        new_population = self.steady_state_strategy.select(self.population, self.fitness_scores) if self.steady_state_strategy is not None else []
        
        # Create remaining offspring
        remaining_size = self.population_size - len(new_population)
        if remaining_size > 0:
            offspring = self._create_offspring(parents, remaining_size)
            new_population.extend(offspring)
        
        return new_population
    
    def step(self):
        """Perform one generation of evolution."""
        if self.generation >= self.max_generations:
            return False
        
        # Initialize population if empty
        if len(self.population) == 0:
            self._initialize_population()
        
        # Create new population using all enabled strategies
        self.population = self._create_next_generation()
        
        # Evaluate new population
        self._evaluate_population()
        
        # Check for convergence and manage population
        avg_fitness = np.mean(self.fitness_scores)
        if self.convergence_strategy.should_converge(
            self.generation, self.best_fitness, avg_fitness, self.max_generations
        ):
            return False
            
        # Apply population management strategy
        self.population, self.fitness_scores = self.convergence_strategy.manage_population(
            self.population,
            self.fitness_scores,
            self.rng,
            self.grid_size_x,
            self.grid_size_y,
            self.num_morphogens,
            self.max_growth_steps
        )
        
        # Re-evaluate if population was modified
        if len(self.population) > 0:
            self._evaluate_population()
        
        self.generation += 1
        return True
    
    @property
    def current_population(self):
        """Get the current population."""
        return self.population

    def get_mutation_rate(self):
        """Get the current mutation rate."""
        return self.mutation_rate