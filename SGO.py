import time
import numpy as np

def SGO(num_players, fitness, lower_bound, upper_bound, num_iterations):
    population, problem_dimension = num_players.shape
    best_fitness = np.zeros((1, num_iterations))
    best_solution = np.zeros((1, num_iterations))
    Conv = np.zeros((1, num_iterations))  # Convergance  array
    ct = time.time()
    # while iteration < num_iterations + 1:
    for iteration in range(num_iterations):
        fitness_values = fitness(population)

        # Select the winner based on fitness (lower fitness is better)
        winner_idx = np.argmin(fitness_values)

        # Randomly mutate the winner's strategy
        mutation_factor = np.random.uniform(0.1, 0.5)
        mutation_direction = np.random.choice([-1, 1], size=problem_dimension)
        mutated_solution = population[winner_idx] + mutation_factor * mutation_direction

        # Clip the mutated solution to the bounds
        mutated_solution = np.clip(mutated_solution, lower_bound, upper_bound)

        # Replace the loser with the mutated solution
        loser_idx = np.argmax(fitness_values)
        population[loser_idx] = mutated_solution

        # Print the best fitness value for this iteration
        best_fitness[0, iteration] = np.min(fitness_values)
        # Return the best solution and its fitness value
        best_idx = np.argmin(fitness(population))
        best_solution[0, iteration] = population[best_idx]
        # best_fitness = np.min(fitness(population))
        Conv[0, iteration] = best_fitness  # Update the  convergence curve

    ct = time.time() - ct
    return best_fitness, Conv, best_solution, ct
