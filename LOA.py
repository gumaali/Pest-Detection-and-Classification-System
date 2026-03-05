import time
import numpy as np

def LOA(lyrebirds,obj_func,lb, ub, max_iterations):
    # Lyrebird Optimization Algorithm (LOA)
    num_variables,num_lyrebirds = lyrebirds.shape
    crossover_rate = 0.8
    mutation_rate = 0.1
    sigma = 0.1
    fitness = np.array([obj_func(ind) for ind in lyrebirds])
    history = np.zeros(max_iterations)
    best_fitness = lyrebirds[fitness.argmin()]
    best_solution = fitness.min()
    ct = time.time()
    for iteration in range(max_iterations):
        sorted_indices = np.argsort(fitness)
        fitness = fitness[sorted_indices]
        lyrebirds = lyrebirds[sorted_indices, :]

        # Selection (keep top half)
        lyrebirds = lyrebirds[:num_lyrebirds // 2, :]

        # Crossover
        num_crossovers = round(crossover_rate * (num_lyrebirds // 2))
        children = []
        for _ in range(num_crossovers):
            parent1 = lyrebirds[np.random.randint(0, num_lyrebirds // 2)]
            parent2 = lyrebirds[np.random.randint(0, num_lyrebirds // 2)]
            alpha = np.random.rand()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

            children.append(child1)
            children.append(child2)

        if children:
            lyrebirds = np.vstack([lyrebirds] + children)

        # Mutation
        num_mutations = round(mutation_rate * num_lyrebirds)
        for _ in range(num_mutations):
            idx = np.random.randint(0, lyrebirds.shape[0])
            lyrebird = lyrebirds[idx, :]

            mutation = sigma * np.random.randn(num_variables)
            lyrebird = lyrebird + mutation

            # Clip within bounds
            lyrebird = np.clip(lyrebird, lb, ub)
            lyrebirds[idx, :] = lyrebird

        # Track best
        best_solution = lyrebirds[0, :]
        best_fitness = fitness[0]
        history[iteration] = best_fitness
    ct=time.time()-ct
    return best_fitness, history,best_solution,ct