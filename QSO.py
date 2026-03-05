
import numpy as np
import time


def QSO(X, fitness_func, lb, ub , max_iter):
    # Quokka Swarm Optimization (QSO)
    bounds = lb, ub
    n_quokkas, dim = X.shape
    D = np.random.rand(n_quokkas, dim)  # Initial droughts

    # Initialize environmental factors
    T = np.random.uniform(0.2, 0.44)
    H = np.random.uniform(0.3, 0.65)
    N = np.random.uniform(0, 1)

    # Evaluate initial fitness
    fitness = np.array([fitness_func(ind) for ind in X])
    start_time = time.time()
    convergence = np.zeros(max_iter)
    for iter in range(max_iter):
        # Identify the leader
        leader_idx = np.argmin(fitness)
        leader = X[leader_idx]

        for i in range(n_quokkas):
            w = np.random.rand(dim)
            X = leader - X[i]
            rand = np.random.rand(dim)

            # Update drought
            D_new = (T + H) / (0.8 * D[i]) + w * rand * X

            # Update position
            X[i] = X[i] + D_new * N

            # Ensure bounds
            bounds = [(-5.12, 5.12)] * dim  # or whatever range you want
            lower_bounds = np.array([b[0] for b in bounds])
            upper_bounds = np.array([b[1] for b in bounds])
            X[i] = np.clip(X[i], lb[i], ub[i])
            # Save updated drought
            D[i] = D_new

        # Return best solution
        best_idx = np.argmin(fitness)
        best_score = fitness[best_idx]
        best_solution = X[best_idx].copy()
        convergence[iter] = best_score
        elapsed_time = time.time() - start_time
        return best_score, convergence, best_solution, elapsed_time