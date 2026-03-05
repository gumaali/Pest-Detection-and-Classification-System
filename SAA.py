import numpy as np
import time

def SAA(population, func,lb, ub, max_iter=500):
    # Snow Avalanches Algorithm (SAA)
    si = 0.5
    pop_size = population.shape[0]
    dim = population.shape[1]
    fitness = np.apply_along_axis(func, 1, population)
    best_idx = np.argmin(fitness)
    Xbest = population[best_idx].copy()
    fbest = fitness[best_idx]
    convergence = np.zeros((max_iter))
    start_time = time.time()
    for t in range(max_iter):
        for i in range(pop_size):
            # Step 8: Select random members
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            Xi, Xr1, Xr2, Xr3 = population[i], population[r1], population[r2], population[r3]

            rand_val = np.random.rand()

            # Eq. (1): Avalanche due to mountain slope
            if rand_val < si:
                Xnew = Xbest + np.random.rand(dim) * (Xr1 - Xr2)

            # Eq. (2): Avalanche due to human factors
            elif rand_val < 2 * si:
                Xnew = Xr3 + np.random.rand(dim) * (Xr1 - Xr2)

            # Eq. (3): Avalanche due to weather
            elif rand_val < 3 * si:
                Xnew = Xi + np.random.rand(dim) * (Xr1 - Xr2)

            # Eq. (4): Normal conditions
            else:
                Xnew = Xi + np.random.rand(dim) * (ub[i] - lb[i])

            # Apply bounds
            Xnew = np.clip(Xnew, lb[i], ub[i])
            fnew = func(Xnew)
            if fnew < fitness[i]:
                population[i], fitness[i] = Xnew, fnew
            if fnew < fbest:
                Xbest, fbest = Xnew.copy(), fnew
            convergence[t] = fbest
    elapsed_time = time.time() - start_time
    return  fbest,convergence,Xbest, elapsed_time