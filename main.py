import time
import numpy as np
import matplotlib.pyplot as plt

from functions.testFunctions import quadraticFunction, rosenbrockFunction, rastriginFunction
from optimizers.gradient import GradientDescentMinimizer
from optimizers.particleSwarm import ParticleSwarmMinimizer
from optimizers.downhillSimplex import DownhillSimplexMinimizer


def find_n_count_global_minimum(function, bounds, n_runs=100):
    optimizers = ["GradientDescentMinimizer", "ParticleSwarmMinimizer", "DownhillSimplexMinimizer", "DownhillSimplexMinimizer_Multiple"]
    global_min = 0

    func = function.func
    grad_func = function.grad_func

    gd_timing = []
    ps_timing = []
    ds_timing = []
    ds_multi_timing = []

    # run each optimizer 100 times and count the number of times the global minimum is found
    for optim in optimizers:
        count = 0
        for _ in range(n_runs):
            initial_point = np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=2)
            match optim:
                case "GradientDescentMinimizer":
                    optimizer = GradientDescentMinimizer
                    minimizer = optimizer(step_size=0.01, max_iterations=1000, tolerance=1e-6, statistics=False)
                    start_time = time.time()
                    x_min, func_min, history = minimizer.minimize(func, grad_func, initial_point)
                    gd_timing.append(time.time() - start_time)
                case "ParticleSwarmMinimizer":
                    optimizer = ParticleSwarmMinimizer
                    minimizer = optimizer(num_particles=30, inertia=0.5, cognitive=1.0, social=1.0, max_iterations=1000, tolerance=1e-6, statistics=False)
                    start_time = time.time()
                    x_min, func_min, history, all_trajectories = minimizer.minimize(func, bounds)
                    ps_timing.append(time.time() - start_time)
                case "DownhillSimplexMinimizer":
                    optimizer = DownhillSimplexMinimizer
                    minimizer = optimizer(alpha=1, gamma=2, rho=0.4, sigma=0.25, max_iterations=1000, tolerance=1e-6, statistics=False)
                    start_time = time.time()
                    x_min, func_min, history = minimizer.minimize(lambda p: func(*p), initial_point)
                    ds_timing.append(time.time() - start_time)
                case "DownhillSimplexMinimizer_Multiple":
                    optimizer = DownhillSimplexMinimizer
                    minimizer = optimizer(alpha=1, gamma=2, rho=0.4, sigma=0.25, max_iterations=1000, tolerance=1e-6, statistics=False)
                    start_time = time.time()
                    x_min, func_min, history = minimizer.multiple_minimize(lambda p: func(*p), initial_point, num_restarts=20)
                    ds_multi_timing.append(time.time() - start_time)
                case _:
                    print("Invalid optimizer")
            if np.abs(func_min - global_min) < 1e-6:
                count += 1


        print(optim + " found the global minimum", count, "times", "in", time.time() - start_time, "seconds")
        # print statistics
        print("Average time:", (time.time() - start_time) / n_runs)
    return gd_timing, ps_timing, ds_timing, ds_multi_timing



def evaluate():
    optimizers = [GradientDescentMinimizer, ParticleSwarmMinimizer, DownhillSimplexMinimizer]
    test_functions = [quadraticFunction(), rosenbrockFunction(), rastriginFunction()]

    for test_function in test_functions:
        print("--------------------")
        print("Function:", test_function.__class__.__name__)
        for optimizer in optimizers:
            print("Optimizer:", optimizer.__name__)
            initial_point = np.array([-2, 4])
            bounds = ((-10, 10), (-10, 10))

            func = test_function.func
            grad_func = test_function.grad_func

            match optimizer.__name__:
                case "GradientDescentMinimizer":
                    if test_function.__class__.__name__ == "quadraticFunction":
                        step_size = 0.8
                    elif test_function.__class__.__name__ == "rosenbrockFunction":
                        step_size = 0.0005
                    else:
                        step_size = 0.001
                    minimizer = optimizer(step_size=step_size, max_iterations=100, tolerance=1e-6)
                    x_min, func_min, history_gd = minimizer.minimize(func, grad_func, initial_point)

                case "ParticleSwarmMinimizer":
                    minimizer = optimizer(num_particles=30, inertia=0.5, cognitive=1.0, social=1.0, max_iterations=100, tolerance=1e-6)
                    x_min, func_min, history_ps, all_trajectories = minimizer.minimize(func, bounds)

                case "DownhillSimplexMinimizer":
                    minimizer = optimizer(alpha=1, gamma=2, rho=0.4, sigma=0.4, max_iterations=100, tolerance=1e-6)
                    x_min, func_min, history_ds = minimizer.minimize(lambda p: func(*p), initial_point)
            
            print("Best position:", x_min)
            print("Best value:", func_min)

        # plot comparison of history in log scale
        plt.figure()
        for i, history in enumerate([history_gd, history_ps, history_ds]):
            plt.plot(range(len(history)), [h[1] for h in history], label=optimizers[i].__name__)
        # plt.yscale('log')
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.title('Convergence with function: ' + test_function.__class__.__name__)
        plt.legend()
        function_values_gd = [h[1] for h in history_gd] + [h[1] for h in history_ps] + [h[1] for h in history_ds]
        function_values_ps = [h[1] for h in history_ps] + [h[1] for h in history_gd] + [h[1] for h in history_ds]
        function_values_ds = [h[1] for h in history_ds] + [h[1] for h in history_gd] + [h[1] for h in history_ps]
        function_values = np.array([function_values_gd, function_values_ps, function_values_ds])
        # if np.max(function_values) > 1e8:
        #     plt.ylim(np.min(function_values), 1e8)
        # else:
        #     plt.ylim(np.min(function_values), np.max(function_values))
        plt.savefig('convergence_' + test_function.__class__.__name__ + '.png')
        plt.show()
        plt.close()


# Example usage
if __name__ == "__main__":
    # evaluate()
    gd_timing, ps_timing, ds_timing, ds_multi_timing = find_n_count_global_minimum(rosenbrockFunction(), ((-10, 10), (-10, 10)), n_runs=100)
    # boxplot of timing
    fig, ax = plt.subplots()
    ax.boxplot([gd_timing, ps_timing, ds_timing, ds_multi_timing])
    ax.set_xticklabels(['Gradient Descent', 'Particle Swarm', 'Downhill Simplex', 'Downhill Simplex Multiple'])
    ax.set_ylabel('Time per run (s)')
    ax.set_title('Timing of Optimizers (100 runs each)')
    ax.grid()
    plt.savefig('timing.png', dpi=400)
    plt.show()


    