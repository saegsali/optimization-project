import time
import numpy as np
import matplotlib.pyplot as plt
from functions.testFunctions import quadraticFunction, rosenbrockFunction, rastriginFunction
from optimizers.gradient import GradientDescentMinimizer

def evaluate():
    optimizer = GradientDescentMinimizer
    test_functions = [rastriginFunction()]
    step_sizes = [0.001, 0.002, 0.08]  # Different step sizes for testing

    for test_function in test_functions:
        print("--------------------")
        print("Function:", test_function.__class__.__name__)
        initial_point = np.array([-2, 4])

        func = test_function.func
        grad_func = test_function.grad_func

        fig, ax1 = plt.subplots()

        # Create a twin Axes sharing the x-axis
        ax2 = ax1.twinx()

        for i, step_size in enumerate(step_sizes):
            minimizer = optimizer(step_size=step_size, max_iterations=100, tolerance=1e-6)
            x_min, func_min, history_gd = minimizer.minimize(func, grad_func, initial_point)
            
            print(f"Step size: {step_size}")
            print("Best position:", x_min)
            print("Best value:", func_min)
            
            if i < 2:
                ax1.plot(range(len(history_gd)), [h[1] for h in history_gd], label=f'Step size: {step_size}')
            else:
                ax2.plot(range(len(history_gd)), [h[1] for h in history_gd], label=f'Step size: {step_size}', linestyle='-.', color='green')

        ax1.set_yscale('log')
        ax2.set_yscale('log')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Value (solid lines)')
        ax2.set_ylabel('Best Value (dashed line)')

        plt.title('Convergence with function: ' + test_function.__class__.__name__)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')


        plt.grid()
        plt.savefig('convergence_' + test_function.__class__.__name__ + '.png')
        plt.show()
        plt.close()

# Example usage
if __name__ == "__main__":
    evaluate()
