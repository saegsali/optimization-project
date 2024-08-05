import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DownhillSimplexMinimizer:
    def __init__(self, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iterations=1000, tolerance=1e-6, statistics=True):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.statistics = statistics

    def minimize(self, func, initial_point):
        def initialize_simplex(point):
            """Initialize simplex around the initial point with random perturbations."""
            simplex = [point]
            for i in range(len(point)): # In our case, len(point) == 2 -> create a triangle
                vertex = point.copy()
                vertex[i] += np.random.uniform(-10, 10)
                simplex.append(vertex)
            return np.array(simplex) 

        def order_simplex(simplex):
            """Sort simplex vertices based on their function values."""
            return sorted(simplex, key=func)

        def centroid(simplex):
            """Calculate the centroid of the best n vertices (excluding the worst)."""
            return np.mean(simplex[:-1], axis=0)

        def reflect(centroid, worst):
            """Reflect the worst point through the centroid."""
            return centroid + self.alpha * (centroid - worst)

        def expand(centroid, reflected):
            """Expand beyond the reflected point."""
            return centroid + self.gamma * (reflected - centroid)

        def contract_outside(centroid, reflected):
            """Contract towards the reflected point."""
            return centroid + self.rho * (reflected - centroid)

        def contract_inside(centroid, worst):
            """Contract towards the centroid from the worst point."""
            return centroid + self.rho * (worst - centroid)

        def shrink(simplex):
            """Shrink the simplex towards the best point."""
            return [simplex[0]] + [simplex[0] + self.sigma * (v - simplex[0]) for v in simplex[1:]]


        history = []
        simplex = initialize_simplex(initial_point)
        simplex = order_simplex(simplex)
        if self.statistics:
            history.append((simplex.copy(), func(simplex[0])))

        for iteration in range(self.max_iterations):
            simplex = order_simplex(simplex)
            best, worst = simplex[0], simplex[-1]
            second_worst = simplex[-2]
            centroid_point = centroid(simplex)
            # Reflect the worst point
            reflected = reflect(centroid_point, worst)
            f_reflected = func(reflected)

            # Update the simplex based on the reflection
            if f_reflected < func(best) and f_reflected < func(second_worst):
                expanded = expand(centroid_point, reflected)
                # If the expanded point is better than the reflected point, replace the worst point with the expanded point
                if func(expanded) < f_reflected:
                    simplex[-1] = expanded
                else:
                    simplex[-1] = reflected
            elif func(best) <= f_reflected < func(second_worst):
                simplex[-1] = reflected
            # If the reflected point is worse than the second worst point, try to contract
            else:
                if f_reflected < func(worst):
                    contracted = contract_outside(centroid_point, reflected)
                    if func(contracted) < f_reflected:
                        simplex[-1] = contracted
                    else:
                        simplex = shrink(simplex)
                else:
                    contracted = contract_inside(centroid_point, worst)
                    if func(contracted) < func(worst):
                        simplex[-1] = contracted
                    else:
                        simplex = shrink(simplex)

            best_value = func(simplex[0])
            if self.statistics:
                history.append((simplex.copy(), best_value))

            # Terminate if the standard deviation of the function values is below the tolerance
            if np.std([func(x) for x in simplex]) < self.tolerance:
                break

        x_min = simplex[0]
        func_min = func(x_min)
        return x_min, func_min, history
    
    def multiple_minimize(self, func, initial_point, num_restarts=10):
        best_x = None
        best_func_value = float('inf')
        best_history = None

        for _ in range(num_restarts):
            x_min, func_min, history = self.minimize(func, initial_point)
            if func_min < best_func_value:
                best_x = x_min
                best_func_value = func_min
                best_history = history

            # Update initial_point for next run to a new random point
            initial_point = initial_point + np.random.uniform(-10, 10, size=initial_point.shape)

        return best_x, best_func_value, best_history


if __name__ == "__main__":
    def func(x, y):
        # use reastrigin function
        return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

    initial_point = np.array([5, -8])

    ds_minimizer = DownhillSimplexMinimizer(max_iterations=1000, tolerance=1e-6)
    best_position, best_value, history = ds_minimizer.minimize(lambda p: func(p[0], p[1]), initial_point)

    print("Best position:", best_position)
    print("Best value:", best_value)

    # Plotting
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')

    # Highlight the trajectory of the simplex points
    points = np.array([h[0][0] for h in history])  # Extract the best point at each iteration
    ax.plot(points[:, 0], points[:, 1], func(points[:, 0], points[:, 1]), 'r.-', markersize=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.view_init(elev=50, azim=230)  # Adjust the view angle for better visibility

    plt.figure()
    plt.contour(X, Y, Z, levels=np.logspace(-1, 5, 35), cmap='viridis')
    for i, (simplex, _) in enumerate(history):
        plt.gca().clear()
        plt.contour(X, Y, Z, levels=np.logspace(-1, 5, 35), cmap='viridis')
        for triangle in history[:i+1]:
            simplex_vertices = triangle[0]
            polygon = Polygon(simplex_vertices[:3], edgecolor='r', fill=False)
            plt.gca().add_patch(polygon)
        plt.plot(points[:i+1, 0], points[:i+1, 1], 'ro-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Downhill Simplex Trajectory - Iteration {i}')
        plt.pause(0.1)

    plt.show()
