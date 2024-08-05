import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

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
        history.append((simplex.copy(), func(simplex[0])))

        centroids = [centroid(simplex)]

        for iteration in range(self.max_iterations):
            simplex = order_simplex(simplex)
            best, worst = simplex[0], simplex[-1]
            second_worst = simplex[-2]
            centroid_point = centroid(simplex)
            reflected = reflect(centroid_point, worst)
            f_reflected = func(reflected)

            if f_reflected < func(best) and f_reflected < func(second_worst):
                expanded = expand(centroid_point, reflected)
                if func(expanded) < f_reflected:
                    simplex[-1] = expanded
                else:
                    simplex[-1] = reflected
            elif func(best) <= f_reflected < func(second_worst):
                simplex[-1] = reflected
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
            history.append((simplex.copy(), best_value))
            centroids.append(centroid(simplex))

            if np.std([func(x) for x in simplex]) < self.tolerance:
                break

        x_min = simplex[0]
        func_min = func(x_min)
        return x_min, func_min, history, centroids

# Example usage
if __name__ == "__main__":
    def func(x, y):
        # Use Rastrigin function
        return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

    initial_point = np.array([3, -2])

    ds_minimizer = DownhillSimplexMinimizer(max_iterations=1000, tolerance=1e-6)
    best_position, best_value, history, centroids = ds_minimizer.minimize(lambda p: func(p[0], p[1]), initial_point)

    print("Best position:", best_position)
    print("Best value:", best_value)

    # Plotting
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')

    # Convert initial simplex to a numpy array
    initial_simplex = np.array(history[0][0])
    ax.plot_trisurf(initial_simplex[:, 0], initial_simplex[:, 1], func(initial_simplex[:, 0], initial_simplex[:, 1]), color='red', alpha=0.5)

    # Plot the path of the centroids
    centroids = np.array(centroids)
    ax.plot(centroids[:, 0], centroids[:, 1], func(centroids[:, 0], centroids[:, 1]), 'ro-', markersize=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.view_init(elev=50, azim=230)  # Adjust the view angle for better visibility

    plt.figure()
    plt.contour(X, Y, Z, levels=np.logspace(0, 8, 35), cmap='viridis')

    # Plot the initial simplex as a closed polygon
    closed_initial_simplex = np.vstack([initial_simplex, initial_simplex[0]])
    plt.plot(closed_initial_simplex[:, 0], closed_initial_simplex[:, 1], 'ro-', markersize=5, label='Initial Simplex')
    plt.plot(centroids[:, 0], centroids[:, 1], 'bo-', markersize=5, label='Centroid Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Downhill Simplex - Initial Simplex and Centroid Path')

    plt.show()