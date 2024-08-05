import numpy as np
import matplotlib.pyplot as plt

class GradientDescentMinimizer:
    def __init__(self, step_size=0.01, max_iterations=1000, tolerance=1e-6, statistics=True):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.statistics = statistics
    
    def minimize(self, func, grad_func, initial_point):
        """
        Minimize the given 2D function using gradient descent.
        
        Parameters:
        func : callable
            The function to minimize, which takes a 2D point (x, y) and returns a scalar.
        grad_func : callable
            The gradient of the function, which takes a 2D point (x, y) and returns a tuple (df/dx, df/dy).
        initial_point : ndarray
            The starting point for the gradient descent.
        
        Returns:
        x_min : ndarray
            The point which minimizes the function.
        func_min : float
            The minimum value of the function.
        history : list of tuple
            The history of points and function values during the optimization.
        """
        x = initial_point
        history = [(x, func(*x))]
        
        for i in range(self.max_iterations):
            grad = grad_func(*x)
            x_new = x - self.step_size * np.array(grad)
            f_new = func(*x_new)
            
            if self.statistics:
                history.append((x_new, f_new))
            
            if np.linalg.norm(x_new - x) < self.tolerance:
                break
            
            x = x_new
        
        x_min = x
        func_min = func(*x_min)
        return x_min, func_min, history


if __name__ == "__main__":
    def func(x, y):
        return x**2 + y**2

    def grad_func(x, y):
        return 2*x, 2*y

    initial_point = np.array([5, -6])

    minimizer = GradientDescentMinimizer(step_size=0.01, max_iterations=200, tolerance=1e-3)
    x_min, func_min, history = minimizer.minimize(func, grad_func, initial_point)

    points = [entry[0] for entry in history]
    points = np.array(points)

    # Plot the function
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')

    # Highlight the trajectory
    ax.plot(points[:, 0], points[:, 1], func(points[:, 0], points[:, 1]), 'r.-', markersize=5, label='Optimization Path')
    ax.scatter(points[:, 0], points[:, 1], func(points[:, 0], points[:, 1]), color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.view_init(elev=50, azim=230)  # Adjust the view angle for better visibility
    ax.legend()
    plt.show()