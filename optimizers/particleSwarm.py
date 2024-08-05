import numpy as np
import matplotlib.pyplot as plt

class ParticleSwarmMinimizer:
    def __init__(self, num_particles=30, inertia=0.5, cognitive=1.0, social=1.0, max_iterations=1000, tolerance=1e-6, statistics=True):
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.statistics = statistics

    def minimize(self, func, bounds):
        """
        Minimize the given 2D function using particle swarm optimization.
        
        Parameters: 
        func : callable
            The function to minimize, which takes a 2D point (x, y) and returns a scalar.
        bounds : tuple
            The bounds for the search space, ((x_min, x_max), (y_min, y_max)).
        
        Returns:
        g_best_position : ndarray
            The position which minimizes the function.
        g_best_value : float
            The minimum value of the function.
        history : list of tuple
            The history of the global best positions and function values during the optimization.
        trajectory : list of list of ndarray
            The trajectories of all particles during the optimization.
        """
        # Initialize particle positions and velocities
        particle_positions = np.random.uniform(low=[bounds[0][0], bounds[1][0]], high=[bounds[0][1], bounds[1][1]], size=(self.num_particles, 2))
        particle_velocities = np.random.uniform(low=-1, high=1, size=(self.num_particles, 2))
        
        # Initialize personal best positions and values
        p_best_positions = np.copy(particle_positions)
        p_best_values = np.apply_along_axis(lambda pos: func(pos[0], pos[1]), 1, p_best_positions)
        
        # Initialize global best position and value
        g_best_index = np.argmin(p_best_values)
        g_best_position = p_best_positions[g_best_index]
        g_best_value = p_best_values[g_best_index]
        
        history = [(g_best_position, g_best_value)]
        trajectory = [[] for _ in range(self.num_particles)]
        
        for iteration in range(self.max_iterations):
            # Update particle velocities and positions
            for i in range(self.num_particles):
                if self.statistics:
                    trajectory[i].append([particle_positions[i].copy(), func(particle_positions[i][0], particle_positions[i][1])])
                
                r1, r2 = np.random.random(2)
                cognitive_component = self.cognitive * r1 * (p_best_positions[i] - particle_positions[i])
                social_component = self.social * r2 * (g_best_position - particle_positions[i])
                particle_velocities[i] = self.inertia * particle_velocities[i] + cognitive_component + social_component
                particle_positions[i] += particle_velocities[i]
                
                # Clamp positions to bounds
                particle_positions[i] = np.clip(particle_positions[i], [bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]])
                
                # Evaluate new position
                current_value = func(particle_positions[i][0], particle_positions[i][1])
                
                # Update personal best if necessary
                if current_value < p_best_values[i]:
                    p_best_positions[i] = particle_positions[i]
                    p_best_values[i] = current_value
                    
                    # Update global best if necessary
                    if current_value < g_best_value:
                        g_best_position = particle_positions[i]
                        g_best_value = current_value
                        history.append((g_best_position, g_best_value))
            
            # Check for convergence
            if np.linalg.norm(particle_positions - g_best_position) < self.tolerance:
                # print(f"Converged after {iteration} iterations.")
                break
        
        return g_best_position, g_best_value, history, trajectory


if __name__ == "__main__":
    def func(x, y):
        return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
    limit = 5
    bounds = ((-limit, limit), (-limit, limit))
    
    pso = ParticleSwarmMinimizer(num_particles=10, inertia=0.2, cognitive=0.5, social=1.0, max_iterations=1000, tolerance=1e-6)
    best_position, best_value, history, trajectory = pso.minimize(func, bounds)
    
    print("Best position:", best_position)
    print("Best value:", best_value)
    
    # # Plotting
    # x = np.linspace(-3, 3, 100)
    # y = np.linspace(-3, 3, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = func(X, Y)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, alpha=0.3, cmap='gray', edgecolor='none')
    
    # sort trajectory by best value
    # trajectory.sort(key=lambda x: x[1])
    # read position of best value
    # best_position = [i[0] for i in trajectory]
    # colours = plt.cm.viridis(np.linspace(0, 1, len(best_position)))

    # for i, traj in enumerate(best_position):
    #     traj_array = np.array(traj)
    #     ax.plot(traj_array[0], best_position[1], func(best_position[0], best_position[1]))

    # for i, particle in enumerate(trajectory):
    #     positions = np.array([pos for pos, val in particle])
    #     values = np.array([val for pos, val in particle])
    #     plt.plot(positions[:, 0], positions[:, 1], values, markersize=5, marker='o')
        
    # Plotting
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plt.figure()
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    for i, particle in enumerate(trajectory):
        positions = np.array([pos for pos, val in particle])
        values = np.array([val for pos, val in particle])
        # plt.gca().clear()
        plt.plot(positions[:, 0], positions[:, 1], markersize=5, marker='o')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Trajectory of Particles (Rastringin Function)')
        # plt.pause(0.1)

    plt.savefig('trajectory.png')
    plt.show()


    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('f(X, Y)')
    # ax.view_init(elev=50, azim=230)  # Adjust the view angle for better visibility
    
    # plt.show()

    # # plot best values
    # plt.scatter(range(len(history)), [h[1] for h in history])
    # plt.grid()
    # plt.xlabel('Iteration')
    # plt.ylabel('Best Value')
    # plt.title('Best Value over Iterations')
    # plt.show()

