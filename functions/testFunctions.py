import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Define function parameters
x_min, x_max = -4, 4
y_min, y_max = -4, 4

elev = 30
azim = 230

alpha = 0.8

class quadraticFunction():
    # Define a sample 3D function and its gradient
    def func(self, x, y):
        return x**2 + y**2
    
    def grad_func(self, x, y):
        return 2*x, 2*y
    
    def plot(self):
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=0.8, cmap='viridis')

        # ax.set_title('Quadratic Function')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('f(X, Y)')
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        # save as svg with transparent background
        plt.savefig('quadratic.svg', format='svg', dpi=600, transparent=True)
        # save as png with transparent background
        plt.savefig('quadratic.png', format='png', dpi=200, transparent=True)
        plt.show()
    
class rosenbrockFunction():
    # Define a sample 3D function and its gradient
    def func(self, x, y):
        try:
            return 100*(y - x**2)**2 + (1 - x)**2
        except:
            print("Error in rosenbrockFunction")
            return 1000000
    
    def grad_func(self, x, y):
        grad_x = -400*x*(y - x**2) + 2*x - 2
        grad_y = 200*(y - x**2)
        
        return (grad_x, grad_y)
    
    def plot(self):
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-0.75, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=alpha, cmap='viridis')

        # ax.set_title('Rosenbrock Function')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('f(X, Y)')
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        # save as svg with transparent background
        plt.savefig('rosenbrock.svg', format='svg', dpi=600, transparent=True)
        # save as png with transparent background
        plt.savefig('rosenbrock.png', format='png', dpi=200, transparent=True)
        plt.show()
    
class rastriginFunction():
    # Define a sample 3D function and its gradient
    def func(self, x, y):
        return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
    
    def grad_func(self, x, y):
        grad_x = 2*x + 20*np.pi*np.sin(2*np.pi*x)
        grad_y = 2*y + 20*np.pi*np.sin(2*np.pi*y)
        return (grad_x, grad_y)
    
    def plot(self):
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=alpha, cmap='viridis')

        # ax.set_title('Rastrigin Function')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('f(X, Y)')
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        # save as svg with transparent background
        plt.savefig('rastrigin.svg', format='svg', dpi=600, transparent=True)
        # save as png with transparent background
        plt.savefig('rastrigin.png', format='png', dpi=200, transparent=True)
        plt.show()

    

if __name__ == "__main__":

    # Plot the rastrigin function
    f = quadraticFunction()
    f.plot()

    f = rosenbrockFunction()
    f.plot()

    f = rastriginFunction()
    f.plot()

    # f = alpineFunction()
    # f.plot()
