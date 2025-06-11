import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from helper_functions import optimize

def cost_function(p):
    # simple 2D quadratic cost function with minimum at (3, -1).
    x, y = p
    return (x - 3)**2 + (y + 1)**2

def main():
    # define bounds
    mi = np.array([-5, -5])
    ma = np.array([5, 5])

    # setup contour plot
    fig, ax = plt.subplots()
    x = np.linspace(mi[0], ma[0], 200)
    y = np.linspace(mi[1], ma[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = cost_function([X, Y])
    ax.contourf(X, Y, Z, levels=50, cmap='viridis')

    scatter = ax.scatter([], [], c='white', s=10, label='Population')
    best_marker, = ax.plot([], [], 'ro', label='Best')
    ax.legend()
    ax.set_title("Optimization Progress")

    plt.ion()
    plt.show()

    def draw(pop, best_idx):
        if pop.shape[0] < 2:
            print("Population has fewer than 2 dimensions — nothing to plot.")
            return
        scatter.set_offsets(pop[:2].T)  # only first 2 dims
        best_marker.set_data([pop[0, best_idx]], [pop[1, best_idx]])
        ax.set_title(f"Best Cost: {cost_function(pop[:, best_idx]):.4f}")
        fig.canvas.draw_idle()
        plt.pause(0.1)

    # run the optimizer
    best = optimize(cost_function, mi, ma, draw_function=draw, n=100, max_iter=1000, random_state=42, min_iterations=1000)

    print("Best parameters found:", best)
    print("Minimum cost:", cost_function(best))

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()