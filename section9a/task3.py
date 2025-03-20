import numpy as np
import matplotlib.pyplot as plt

# Task 3A: Gradient Descent
def H(theta):
    return theta**4 - 8*theta**2 - 2*np.cos(4*np.pi*theta)

def gradient(theta):
    return 4*theta**3 - 16*theta + 8*np.pi*np.sin(4*np.pi*theta)

def gradient_descent(theta0, alpha=0.01, tol=1e-6, max_iter=1000):
    theta = theta0
    history = [theta]
    for _ in range(max_iter):
        theta -= alpha * gradient(theta)
        history.append(theta)
        if abs(gradient(theta)) < tol:
            break
    return np.array(history)

def plot_gradient_descent():
    initial_guesses = [-1, 0.5, 3]
    plt.figure(figsize=(8, 6))
    x = np.linspace(-2, 3, 100)
    plt.plot(x, H(x), label="H(theta)")
    
    for theta0 in initial_guesses:
        history = gradient_descent(theta0)
        plt.plot(history, H(history), 'ro-', markersize=3, label=f"Start: {theta0}")
    
    plt.legend()
    plt.xlabel("Theta")
    plt.ylabel("H(Theta)")
    plt.title("Gradient Descent Optimization")
    plt.grid()
    plt.savefig("task3a.png")

plot_gradient_descent()

# Task 3B: Metropolis-Hastings Algorithm
def metropolis_hastings(theta0, beta=1, sigma=0.5, steps=10000):
    theta = theta0
    samples = [theta]
    for _ in range(steps):
        theta_star = theta + np.random.normal(0, sigma)
        r = np.exp(-beta * (H(theta_star) - H(theta)))
        if r > np.random.rand():
            theta = theta_star
        samples.append(theta)
    return np.array(samples)

def plot_metropolis():
    initial_guesses = [-1, 0.5, 3]
    plt.figure(figsize=(8, 6))
    x = np.linspace(-2, 3, 100)
    plt.plot(x, H(x), label="H(theta)")
    
    for theta0 in initial_guesses:
        samples = metropolis_hastings(theta0)
        plt.plot(samples, H(samples), 'bo', markersize=1, label=f"Start: {theta0}")
    
    plt.legend()
    plt.xlabel("Theta")
    plt.ylabel("H(Theta")
    plt.title("Metropolis-Hastings Sampling")
    plt.grid()
    plt.savefig("task3b.png")

plot_metropolis()

# Task 3C: Simulated Annealing
def simulated_annealing(theta0, beta0=1, delta_beta=0.01, sigma=0.5, steps=10000):
    theta = theta0
    beta = beta0
    samples = [theta]
    for _ in range(steps):
        theta_star = theta + np.random.normal(0, sigma)
        r = np.exp(-beta * (H(theta_star) - H(theta)))
        if r > np.random.rand():
            theta = theta_star
        samples.append(theta)
        beta += delta_beta
    return np.array(samples)

def plot_simulated_annealing():
    initial_guesses = [-1, 0.5, 3]
    plt.figure(figsize=(8, 6))
    x = np.linspace(-2, 3, 100)
    plt.plot(x, H(x), label="H(theta)")
    
    for theta0 in initial_guesses:
        samples = simulated_annealing(theta0)
        plt.plot(samples, H(samples), 'go', markersize=1, label=f"Start: {theta0}")
    
    plt.legend()
    plt.xlabel("Theta")
    plt.ylabel("H(Theta")
    plt.title("Simulated Annealing")
    plt.grid()
    plt.savefig("task3c.png")

plot_simulated_annealing()
