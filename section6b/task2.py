import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 0.5     # Damping coefficient
D = 1.0        # Noise strength
T = 10.0       # Total time
N = 1000       # Number of time steps
v0 = 1.0       # Initial velocity

# Time step size
dt = T / N

# Generate white noise
dW = np.random.randn(N) * np.sqrt(dt)

# Langevin equation: dv/dt = -gamma * v + sqrt(2D) * eta(t)
v = np.zeros(N)
v[0] = v0

for i in range(1, N):
    v[i] = v[i-1] - gamma * v[i-1] * dt + np.sqrt(2 * D) * dW[i-1]

# Time array
t = np.linspace(0, T, N)

# Calculate mean and variance
v_mean = np.mean(v)
v_variance = np.var(v)

# Plot the velocity trajectory
plt.figure(figsize=(10, 5))
plt.plot(t, v, label="Langevin Velocity")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Langevin Equation Simulation")
plt.grid(True)
plt.legend()
plt.savefig("task2.png")

# Print mean and variance
print(f"Mean of velocity: {v_mean}")
print(f"Variance of velocity: {v_variance}")
