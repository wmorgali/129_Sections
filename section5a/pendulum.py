import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.spatial import ConvexHull
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Part c.

# Constants
m1, m2 = 1.0, 1.0   # Masses
L1, L2 = 1.0, 1.0   # Lengths
g = 9.81            # Gravity

# Equations of motion (Hamiltonian Formulation)
def equations(t, state):
    theta1, theta2, p1, p2 = state
    
    # Compute velocities from momenta
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    
    M = np.array([
        [(m1 + m2) * L1**2, m2 * L1 * L2 * c],
        [m2 * L1 * L2 * c, m2 * L2**2]
    ])
    inv_M = np.linalg.inv(M)
    
    q_dot = inv_M @ np.array([p1, p2])
    theta1_dot, theta2_dot = q_dot
    
    # Compute torques
    f1 = -m2 * L1 * L2 * s * theta2_dot**2 - (m1 + m2) * g * L1 * np.sin(theta1)
    f2 = m2 * L1 * L2 * s * theta1_dot**2 - m2 * g * L2 * np.sin(theta2)
    
    p_dot = inv_M @ np.array([f1, f2])
    p1_dot, p2_dot = p_dot
    
    return [theta1_dot, theta2_dot, p1_dot, p2_dot]

# Initial conditions
state0 = [np.pi / 4, np.pi / 2, 0.0, 0.0]  # (theta1, theta2, p1, p2)
num_points = 50  # Number of initial conditions for phase space density
perturbation = 0.01  # Small perturbation to initial conditions
initial_conditions = []
for i in range(num_points):
    theta1 = np.pi / 4 + np.random.uniform(-perturbation, perturbation)
    theta2 = np.pi / 2 + np.random.uniform(-perturbation, perturbation)
    p1 = np.random.uniform(-perturbation, perturbation)
    p2 = np.random.uniform(-perturbation, perturbation)
    initial_conditions.append([theta1, theta2, p1, p2])

t_span = (0, 10)  # Simulate for 10 seconds
t_eval = np.linspace(*t_span, 1000)

t_eval2 = np.linspace(*t_span, 500)

# Solve for each initial condition
trajectories = []
for state0 in initial_conditions:
    sol = solve_ivp(equations, t_span, state0, t_eval=t_eval2, method='RK45')
    trajectories.append(sol.y)

# Extract theta2 and p2 for all trajectories
phase_space_data = np.array([traj[[1, 3], :].T for traj in trajectories])

# Compute convex hull volume over time
volumes = []
for i in range(len(t_eval2)):  # Use t_eval2 to match phase_space_data size
    points = np.array([phase_space_data[j, i, :] for j in range(num_points)])
    hull = ConvexHull(points)
    volumes.append(hull.volume)


# Solve the system
sol = solve_ivp(equations, t_span, state0, t_eval=t_eval, method='RK45')

# Extract results
theta1_vals, theta2_vals, p1_vals, p2_vals = sol.y

# Convert to Cartesian coordinates
x1 = L1 * np.sin(theta1_vals)
y1 = -L1 * np.cos(theta1_vals)
x2 = x1 + L2 * np.sin(theta2_vals)
y2 = y1 - L2 * np.cos(theta2_vals)

# Plot Phase Space (theta2 vs p2)
plt.figure(figsize=(8, 4))
plt.plot(theta2_vals, p2_vals, 'b')
plt.xlabel("Theta2")
plt.ylabel("Momentum p2")
plt.title("Phase Space Trajectory (Theta2 vs p2)")
plt.grid()
plt.savefig("double_pendulum_phase_space.png")

# Animate the double pendulum motion
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
line, = ax.plot([], [], 'o-', lw=2)

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)
ani.save("double_pendulum.gif", writer=PillowWriter(fps=30))


# Part d.
plt.figure(figsize=(8, 4))
plt.plot(t_eval2, volumes, 'r')
plt.xlabel("Time")
plt.ylabel("Phase Space Volume")
plt.title("Evolution of Phase Space Volume in (Theta2, p2)")
plt.grid()
plt.savefig("double_pendulum_phase_space_volume.png")