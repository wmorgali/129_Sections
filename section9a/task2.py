import numpy as np
import matplotlib.pyplot as plt

# Constants
e = 0.6  # Eccentricity
Tf = 200  # Final time
N1 = 100000  # Steps for explicit Euler
N2 = 400000  # Steps for symplectic Euler

# Time steps
dt1 = Tf / N1
dt2 = Tf / N2

# Initial conditions
q1_0, q2_0 = 1 - e, 0
p1_0, p2_0 = 0, np.sqrt((1 + e) / (1 - e))

def acceleration(q1, q2):
    r = (q1**2 + q2**2)**(3/2)
    return -q1 / r, -q2 / r

def explicit_euler(N, dt):
    q1, q2 = np.zeros(N), np.zeros(N)
    p1, p2 = np.zeros(N), np.zeros(N)
    
    q1[0], q2[0] = q1_0, q2_0
    p1[0], p2[0] = p1_0, p2_0
    
    for n in range(N - 1):
        q1[n + 1] = q1[n] + dt * p1[n]
        q2[n + 1] = q2[n] + dt * p2[n]
        
        a1, a2 = acceleration(q1[n], q2[n])
        p1[n + 1] = p1[n] + dt * a1
        p2[n + 1] = p2[n] + dt * a2
    
    return q1, q2

def symplectic_euler(N, dt):
    q1, q2 = np.zeros(N), np.zeros(N)
    p1, p2 = np.zeros(N), np.zeros(N)
    
    q1[0], q2[0] = q1_0, q2_0
    p1[0], p2[0] = p1_0, p2_0
    
    for n in range(N - 1):
        a1, a2 = acceleration(q1[n], q2[n])
        p1[n + 1] = p1[n] + dt * a1
        p2[n + 1] = p2[n] + dt * a2
        
        q1[n + 1] = q1[n] + dt * p1[n + 1]
        q2[n + 1] = q2[n] + dt * p2[n + 1]
    
    return q1, q2

# Compute orbits
q1_euler, q2_euler = explicit_euler(N1, dt1)
q1_symplectic, q2_symplectic = symplectic_euler(N2, dt2)

# Plot results
plt.figure(figsize=(8, 8))
plt.plot(q1_euler, q2_euler, label="Explicit Euler", alpha=0.6)
plt.plot(q1_symplectic, q2_symplectic, label="Symplectic Euler", alpha=0.6)
plt.xlabel("q1")
plt.ylabel("q2")
plt.legend()
plt.title("Planetary Orbit Simulation")
plt.grid()
plt.savefig("planetary_orbit.png")