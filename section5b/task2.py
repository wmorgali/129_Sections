import numpy as np
import matplotlib.pyplot as plt

#Part C:

# Constants
N = 100  # Total number of bosons
epsilon = 1.0  # Energy gap (normalized units)
kB = 1.0  # Boltzmann constant (normalized units)
T_vals = np.linspace(0.1, 20, 1000)  # Temperature range

# Compute the average occupation numbers
n_excited = []
n_ground = []

for T in T_vals:
    beta = 1 / (kB * T)
    exp_term = np.exp(-beta * epsilon)
    
    Z_C = (1 - exp_term**(N+1)) / (1 - exp_term)
    avg_n_excited = (exp_term / (1 - exp_term)) - ((N+1) * exp_term**(N+1) / (1 - exp_term**(N+1)))
    avg_n_ground = N - avg_n_excited
    
    n_excited.append(avg_n_excited)
    n_ground.append(avg_n_ground)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(T_vals, n_ground, label=r'$\langle n_0 \rangle_C$ (Ground state)', color='blue')
plt.plot(T_vals, n_excited, label=r'$\langle n_{\epsilon} \rangle_C$ (Excited state)', color='red')
plt.xlabel('Temperature (K)')
plt.ylabel('Average Particle Number (Percent of Total)')
plt.title('Average Occupation Numbers in the Canonical Ensemble ')
plt.legend()
plt.grid()
plt.savefig("task_C_figure.png", dpi=300)

#Part E:

n_excited_Q = []
n_ground_Q = []

for T in T_vals:
    beta = 1 / (kB * T)
    exp_term = np.exp(-beta * epsilon)

    # Quantum Case
    avg_n_excited_Q = 1 / (np.exp(beta * epsilon) - 1)
    avg_n_ground_Q = N - avg_n_excited_Q

    n_excited_Q.append(avg_n_excited_Q)
    n_ground_Q.append(avg_n_ground_Q)

# Fix: Ensure lists are passed to plt.plot
plt.figure(figsize=(8, 6))
plt.plot(T_vals, n_ground_Q, label=r'$\langle n_0 \rangle_Q$ (Ground state)', color='blue')
plt.plot(T_vals, n_excited_Q, label=r'$\langle n_{\epsilon} \rangle_Q$ (Excited state)', color='red')
plt.xlabel('Temperature (K)')
plt.ylabel('Average Particle Number (Percent of Total)')
plt.title('Average Occupation Numbers in the Canonical Ensemble (Quantum)')
plt.legend()
plt.grid()
plt.savefig("task_E_figure.png", dpi=300)

# Part H: Grand Canonical Ensemble Particle Number
N_large = 10**5  # Large particle number
mu_vals = np.linspace(-epsilon, -0.01, 1000)  # Fix: Avoid exactly zero to prevent division by zero

n_ground_G = []
for mu in mu_vals:
    avg_n_ground_G = 1 / (np.exp(-mu) - 1)
    n_ground_G.append(avg_n_ground_G)

# Plot Grand Canonical Ensemble Occupation
plt.figure(figsize=(8, 6))
plt.plot(mu_vals, n_ground_G, label=r'$\langle n_0 \rangle_G$ (Grand Canonical)', color='green')
plt.xlabel('Chemical Potential $\mu$')
plt.ylabel('Average Particle Number')
plt.title('Ground State Occupation in the Grand Canonical Ensemble')
plt.legend()
plt.grid()
plt.savefig("task_H_figure.png", dpi=300)

# Part I: Near-Degenerate Bose System (BEC)
degeneracies = np.linspace(1, 100, 1000)  # Simulated degeneracies
n_ground_I = [1 / (np.exp(-g) - 1) for g in degeneracies]

# Fix: Filter out negative/zero values before applying log
valid_indices = [i for i, val in enumerate(n_ground_I) if val > 0]
degeneracies_valid = [degeneracies[i] for i in valid_indices]
n_ground_I_valid = [n_ground_I[i] for i in valid_indices]


# Part J: Near-Degenerate Bose System Without BEC
mu_vals_no_bec = np.linspace(-epsilon, -0.01, 1000)  # Fix: Avoid zero for division safety
n_ground_J = [1 / (np.exp(-mu) + 1) for mu in mu_vals_no_bec]  # Modified degeneracy function

plt.figure(figsize=(8, 6))
plt.plot(mu_vals_no_bec, n_ground_J, label='Ground State Occupation (No BEC)', color='orange')
plt.xlabel('Chemical Potential $\mu$')
plt.ylabel('⟨n0⟩')
plt.title('Near-Degenerate System Without BEC')
plt.legend()
plt.grid()
plt.savefig("task_J_figure.png", dpi=300)

# Part K: Near-Degenerate Bose System With BEC
T_c = 1.5  # Critical temperature
T_vals_k = np.linspace(0.1, 5, 1000)
n_ground_K = [N_large * (1 - (T / T_c)**1.5) if T < T_c else 0 for T in T_vals_k]  # BEC formula

plt.figure(figsize=(8, 6))
plt.plot(T_vals_k, n_ground_K, label='Ground State Occupation (BEC)', color='brown')
plt.axvline(T_c, color='black', linestyle='dashed', label='Critical Temperature $T_c$')
plt.xlabel('Temperature (T)')
plt.ylabel('⟨n0⟩')
plt.title('Near-Degenerate System With BEC')
plt.legend()
plt.grid()
plt.savefig("task_K_figure.png", dpi=300)
