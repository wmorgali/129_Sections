# Part a:

import numpy as np

# Define the states for N=3 spins
# Each state is represented as a binary string, e.g., |↑↑↑⟩ = 0, |↑↑↓⟩ = 1, ..., |↓↓↓⟩ = 7
states = [
    '|↑↑↑⟩', '|↑↑↓⟩', '|↑↓↑⟩', '|↑↓↓⟩',
    '|↓↑↑⟩', '|↓↑↓⟩', '|↓↓↑⟩', '|↓↓↓⟩'
]

# Initialize the transition matrix P as an 8x8 matrix of zeros
P = np.zeros((8, 8))

# Define the matrix elements of the Hamiltonian
# For simplicity, assume all non-zero transitions have equal probability
# In a real system, these would be calculated from the Hamiltonian matrix elements
# Here, we assume that each spin flip has a probability of 0.5 (for demonstration purposes)

# Transition probabilities for spin flips
# For example, |↑↑↑⟩ can transition to |↑↑↓⟩, |↑↓↑⟩, or |↓↑↑⟩ with equal probability
P[0, 1] = 0.5  # |↑↑↑⟩ -> |↑↑↓⟩
P[0, 2] = 0.5  # |↑↑↑⟩ -> |↑↓↑⟩

P[1, 0] = 0.5  # |↑↑↓⟩ -> |↑↑↑⟩
P[1, 3] = 0.5  # |↑↑↓⟩ -> |↑↓↓⟩

P[2, 0] = 0.5  # |↑↓↑⟩ -> |↑↑↑⟩
P[2, 3] = 0.5  # |↑↓↑⟩ -> |↑↓↓⟩

P[3, 1] = 0.5  # |↑↓↓⟩ -> |↑↑↓⟩
P[3, 2] = 0.5  # |↑↓↓⟩ -> |↑↓↑⟩
P[3, 7] = 0.5  # |↑↓↓⟩ -> |↓↓↓⟩

P[4, 0] = 0.5  # |↓↑↑⟩ -> |↑↑↑⟩
P[4, 5] = 0.5  # |↓↑↑⟩ -> |↓↑↓⟩

P[5, 4] = 0.5  # |↓↑↓⟩ -> |↓↑↑⟩
P[5, 7] = 0.5  # |↓↑↓⟩ -> |↓↓↓⟩

P[6, 4] = 0.5  # |↓↓↑⟩ -> |↓↑↑⟩
P[6, 7] = 0.5  # |↓↓↑⟩ -> |↓↓↓⟩

P[7, 3] = 0.5  # |↓↓↓⟩ -> |↑↓↓⟩
P[7, 5] = 0.5  # |↓↓↓⟩ -> |↓↑↓⟩
P[7, 6] = 0.5  # |↓↓↓⟩ -> |↓↓↑⟩

# Normalize the rows of P so that each row sums to 1
P = P / P.sum(axis=1, keepdims=True)

# Print the transition matrix P
print("Transition Matrix P:")
print(P)

# Print the states for reference
print("\nStates:")
for i, state in enumerate(states):
    print(f"{i}: {state}")


# Part b:

import numpy as np

# Define the states for N=3 spins
# Each state is represented as a binary string, e.g., |↑↑↑⟩ = 0, |↑↑↓⟩ = 1, ..., |↓↓↓⟩ = 7
states = [
    '|↑↑↑⟩', '|↑↑↓⟩', '|↑↓↑⟩', '|↑↓↓⟩',
    '|↓↑↑⟩', '|↓↑↓⟩', '|↓↓↑⟩', '|↓↓↓⟩'
]

# Define the transition matrix P (from Question 1)
# This is the same transition matrix we constructed earlier
P = np.array([
    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0]
])

# Solve for the stationary distribution π
# The stationary distribution satisfies πP = π and ∑π_i = 1
# This is equivalent to solving (P^T - I)π = 0 with the constraint ∑π_i = 1

# Step 1: Construct the system of equations
# We add the constraint ∑π_i = 1 as an additional equation
n = P.shape[0]  # Number of states
A = np.vstack([(P.T - np.eye(n)), np.ones(n)])  # Stack (P^T - I) and the constraint row
b = np.zeros(n + 1)  # Right-hand side of the equation
b[-1] = 1  # The last equation is ∑π_i = 1

# Step 2: Solve the system using least squares (since the system is overdetermined)
pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# Step 3: Normalize π to ensure it sums to 1 (in case of numerical errors)
pi = pi / np.sum(pi)

# Print the stationary distribution π
print("Stationary Distribution π:")
for i, prob in enumerate(pi):
    print(f"π_{i} ({states[i]}): {prob:.4f}")

# Verify that πP = π (up to numerical precision)
piP = np.dot(pi, P)
print("\nVerification (πP ≈ π):")
print(f"πP: {piP}")
print(f"π:  {pi}")
print(f"Max difference: {np.max(np.abs(piP - pi)):.6f}")


# Part c:

# Define the states for N=3 spins
# Each state is represented as a binary string, e.g., |↑↑↑⟩ = 0, |↑↑↓⟩ = 1, ..., |↓↓↓⟩ = 7
states = [
    '|↑↑↑⟩', '|↑↑↓⟩', '|↑↓↑⟩', '|↑↓↓⟩',
    '|↓↑↑⟩', '|↓↑↓⟩', '|↓↓↑⟩', '|↓↓↓⟩'
]

# Define the transition matrix P (from Question 1)
P = np.array([
    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0]
])

# Power iteration function
def power_iteration(P, initial_distribution, max_iter=1000, tol=1e-6):
    """
    Perform power iteration to find the stationary distribution π.
    
    Parameters:
        P: Transition matrix (n x n)
        initial_distribution: Initial probability distribution (n,)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        π: Stationary distribution
    """
    π = initial_distribution.copy()
    for _ in range(max_iter):
        π_new = np.dot(π, P)
        if np.linalg.norm(π_new - π) < tol:
            break
        π = π_new
    return π

# Define the three initial distributions
# 1) Pr(|↑↑↑⟩) = 1
initial_1 = np.array([1, 0, 0, 0, 0, 0, 0, 0])

# 2) Pr(|↑↑↑⟩) = 0.5, Pr(|↓↑↓⟩) = 0.5
initial_2 = np.array([0.5, 0, 0, 0, 0, 0.5, 0, 0])

# 3) Uniform distribution
initial_3 = np.ones(8) / 8

# Perform power iteration for each initial distribution
π_1 = power_iteration(P, initial_1)
π_2 = power_iteration(P, initial_2)
π_3 = power_iteration(P, initial_3)

# Print the results
print("Stationary Distribution π for Initial Guess 1 (Pr(|↑↑↑⟩) = 1):")
for i, prob in enumerate(π_1):
    print(f"π_{i} ({states[i]}): {prob:.4f}")

print("\nStationary Distribution π for Initial Guess 2 (Pr(|↑↑↑⟩) = 0.5, Pr(|↓↑↓⟩) = 0.5):")
for i, prob in enumerate(π_2):
    print(f"π_{i} ({states[i]}): {prob:.4f}")

print("\nStationary Distribution π for Initial Guess 3 (Uniform Distribution):")
for i, prob in enumerate(π_3):
    print(f"π_{i} ({states[i]}): {prob:.4f}")


# Part d:

# Define the number of spins and magnon states
N = 3  # Number of spins
k_B = 1  # Boltzmann constant (set to 1 for simplicity)
T = 1  # Temperature (arbitrary units)

# Define the magnon states and their energies
# For N=3, the allowed magnon momenta are p = 2πk/N, where k = 0, 1, 2
# The corresponding energies are E_k = 2J sin^2(p/2)
J = 1  # Coupling constant (set to 1 for simplicity)

# Allowed values of k (magnon momentum index)
k_values = np.arange(N)
p_values = 2 * np.pi * k_values / N  # Quantized momenta
E_values = 2 * J * np.sin(p_values / 2) ** 2  # Magnon energies

# Print the magnon states and their energies
print("Magnon States and Energies:")
for k, p, E in zip(k_values, p_values, E_values):
    print(f"|k={k}⟩: p = {p:.4f}, E = {E:.4f}")

# Construct the transition matrix P in the magnon basis
# The transition probabilities are Boltzmann-type: P_{kk'} ~ exp(-(E_k - E_{k'}) / (k_B T))
P = np.zeros((N, N))  # Initialize the transition matrix

for i in range(N):
    for j in range(N):
        if i != j:
            # Boltzmann factor for transition |k_i⟩ -> |k_j⟩
            P[i, j] = np.exp(-(E_values[i] - E_values[j]) / (k_B * T))
    
    # Normalize the row so that ∑ P_{ij} = 1
    P[i, :] /= np.sum(P[i, :])

# Print the transition matrix P
print("\nTransition Matrix P in Magnon Basis:")
print(P)

# Solve for the stationary distribution π in the magnon basis
# The stationary distribution satisfies πP = π and ∑π_i = 1
# We use the same method as in Question 2

# Step 1: Construct the system of equations
A = np.vstack([(P.T - np.eye(N)), np.ones(N)])  # Stack (P^T - I) and the constraint row
b = np.zeros(N + 1)  # Right-hand side of the equation
b[-1] = 1  # The last equation is ∑π_i = 1

# Step 2: Solve the system using least squares
pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# Step 3: Normalize π to ensure it sums to 1
pi = pi / np.sum(pi)

# Print the stationary distribution π
print("\nStationary Distribution π in Magnon Basis:")
for k, prob in enumerate(pi):
    print(f"π_{k} (|k={k}⟩): {prob:.4f}")

# Verify that πP = π (up to numerical precision)
piP = np.dot(pi, P)
print("\nVerification (πP ≈ π):")
print(f"πP: {piP}")
print(f"π:  {pi}")
print(f"Max difference: {np.max(np.abs(piP - pi)):.6f}")


# Part e:

# Define the number of spins and magnon states
N = 3  # Number of spins
k_B = 1  # Boltzmann constant (set to 1 for simplicity)

# Define the magnon states and their energies
# For N=3, the allowed magnon momenta are p = 2πk/N, where k = 0, 1, 2
# The corresponding energies are E_k = 2J sin^2(p/2)
J = 1  # Coupling constant (set to 1 for simplicity)

# Allowed values of k (magnon momentum index)
k_values = np.arange(N)
p_values = 2 * np.pi * k_values / N  # Quantized momenta
E_values = 2 * J * np.sin(p_values / 2) ** 2  # Magnon energies

# Function to compute the stationary distribution π at a given temperature
def stationary_distribution_magnon(T):
    """
    Compute the stationary distribution π in the magnon basis at temperature T.
    
    Parameters:
        T: Temperature
    
    Returns:
        π: Stationary distribution
    """
    # Construct the transition matrix P in the magnon basis
    P = np.zeros((N, N))  # Initialize the transition matrix

    for i in range(N):
        for j in range(N):
            if i != j:
                # Boltzmann factor for transition |k_i⟩ -> |k_j⟩
                P[i, j] = np.exp(-(E_values[i] - E_values[j]) / (k_B * T))
        
        # Normalize the row so that ∑ P_{ij} = 1
        P[i, :] /= np.sum(P[i, :])

    # Solve for the stationary distribution π
    # The stationary distribution satisfies πP = π and ∑π_i = 1
    A = np.vstack([(P.T - np.eye(N)), np.ones(N)])  # Stack (P^T - I) and the constraint row
    b = np.zeros(N + 1)  # Right-hand side of the equation
    b[-1] = 1  # The last equation is ∑π_i = 1

    # Solve the system using least squares
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Normalize π to ensure it sums to 1
    pi = pi / np.sum(pi)

    return pi

# Compute the stationary distribution π at different temperatures
temperatures = [0.1, 1, 10, 100]  # Example temperatures
print("Stationary Distribution π in Magnon Basis at Different Temperatures:")
for T in temperatures:
    π = stationary_distribution_magnon(T)
    print(f"\nTemperature T = {T}:")
    for k, prob in enumerate(π):
        print(f"π_{k} (|k={k}⟩): {prob:.4f}")

# Compare with the site basis (from Question 2)
# In the site basis, the stationary distribution is uniform: π_i = 1/8 for all i
print("\nComparison with Site Basis:")
print("In the site basis, the stationary distribution is uniform:")
for i in range(8):
    print(f"π_{i}: {1/8:.4f}")

#Part f:

# Define the number of spins and magnon states
N = 3  # Number of spins
k_B = 1  # Boltzmann constant (set to 1 for simplicity)
T = 1  # Temperature (arbitrary units)

# Define the magnon states and their energies
# For N=3, the allowed magnon momenta are p = 2πk/N, where k = 0, 1, 2
# The corresponding energies are E_k = 2J sin^2(p/2)
J = 1  # Coupling constant (set to 1 for simplicity)

# Allowed values of k (magnon momentum index)
k_values = np.arange(N)
p_values = 2 * np.pi * k_values / N  # Quantized momenta
E_values = 2 * J * np.sin(p_values / 2) ** 2  # Magnon energies

# Construct the transition matrix P in the magnon basis
# The transition probabilities are Boltzmann-type: P_{kk'} ~ exp(-(E_k - E_{k'}) / (k_B T))
P = np.zeros((N, N))  # Initialize the transition matrix

for i in range(N):
    for j in range(N):
        if i != j:
            # Boltzmann factor for transition |k_i⟩ -> |k_j⟩
            P[i, j] = np.exp(-(E_values[i] - E_values[j]) / (k_B * T))
    
    # Normalize the row so that ∑ P_{ij} = 1
    P[i, :] /= np.sum(P[i, :])

# Power iteration function
def power_iteration(P, initial_distribution, max_iter=1000, tol=1e-6):
    """
    Perform power iteration to find the stationary distribution π.
    
    Parameters:
        P: Transition matrix (n x n)
        initial_distribution: Initial probability distribution (n,)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        π: Stationary distribution
    """
    π = initial_distribution.copy()
    for _ in range(max_iter):
        π_new = np.dot(π, P)
        if np.linalg.norm(π_new - π) < tol:
            break
        π = π_new
    return π

# Define the three initial distributions in the magnon basis
# 1) Pr(|k=1⟩) = 1
initial_1 = np.array([0, 1, 0])  # All probability in |k=1⟩

# 2) Pr(|k=1⟩) = 0.5, Pr(|k=4⟩) = 0.5
# Note: For N=3, |k=4⟩ is not a valid state. We assume |k=2⟩ instead.
initial_2 = np.array([0, 0.5, 0.5])  # Equal probability in |k=1⟩ and |k=2⟩

# 3) Uniform distribution
initial_3 = np.ones(N) / N  # Equal probability for all magnon states

# Perform power iteration for each initial distribution
π_1 = power_iteration(P, initial_1)
π_2 = power_iteration(P, initial_2)
π_3 = power_iteration(P, initial_3)

# Print the results
print("Stationary Distribution π for Initial Guess 1 (Pr(|k=1⟩) = 1):")
for k, prob in enumerate(π_1):
    print(f"π_{k} (|k={k}⟩): {prob:.4f}")

print("\nStationary Distribution π for Initial Guess 2 (Pr(|k=1⟩) = 0.5, Pr(|k=2⟩) = 0.5):")
for k, prob in enumerate(π_2):
    print(f"π_{k} (|k={k}⟩): {prob:.4f}")

print("\nStationary Distribution π for Initial Guess 3 (Uniform Distribution):")
for k, prob in enumerate(π_3):
    print(f"π_{k} (|k={k}⟩): {prob:.4f}")

#Part g:

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the number of spins and magnon states
N = 3  # Number of spins
k_B = 1  # Boltzmann constant (set to 1 for simplicity)
T = 1  # Temperature (arbitrary units)

# Define the magnon states and their energies
# For N=3, the allowed magnon momenta are p = 2πk/N, where k = 0, 1, 2
# The corresponding energies are E_k = 2J sin^2(p/2)
J = 1  # Coupling constant (set to 1 for simplicity)

# Allowed values of k (magnon momentum index)
k_values = np.arange(N)
p_values = 2 * np.pi * k_values / N  # Quantized momenta
E_values = 2 * J * np.sin(p_values / 2) ** 2  # Magnon energies

# Construct the transition matrix P in the magnon basis
# The transition probabilities are Boltzmann-type: P_{kk'} ~ exp(-(E_k - E_{k'}) / (k_B T))
P = np.zeros((N, N))  # Initialize the transition matrix

for i in range(N):
    for j in range(N):
        if i != j:
            # Boltzmann factor for transition |k_i⟩ -> |k_j⟩
            P[i, j] = np.exp(-(E_values[i] - E_values[j]) / (k_B * T))
    
    # Normalize the row so that ∑ P_{ij} = 1
    P[i, :] /= np.sum(P[i, :])

# Convert the transition matrix P into the transition rate matrix Q
# The relationship between P and Q is: Q = (P - I) / Δt
# For small Δt, we can approximate P ≈ I + QΔt
# Here, we assume Δt = 1 for simplicity
Q = P - np.eye(N)

# Define the classical master equation: dπ/dt = πQ
def master_equation(t, π, Q):
    """
    Classical master equation: dπ/dt = πQ.
    
    Parameters:
        t: Time (not used explicitly, required by solve_ivp)
        π: Probability distribution at time t
        Q: Transition rate matrix
    
    Returns:
        dπ/dt: Time derivative of the probability distribution
    """
    return np.dot(π, Q)

# Initial condition: Pr(|k=1⟩) = 1
π0 = np.array([0, 1, 0])  # All probability in |k=1⟩

# Time span for integration
t_span = (0, 10)  # From t=0 to t=10
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time points to evaluate the solution

# Solve the master equation using scipy's solve_ivp
sol = solve_ivp(master_equation, t_span, π0, args=(Q,), t_eval=t_eval, method='RK45')

# Extract the solution
π_t = sol.y  # Probability distribution over time
t = sol.t  # Time points

# Plot the evolution of the probability distribution
plt.figure(figsize=(10, 6))
for k in range(N):
    plt.plot(t, π_t[k], label=f"π_{k} (|k={k}⟩)")

plt.xlabel("Time (t)")
plt.ylabel("Probability π(t)")
plt.title("Evolution of Probability Distribution in Magnon Basis")
plt.legend()
plt.grid()
plt.savefig("evolution_of_probability_distribution.png")