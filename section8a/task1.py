import numpy as np
import itertools
import matplotlib.pyplot as plt

def generate_lattice(L):
    """Generate a random LxL spin lattice with values -1 or 1"""
    return np.random.choice([-1, 1], size=(L, L))

def ising_hamiltonian(lattice, J=1, B=0):
    """Compute the Ising Hamiltonian for a given lattice configuration"""
    L = lattice.shape[0]
    energy = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = [lattice[(i+1) % L, j], lattice[i, (j+1) % L]]  # Periodic Boundary
            energy += -J * S * sum(neighbors) - B * S
    return energy

def compute_partition_function(L, J=1, B=0, T=1):
    """Compute the partition function by summing over all possible states"""
    beta = 1 / T
    states = list(itertools.product([-1, 1], repeat=L*L))  # All possible states
    Z = sum(np.exp(-beta * ising_hamiltonian(np.array(state).reshape(L, L), J, B)) for state in states)
    return Z

def sample_distribution(L, J=1, B=0, T=1):
    """Generate samples from the exact distribution"""
    beta = 1 / T
    states = list(itertools.product([-1, 1], repeat=L*L))
    energies = [ising_hamiltonian(np.array(state).reshape(L, L), J, B) for state in states]
    weights = np.exp(-beta * np.array(energies))
    weights /= np.sum(weights)  # Normalize to get probability distribution
    sampled_index = np.random.choice(len(states), p=weights)
    return np.array(states[sampled_index]).reshape(L, L), weights, states, weights

def compute_expectation_values(L, J=1, B=0, T=1):
    """Compute expectation values for energy and magnetization"""
    beta = 1 / T
    states = list(itertools.product([-1, 1], repeat=L*L))
    energies = np.array([ising_hamiltonian(np.array(state).reshape(L, L), J, B) for state in states])
    magnetizations = np.array([np.sum(state) for state in states])
    weights = np.exp(-beta * energies)
    Z = np.sum(weights)
    weights /= Z  # Normalize
    
    expected_energy = np.sum(weights * energies)
    expected_magnetization = np.sum(weights * magnetizations)
    
    return expected_energy, expected_magnetization, states, weights

def gibbs_sampler(lattice, J=1, B=0, T=1, steps=1000):
    """Perform Gibbs sampling to update the lattice"""
    beta = 1 / T
    L = lattice.shape[0]
    
    for _ in range(steps):
        i, j = np.random.randint(0, L, size=2)  # Pick a random spin
        S = lattice[i, j]
        
        # Compute energy change if this spin flips
        neighbors = [lattice[(i+1) % L, j], lattice[i, (j+1) % L], lattice[(i-1) % L, j], lattice[i, (j-1) % L]]
        dE = 2 * J * S * sum(neighbors) + 2 * B * S
        
        # Flip spin with Metropolis acceptance probability
        if np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1
    
    return lattice

def compute_magnetization(lattice):
    """Compute magnetization of a given lattice configuration"""
    return np.sum(lattice) / lattice.size


def gibbs_iteration(L, J=1, B=0, T=1, steps=10000, burn_in=1000):
    """Perform Gibbs sampling with burn-in and visualize convergence"""
    lattice = generate_lattice(L)
    magnetization_values = []
    
    for step in range(steps):
        lattice = gibbs_sampler(lattice, J, B, T, 1)  # Perform one Gibbs update
        if step >= burn_in:
            magnetization_values.append(compute_magnetization(lattice))    

    # Plot magnetization convergence
    plt.plot(magnetization_values)
    plt.xlabel('Iteration')
    plt.ylabel('Magnetization')
    plt.title('Magnetization Convergence after Burn-in')
    plt.savefig('task1_convergence.png')
    
    return lattice, magnetization_values



def landau_free_energy(T, M, Tc=2.269):
    """Landau Free Energy approximation for the Ising Model"""
    a = 1.0
    b = 1.0
    return a * (T - Tc) * M**2 + b * M**4

def magnetization_vs_temperature(L, J=1, B=0, T_range=(0.5, 5), num_temps=20, steps=10000, burn_in=1000):
    """Compute and plot magnetization as a function of temperature."""
    temperatures = np.linspace(T_range[0], T_range[1], num_temps)
    magnetizations = []
    
    for T in temperatures:
        final_lattice, _ = gibbs_iteration(L, J, B, T, steps, burn_in)
        magnetizations.append(compute_magnetization(final_lattice))
    
    # Plot magnetization vs temperature
    plt.plot(temperatures, magnetizations, marker='o')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs Temperature')
    plt.savefig('task1_magnetization.png')



def magnetization_vs_field(L, J=1, T=1, B_range=(-1, 1), num_fields=20, steps=10000, burn_in=1000):
    """Compute and plot magnetization as a function of external magnetic field."""
    fields = np.linspace(B_range[0], B_range[1], num_fields)
    magnetizations = []
    
    for B in fields:
        final_lattice, _ = gibbs_iteration(L, J, B, T, steps, burn_in)
        magnetizations.append(compute_magnetization(final_lattice))
    
    # Plot magnetization vs field
    plt.plot(fields, magnetizations, marker='o')
    plt.xlabel('Magnetic Field B')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs Magnetic Field')
    plt.savefig('task1_magnetization_field.png')

def specific_heat_vs_temperature(L, J=1, B=0, T_range=(0.5, 5), num_temps=20, steps=10000, burn_in=1000):
    """Compute and plot specific heat as a function of temperature."""
    temperatures = np.linspace(T_range[0], T_range[1], num_temps)
    specific_heats = []
    
    for T in temperatures:
        final_lattice, _ = gibbs_iteration(L, J, B, T, steps, burn_in)
        energy = ising_hamiltonian(final_lattice, J, B)
        specific_heat = (energy ** 2 - energy ** 2) / (T ** 2 * L * L)
        specific_heats.append(specific_heat)
    
    # Plot specific heat vs temperature
    plt.plot(temperatures, specific_heats, marker='o')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.title('Specific Heat vs Temperature')
    plt.savefig('task1_specific_heat.png')



# Define parameters
L = 4  # Lattice size
J = 1  # Coupling constant
B = 0  # External field
T = 1  # Temperature

steps = 10000  # Number of Gibbs sampling steps

# Generate initial lattice
lattice = generate_lattice(L)

# Perform Gibbs sampling
final_lattice = gibbs_sampler(lattice, J, B, T, steps)

burn_in = 1000  # Burn-in period

# Perform Gibbs Iteration
final_lattice_tuple = gibbs_iteration(L, J, B, T=1, steps=steps, burn_in=burn_in)
final_lattice, magnetization_values = final_lattice_tuple  # Unpack correctly


# Compute expectation values
expected_energy, expected_magnetization, states, pdf_values = compute_expectation_values(L, J, B, T)
print(f"Expected Energy: {expected_energy}")
print(f"Expected Magnetization: {expected_magnetization}")

# Plot PDF
plt.plot(pdf_values, marker='o')
plt.title('Probability Density Function of States')
plt.xlabel('State Index')
plt.ylabel('Probability')
plt.savefig('task1_pdf.png')

# Plot final lattice
plt.imshow(final_lattice, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Spin')
plt.title('Final 2D Spin Configuration after Gibbs Sampling')
plt.savefig('task1_lattice.png')

final_lattice, magnetization_values = gibbs_iteration(L, J, B, T=1, steps=steps, burn_in=burn_in)

# Compute Landau Free Energy profile
M_values = np.linspace(-1, 1, 100)
F_values = landau_free_energy(T=1, M=M_values)

plt.plot(M_values, F_values)
plt.xlabel('Magnetization')
plt.ylabel('Free Energy')
plt.title('Landau Free Energy Approximation')
plt.savefig('task1_landau.png') 

# Compute and plot magnetization vs temperature
magnetization_vs_temperature(L, J, B)

# Compute and plot magnetization vs magnetic field
magnetization_vs_field(L, J, T=1)

# Compute and plot specific heat vs temperature
specific_heat_vs_temperature(L, J, B)