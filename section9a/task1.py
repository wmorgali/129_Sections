import numpy as np
from scipy.integrate import quad, fixed_quad

# Constants
k_B = 1.38064852e-23  # Boltzmann constant, J/K
h = 6.626e-34  # Planck's constant, J·s
c = 3e8  # Speed of light, m/s
pi = np.pi
hbar = h / (2 * pi)

# Prefactor for the Stefan-Boltzmann constant
prefactor = (k_B**4) / (c**2 * hbar**3 * 4 * pi**2)

# Part A: Define the integrand and perform the integral
def integrand(x):
    return (x**3) / (np.exp(x) - 1)

# Part B: Calculate the Stefan-Boltzmann constant using fixed_quad
def calculate_sigma():
    # Change of variables: x = t / (1 - t), dx = dt / (1 - t)^2
    def transformed_integrand(t):
        x = t / (1 - t)
        return (x**3) / (np.exp(x) - 1) * (1 / (1 - t)**2)

    # Perform the integral using fixed_quad over [0, 1)
    integral, _ = fixed_quad(transformed_integrand, 0, 1, n=100)
    sigma = prefactor * integral
    return sigma

# Part C: Compare with scipy's quad function (supports infinite limits)
def compare_with_quad():
    integral, _ = quad(integrand, 0, np.inf)
    sigma_quad = prefactor * integral
    return sigma_quad

# Main execution
if __name__ == "__main__":
    # Part B: Calculate sigma using fixed_quad
    sigma_fixed_quad = calculate_sigma()
    print(f"Stefan-Boltzmann constant (fixed_quad): {sigma_fixed_quad:.6e} W/m^2K^4")

    # Part C: Calculate sigma using quad
    sigma_quad = compare_with_quad()
    print(f"Stefan-Boltzmann constant (quad): {sigma_quad:.6e} W/m^2K^4")

    # Compare the results
    print(f"Difference between fixed_quad and quad: {abs(sigma_fixed_quad - sigma_quad):.6e}")