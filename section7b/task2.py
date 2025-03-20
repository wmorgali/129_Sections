import numpy as np
from scipy.integrate import fixed_quad
from matplotlib import pyplot as plt

# ========================================================
# Part a: Exact Surface Area Calculation
# ========================================================

def ellipsoid_surface_area(beta, c):
    """
    Calculate the surface area of an ellipsoid using the given formula.

    The surface area \( A \) of an ellipsoid is given by:
    \[
    A = 2\pi \beta^2 \left(1 + \frac{c}{\beta e} \sin^{-1}(e)\right)
    \]
    where \( e = \sqrt{1 - \frac{\beta^2}{c^2}} \) is the eccentricity of the ellipsoid.

    Parameters:
    beta (float): The semi-axis length in the x and y directions.
    c (float): The semi-axis length in the z direction.

    Returns:
    float: The surface area of the ellipsoid.
    """
    # Calculate the eccentricity e
    if beta >= c:
        raise ValueError("beta must be less than c for the ellipsoid to be valid.")
    
    e = np.sqrt(1 - (beta**2 / c**2))
    
    # Calculate the surface area using the formula
    surface_area = 2 * np.pi * beta**2 * (1 + (c / (beta * e)) * np.arcsin(e))
    
    return surface_area


# ========================================================
# Part b: Approximating Surface Area Using Quadrature
# ========================================================

def integrand(theta, beta, c):
    """
    Integrand for the surface area of the ellipsoid.
    The surface area integral can be expressed as:
    \[
    A = \int_{0}^{2\pi} \int_{0}^{\pi} f(\theta, \phi) \, d\theta \, d\phi
    \]
    This function represents the integrand \( f(\theta) \) after simplifying the integral.
    """
    e = np.sqrt(1 - (beta**2 / c**2))
    return 2 * np.pi * beta**2 * (1 + (c / (beta * e)) * np.sin(theta))

def midpoint_rule(f, a, b, n, beta, c):
    """
    Approximate the integral using the midpoint rule.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    n (int): Number of intervals.
    beta (float): Parameter for the ellipsoid.
    c (float): Parameter for the ellipsoid.

    Returns:
    float: Approximated integral value.
    """
    h = (b - a) / n  # Width of each interval
    x = np.linspace(a + h/2, b - h/2, n)  # Midpoints of intervals
    return h * np.sum(f(x, beta, c))

def gaussian_quadrature(f, a, b, beta, c):
    """
    Approximate the integral using Gaussian quadrature.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    beta (float): Parameter for the ellipsoid.
    c (float): Parameter for the ellipsoid.

    Returns:
    float: Approximated integral value.
    """
    result, _ = fixed_quad(f, a, b, args=(beta, c), n=5)  # Using 5 points for Gaussian quadrature
    return result

# ========================================================
# Part c: Monte Carlo Integration
# ========================================================

def monte_carlo_integration(f, a, b, beta, c, n_samples):
    """
    Approximate the integral using Monte Carlo integration.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    beta (float): Parameter for the ellipsoid.
    c (float): Parameter for the ellipsoid.
    n_samples (int): Number of random samples.

    Returns:
    float: Approximated integral value.
    """
    # Generate random samples uniformly distributed in [a, b]
    theta_samples = np.random.uniform(a, b, n_samples)
    
    # Evaluate the integrand at the random samples
    integrand_values = f(theta_samples, beta, c)
    
    # Approximate the integral using Monte Carlo
    integral_approximation = (b - a) * np.mean(integrand_values)
    
    return integral_approximation

def visualize_monte_carlo_error(f, a, b, beta, c, exact_area):
    """
    Visualize the error of Monte Carlo integration as the number of samples increases.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    beta (float): Parameter for the ellipsoid.
    c (float): Parameter for the ellipsoid.
    exact_area (float): The exact surface area of the ellipsoid.
    """
    # Define the number of samples to test
    sample_sizes = [10, 100, 1000, 10000, 100000]
    
    # Store the errors for each sample size
    errors = []
    
    # Perform Monte Carlo integration for each sample size
    for n_samples in sample_sizes:
        mc_area = monte_carlo_integration(f, a, b, beta, c, n_samples)
        error = np.abs(mc_area - exact_area)
        errors.append(error)
    
    # Plot the error vs. number of samples
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, errors, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Monte Carlo Integration Error vs. Number of Samples')
    plt.grid(True)
    plt.savefig('monte_carlo_error.png')

# ========================================================
# Part d: Importance Sampling and Inverse Transformation Sampling
# ========================================================

def importance_sampling(f, a, b, beta, c, n_samples, proposal_func):
    """
    Approximate the integral using importance sampling.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    beta (float): Parameter for the ellipsoid.
    c (float): Parameter for the ellipsoid.
    n_samples (int): Number of random samples.
    proposal_func (function): The proposal function for importance sampling.

    Returns:
    float: Approximated integral value.
    """
    # Generate random samples from the proposal distribution
    theta_samples = np.random.uniform(a, b, n_samples)
    
    # Evaluate the integrand and proposal function at the samples
    integrand_values = f(theta_samples, beta, c)
    proposal_values = proposal_func(theta_samples)
    
    # Approximate the integral using importance sampling
    integral_approximation = np.mean(integrand_values / proposal_values) * (b - a)
    
    return integral_approximation

def inverse_transform_sampling(f, a, b, beta, c, n_samples, proposal_func):
    """
    Approximate the integral using inverse transformation sampling.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    beta (float): Parameter for the ellipsoid.
    c (float): Parameter for the ellipsoid.
    n_samples (int): Number of random samples.
    proposal_func (function): The proposal function for inverse transformation sampling.

    Returns:
    float: Approximated integral value.
    """
    # Generate random samples using inverse transformation sampling
    u = np.random.uniform(0, 1, n_samples)
    theta_samples = np.arccos(1 - 2 * u)  # Example of inverse transform sampling
    
    # Evaluate the integrand at the samples
    integrand_values = f(theta_samples, beta, c)
    
    # Approximate the integral using inverse transformation sampling
    integral_approximation = np.mean(integrand_values) * (b - a)
    
    return integral_approximation

def visualize_sampling_differences(f, a, b, beta, c, exact_area):
    """
    Visualize the differences between uniform sampling, importance sampling, and inverse transformation sampling.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    beta (float): Parameter for the ellipsoid.
    c (float): Parameter for the ellipsoid.
    exact_area (float): The exact surface area of the ellipsoid.
    """
    # Define the number of samples to test
    sample_sizes = [10, 100, 1000, 10000, 100000]
    
    # Store the results for each sampling method
    uniform_results = []
    importance_results = []
    inverse_transform_results = []
    
    # Define the proposal functions
    def q1(x):
        return np.exp(-3 * x)  # Proposal function for importance sampling

    def q2(x):
        return np.sin(5 * x)**2  # Proposal function for inverse transformation sampling

    # Perform Monte Carlo integration for each sample size
    for n_samples in sample_sizes:
        # Uniform sampling
        uniform_area = monte_carlo_integration(f, a, b, beta, c, n_samples)
        uniform_results.append(uniform_area)
        
        # Importance sampling
        importance_area = importance_sampling(f, a, b, beta, c, n_samples, q1)
        importance_results.append(importance_area)
        
        # Inverse transformation sampling
        inverse_transform_area = inverse_transform_sampling(f, a, b, beta, c, n_samples, q2)
        inverse_transform_results.append(inverse_transform_area)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(sample_sizes, uniform_results, marker='o', linestyle='-', color='b', label='Uniform Sampling')
    plt.plot(sample_sizes, importance_results, marker='o', linestyle='-', color='r', label='Importance Sampling')
    plt.plot(sample_sizes, inverse_transform_results, marker='o', linestyle='-', color='g', label='Inverse Transform Sampling')
    plt.axhline(y=exact_area, color='k', linestyle='--', label='Exact Area')
    plt.xscale('log')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Approximated Surface Area')
    plt.title('Comparison of Sampling Methods for Monte Carlo Integration')
    plt.legend()
    plt.grid(True)
    plt.savefig('sampling_methods_comparison.png')





# Example usage
if __name__ == "__main__":
    # Define the values of beta and c
    beta = 1.0  # Example value for beta
    c = 2.0     # Example value for c

    # Part a: Exact surface area
    try:
        exact_area = ellipsoid_surface_area(beta, c)
        print(f"Exact surface area of the ellipsoid with beta={beta} and c={c} is: {exact_area:.4f}")
    except ValueError as e:
        print(e)

    # Part b: Approximating surface area using quadrature
    # Limits of integration (theta ranges from 0 to pi)
    a, b = 0, np.pi

    # Midpoint Rule
    n = 100  # Number of intervals
    midpoint_area = midpoint_rule(integrand, a, b, n, beta, c)
    print(f"Approximated surface area using Midpoint Rule: {midpoint_area:.4f}")

    # Gaussian Quadrature
    gaussian_area = gaussian_quadrature(integrand, a, b, beta, c)
    print(f"Approximated surface area using Gaussian Quadrature: {gaussian_area:.4f}")

    # Part c: Monte Carlo Integration with Error Visualization
    visualize_monte_carlo_error(integrand, a, b, beta, c, exact_area)

    # Part d: Importance Sampling and Inverse Transformation Sampling with Visualization
    visualize_sampling_differences(integrand, a, b, beta, c, exact_area)

import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# Part e: Box-Muller Transform for Gaussian Sampling
# ========================================================

def box_muller_transform(n_samples, mu=0, sigma=1):
    """
    Generate Gaussian-distributed samples using the Box-Muller transform.

    Parameters:
    n_samples (int): Number of samples to generate.
    mu (float): Mean of the Gaussian distribution (default is 0).
    sigma (float): Standard deviation of the Gaussian distribution (default is 1).

    Returns:
    numpy.ndarray: Array of Gaussian-distributed samples.
    """
    # Generate uniform random numbers
    u1 = np.random.uniform(0, 1, n_samples)
    u2 = np.random.uniform(0, 1, n_samples)
    
    # Apply the Box-Muller transform
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    
    # Combine the two sets of samples and scale by mu and sigma
    samples = mu + sigma * z0  # We only need one set of samples (z0 or z1)
    
    return samples

def plot_gaussian_histogram(samples, n_samples, mu, sigma):
    """
    Plot a histogram of Gaussian-distributed samples.

    Parameters:
    samples (numpy.ndarray): Array of Gaussian-distributed samples.
    n_samples (int): Number of samples.
    mu (float): Mean of the Gaussian distribution.
    sigma (float): Standard deviation of the Gaussian distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label='Sampled Data')
    
    # Plot the theoretical Gaussian distribution
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    plt.plot(x, y, 'r-', label='Theoretical Gaussian')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Histogram of Gaussian Samples (N={n_samples}, μ={mu}, σ={sigma})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'gaussian_histogram_{n_samples}.png')


# Example usage
if __name__ == "__main__":
    # Define the parameters for the Gaussian distribution
    mu = 0  # Mean
    sigma = 1  # Standard deviation
    
    # Define the sample sizes to test
    sample_sizes = [10, 100, 1000, 10000, 100000]
    
    # Generate and plot Gaussian samples for each sample size
    for n_samples in sample_sizes:
        # Generate Gaussian samples using the Box-Muller transform
        samples = box_muller_transform(n_samples, mu, sigma)
        
        # Plot the histogram of the samples
        plot_gaussian_histogram(samples, n_samples, mu, sigma)


import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# Part f: Monte Carlo Integration with Gaussian Proposal Function
# ========================================================

def gaussian_proposal_integration(f, a, b, mu, sigma, n_samples):
    """
    Perform Monte Carlo integration using Gaussian-distributed samples as the proposal function.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    mu (float): Mean of the Gaussian proposal function.
    sigma (float): Standard deviation of the Gaussian proposal function.
    n_samples (int): Number of random samples.

    Returns:
    float: Approximated integral value.
    """
    # Generate Gaussian-distributed samples
    samples = np.random.normal(mu, sigma, n_samples)
    
    # Ensure samples are within the integration limits [a, b]
    samples = samples[(samples >= a) & (samples <= b)]
    
    # Evaluate the integrand at the samples
    integrand_values = f(samples)
    
    # Approximate the integral using Monte Carlo
    integral_approximation = np.mean(integrand_values) * (b - a)
    
    return integral_approximation

def visualize_gaussian_integration(f, a, b, mu, sigma, exact_area):
    """
    Visualize the results of Monte Carlo integration using Gaussian-distributed samples.

    Parameters:
    f (function): The integrand function.
    a (float): Lower limit of integration.
    b (float): Upper limit of integration.
    mu (float): Mean of the Gaussian proposal function.
    sigma (float): Standard deviation of the Gaussian proposal function.
    exact_area (float): The exact value of the integral.
    """
    # Define the number of samples to test
    sample_sizes = [10, 100, 1000, 10000, 100000]
    
    # Store the results for each sample size
    results = []
    
    # Perform Monte Carlo integration for each sample size
    for n_samples in sample_sizes:
        result = gaussian_proposal_integration(f, a, b, mu, sigma, n_samples)
        results.append(result)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, results, marker='o', linestyle='-', color='b', label='Gaussian Proposal Integration')
    plt.axhline(y=exact_area, color='r', linestyle='--', label='Exact Area')
    plt.xscale('log')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Approximated Integral Value')
    plt.title('Monte Carlo Integration with Gaussian Proposal Function')
    plt.legend()
    plt.grid(True)
    plt.savefig("gaussian_proposal_integration.png")


# Example usage
if __name__ == "__main__":
    # Define the integrand function (example: f(x) = sin(x))
    def integrand(x):
        return np.sin(x)

    # Define the integration limits
    a, b = 0, np.pi  # Example limits for integration

    # Define the parameters for the Gaussian proposal function
    mu = 1.0  # Mean of the Gaussian proposal function
    sigma = 0.5  # Standard deviation of the Gaussian proposal function

    # Calculate the exact area (for comparison)
    exact_area = 2.0  # Exact integral of sin(x) from 0 to pi is 2.0

    # Visualize the results of Monte Carlo integration with Gaussian proposal function
    visualize_gaussian_integration(integrand, a, b, mu, sigma, exact_area)