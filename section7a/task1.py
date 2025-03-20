import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.special import beta, gamma, factorial
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize_scalar

# Load datasets
def load_dataset(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data)

dataset_1 = load_dataset('dataset_1.json')
dataset_2 = load_dataset('dataset_2.json')
dataset_3 = load_dataset('dataset_3.json')

# Function to calculate the posterior distribution
def bayesian_inference(data, batch_size=50):
    N = len(data)
    M = np.sum(data)
    
    # Prior is uniform, so P(p) = 1
    # Posterior is Beta(M + 1, N - M + 1)
    alpha = M + 1
    beta_ = N - M + 1
    
    # Calculate expectation value and variance
    expectation = alpha / (alpha + beta_)
    variance = (alpha * beta_) / ((alpha + beta_) ** 2 * (alpha + beta_ + 1))
    
    return expectation, variance, alpha, beta_

# Function to plot the posterior distribution
def plot_posterior(alpha, beta_, dataset_name):
    p = np.linspace(0, 1, 1000)
    posterior = beta_dist.pdf(p, alpha, beta_)
    
    plt.plot(p, posterior, label=f'{dataset_name} Posterior')
    plt.xlabel('p')
    plt.ylabel('Probability Density')
    plt.title(f'Posterior Distribution for {dataset_name}')
    plt.legend()
    plt.savefig(f'{dataset_name}_posterior.png')

# Function to compute the MLE for p
def maximum_likelihood_estimation(data):
    N = len(data)
    M = np.sum(data)
    return M / N  # MLE for p is simply the fraction of heads

# Function to compute Stirling's approximation
def stirling_approximation(n):
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)

# Function to plot Stirling's approximation vs. Gamma function
def plot_stirling_approximation():
    n_values = np.arange(1, 11)
    gamma_values = np.log(factorial(n_values))  # Use scipy.special.factorial for vectorized operations
    stirling_values = stirling_approximation(n_values)
    
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Gamma function vs Stirling's approximation
    plt.subplot(2, 1, 1)
    plt.scatter(n_values, gamma_values, label='Gamma Function (log(n!))')
    plt.plot(n_values, stirling_values, label="Stirling's Approximation", color='red')
    plt.xlabel('n')
    plt.ylabel('log(n!)')
    plt.title("Stirling's Approximation vs Gamma Function")
    plt.legend()
    plt.savefig('stirling_vs_gamma.png')
    
    # Plot 2: Difference between Gamma function and Stirling's approximation
    plt.subplot(2, 1, 2)
    difference = gamma_values - stirling_values
    plt.plot(n_values, difference, label="Difference (Gamma - Stirling)", color='green')
    plt.xlabel('n')
    plt.ylabel('Difference')
    plt.title("Difference Between Gamma Function and Stirling's Approximation")
    plt.legend()

    
    plt.tight_layout()
    plt.savefig('stirling_vs_gamma_diff.png')

# Function for bootstrapping
def bootstrap(data, num_iterations=100, sample_sizes=None):
    if sample_sizes is None:
        sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]
    
    bootstrap_results = {}
    
    for size in sample_sizes:
        expectations = []
        variances = []
        
        for _ in range(num_iterations):
            # Resample with replacement
            resampled_data = np.random.choice(data, size=size, replace=True)
            M = np.sum(resampled_data)
            N = size
            alpha = M + 1
            beta_ = N - M + 1
            expectation = alpha / (alpha + beta_)
            variance = (alpha * beta_) / ((alpha + beta_) ** 2 * (alpha + beta_ + 1))
            
            expectations.append(expectation)
            variances.append(variance)
        
        bootstrap_results[size] = {'expectations': expectations, 'variances': variances}
    
    return bootstrap_results

# Function to plot bootstrapping results
def plot_bootstrap_results(bootstrap_results, dataset_name):
    sample_sizes = list(bootstrap_results.keys())
    
    plt.figure(figsize=(15, 10))
    
    for i, size in enumerate(sample_sizes):
        expectations = bootstrap_results[size]['expectations']
        variances = bootstrap_results[size]['variances']
        
        # Plot expectation histogram
        plt.subplot(3, 3, i + 1)
        plt.hist(expectations, bins=20, alpha=0.7, label=f'Expectation (size={size})')
        plt.xlabel('Expectation')
        plt.ylabel('Frequency')
        plt.title(f'Expectation for Sample Size {size}')
        plt.legend()
        plt.savefig(f'{dataset_name}_bootstrap.png')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_bootstrap.png')
    
    plt.figure(figsize=(15, 10))
    
    for i, size in enumerate(sample_sizes):
        expectations = bootstrap_results[size]['expectations']
        variances = bootstrap_results[size]['variances']
        
        # Plot variance histogram
        plt.subplot(3, 3, i + 1)
        plt.hist(variances, bins=20, alpha=0.7, label=f'Variance (size={size})', color='orange')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.title(f'Variance for Sample Size {size}')
        plt.legend()
        plt.savefig(f'{dataset_name}_bootstrap_variance.png')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_bootstrap_variance.png')



# Analyze each dataset for part a)
datasets = {'Dataset 1': dataset_1, 'Dataset 2': dataset_2, 'Dataset 3': dataset_3}
results = {}

for name, data in datasets.items():
    expectation, variance, alpha, beta_ = bayesian_inference(data)
    results[name] = {'Expectation': expectation, 'Variance': variance, 'Alpha': alpha, 'Beta': beta_}
    print(f'{name}: Expectation = {expectation:.4f}, Variance = {variance:.4f}')
    plot_posterior(alpha, beta_, name)

# Print results for part a)
for name, result in results.items():
    print(f'{name}:')
    print(f'  Expectation: {result["Expectation"]:.4f}')
    print(f'  Variance: {result["Variance"]:.4f}')
    print(f'  Alpha: {result["Alpha"]}')
    print(f'  Beta: {result["Beta"]}')

# Part b) Maximum Likelihood Estimation (MLE)
for name, data in datasets.items():
    mle_p = maximum_likelihood_estimation(data)
    print(f'{name}: MLE for p = {mle_p:.4f}')

# Part b) Stirling's approximation
plot_stirling_approximation()

# Part e) Bootstrapping
for name, data in datasets.items():
    print(f'Bootstrapping for {name}:')
    bootstrap_results = bootstrap(data)
    plot_bootstrap_results(bootstrap_results, name)