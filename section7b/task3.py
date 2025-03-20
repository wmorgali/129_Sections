import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# Part a: Rejection Sampling with Uniform Proposal Function
# ========================================================

def rejection_sampling_uniform(pdf, a, b, t_f, n_samples):
    """
    Perform rejection sampling using a uniform proposal function.

    Parameters:
    pdf (function): The target probability density function.
    a (float): Parameter for the PDF.
    b (float): Parameter for the PDF.
    t_f (float): Upper limit for the uniform proposal function.
    n_samples (int): Number of samples to generate.

    Returns:
    numpy.ndarray: Array of accepted samples.
    float: Rejection ratio (number of accepts / number of rejects).
    """
    # Initialize lists to store accepted and rejected samples
    accepted_samples = []
    rejected_samples = []
    
    # Find the maximum value of the PDF
    t_values = np.linspace(0, t_f, 1000)
    pdf_values = pdf(t_values, a, b)
    max_pdf = np.max(pdf_values)
    
    # Perform rejection sampling
    while len(accepted_samples) < n_samples:
        # Generate a random sample from the uniform proposal
        t_sample = np.random.uniform(0, t_f)
        
        # Generate a random value for comparison
        u = np.random.uniform(0, max_pdf)
        
        # Accept or reject the sample
        if u <= pdf(t_sample, a, b):
            accepted_samples.append(t_sample)
        else:
            rejected_samples.append(t_sample)
    
    # Calculate the rejection ratio
    rejection_ratio = len(accepted_samples) / len(rejected_samples)
    
    return np.array(accepted_samples), rejection_ratio

# ========================================================
# Part b: Rejection Sampling with Exponential Proposal Function
# ========================================================

def rejection_sampling_exponential(pdf, a, b, rate, n_samples):
    """
    Perform rejection sampling using an exponential proposal function.

    Parameters:
    pdf (function): The target probability density function.
    a (float): Parameter for the PDF.
    b (float): Parameter for the PDF.
    rate (float): Rate parameter for the exponential proposal function.
    n_samples (int): Number of samples to generate.

    Returns:
    numpy.ndarray: Array of accepted samples.
    float: Rejection ratio (number of accepts / number of rejects).
    """
    # Initialize lists to store accepted and rejected samples
    accepted_samples = []
    rejected_samples = []
    
    # Find the maximum value of the PDF
    t_values = np.linspace(0, 10, 1000)  # Adjust the upper limit as needed
    pdf_values = pdf(t_values, a, b)
    max_pdf = np.max(pdf_values)
    
    # Perform rejection sampling
    while len(accepted_samples) < n_samples:
        # Generate a random sample from the exponential proposal
        t_sample = np.random.exponential(1 / rate)
        
        # Generate a random value for comparison
        u = np.random.uniform(0, max_pdf)
        
        # Accept or reject the sample
        if u <= pdf(t_sample, a, b):
            accepted_samples.append(t_sample)
        else:
            rejected_samples.append(t_sample)
    
    # Calculate the rejection ratio
    rejection_ratio = len(accepted_samples) / len(rejected_samples)
    
    return np.array(accepted_samples), rejection_ratio

# ========================================================
# Define the PDF and parameters
# ========================================================

def pdf(t, a, b):
    """
    The target probability density function:
    \[
    p(x) = e^{-bt} \cos^2(at), \quad t \geq 0
    \]
    """
    return np.exp(-b * t) * np.cos(a * t)**2

# Parameters for the PDF
a = 4
b = 4

# ========================================================
# Part a: Rejection Sampling with Uniform Proposal Function
# ========================================================

# Define the upper limit for the uniform proposal function
t_f = 2  # Adjust as needed

# Define the number of samples to generate
n_samples = 10000

# Perform rejection sampling with uniform proposal
accepted_uniform, rejection_ratio_uniform = rejection_sampling_uniform(pdf, a, b, t_f, n_samples)

# Plot the histogram of accepted samples
plt.figure(figsize=(12, 6))
plt.hist(accepted_uniform, bins=50, density=True, alpha=0.6, color='b', label='Accepted Samples (Uniform Proposal)')
plt.title(f'Rejection Sampling with Uniform Proposal (Rejection Ratio: {rejection_ratio_uniform:.4f})')
plt.xlabel('t')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('task3_1.png')

# ========================================================
# Part b: Rejection Sampling with Exponential Proposal Function
# ========================================================

# Define the rate parameter for the exponential proposal function
rate = 2  # Adjust as needed

# Perform rejection sampling with exponential proposal
accepted_exponential, rejection_ratio_exponential = rejection_sampling_exponential(pdf, a, b, rate, n_samples)

# Plot the histogram of accepted samples
plt.figure(figsize=(12, 6))
plt.hist(accepted_exponential, bins=50, density=True, alpha=0.6, color='r', label='Accepted Samples (Exponential Proposal)')
plt.title(f'Rejection Sampling with Exponential Proposal (Rejection Ratio: {rejection_ratio_exponential:.4f})')
plt.xlabel('t')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('task3_2.png')