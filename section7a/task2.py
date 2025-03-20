import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize
from scipy.stats import expon, norm

# Load datasets
def load_dataset(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data)

# Load the decay datasets
vacuum_data = load_dataset('Vacuum_decay_dataset.json')
cavity_data = load_dataset('Cavity_decay_dataset.json')

# Define the probability distribution for decay in a vacuum
def vacuum_decay_probability(x, lambda_):
    """
    Probability of observing a particle at distance x in a vacuum.
    Follows an exponential distribution with decay constant lambda_.
    """
    return (1 / lambda_) * np.exp(-x / lambda_)

# Define the probability distribution for decay in an optical cavity
def cavity_decay_probability(x, lambda_, mu, sigma, fraction):
    """
    Probability of observing a particle at distance x in an optical cavity.
    The cavity modifies a fraction of particles into a different type with Gaussian decay properties.
    """
    exponential_part = (1 - fraction) * (1 / lambda_) * np.exp(-x / lambda_)
    gaussian_part = fraction * norm.pdf(x, loc=mu, scale=sigma)
    return exponential_part + gaussian_part

# Define the negative log-likelihood function for vacuum decay
def vacuum_neg_log_likelihood(lambda_, data):
    """
    Negative log-likelihood function for vacuum decay.
    Minimizing this function will give the MLE for lambda_.
    """
    return -np.sum(np.log(vacuum_decay_probability(data, lambda_)))

# Define the negative log-likelihood function for cavity decay
def cavity_neg_log_likelihood(params, data):
    """
    Negative log-likelihood function for cavity decay.
    Minimizing this function will give the MLE for lambda_, mu, sigma, and fraction.
    """
    lambda_, mu, sigma, fraction = params
    return -np.sum(np.log(cavity_decay_probability(data, lambda_, mu, sigma, fraction)))

# Part a) Infer the decay constants for vacuum and optical cavity

# Step 1: Infer lambda for vacuum decay
# Initial guess for lambda (decay constant)
lambda_guess_vacuum = 1.0

# Minimize the negative log-likelihood for vacuum decay
result_vacuum = minimize(vacuum_neg_log_likelihood, lambda_guess_vacuum, args=(vacuum_data,), bounds=[(0.1, 10.0)])
lambda_vacuum = result_vacuum.x[0]
print(f"Decay constant for vacuum (lambda): {lambda_vacuum:.4f}")

# Step 2: Infer lambda, mu, sigma, and fraction for optical cavity decay
# Initial guesses for lambda, mu, sigma, and fraction
initial_guess_cavity = [1.0, 5.0, 1.0, 0.5]  # lambda, mu, sigma, fraction

# Minimize the negative log-likelihood for cavity decay
result_cavity = minimize(cavity_neg_log_likelihood, initial_guess_cavity, args=(cavity_data,), 
                         bounds=[(0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.0, 1.0)])
lambda_cavity, mu, sigma, fraction = result_cavity.x
print(f"Decay constant for cavity (lambda): {lambda_cavity:.4f}")
print(f"Gaussian mean (mu): {mu:.4f}")
print(f"Gaussian standard deviation (sigma): {sigma:.4f}")
print(f"Fraction of particles with Gaussian decay: {fraction:.4f}")

# Step 3: Calculate the Fisher information matrix for the cavity decay parameters
# The Fisher information matrix is the negative of the Hessian of the log-likelihood function
# We can use the Hessian from the optimization result
fisher_information_matrix = -result_cavity.hess_inv.todense()  # Inverse of the Hessian
print("Fisher Information Matrix for Cavity Decay Parameters:")
print(fisher_information_matrix)

# Step 4: Plot the fitted distributions
# Plot for vacuum decay
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(vacuum_data, bins=50, density=True, alpha=0.6, label='Vacuum Data')
x_values = np.linspace(0, max(vacuum_data), 1000)
plt.plot(x_values, vacuum_decay_probability(x_values, lambda_vacuum), label=f'Fitted Exponential (λ={lambda_vacuum:.4f})', color='red')
plt.xlabel('Decay Distance (x)')
plt.ylabel('Probability Density')
plt.title('Vacuum Decay Distribution')
plt.legend()
plt.savefig('vacuum_decay_distributions.png')

# Plot for cavity decay
plt.subplot(1, 2, 2)
plt.hist(cavity_data, bins=50, density=True, alpha=0.6, label='Cavity Data')
x_values = np.linspace(0, max(cavity_data), 1000)
plt.plot(x_values, cavity_decay_probability(x_values, lambda_cavity, mu, sigma, fraction), 
         label=f'Fitted Mixed Distribution (λ={lambda_cavity:.4f}, μ={mu:.4f}, σ={sigma:.4f}, fraction={fraction:.4f})', color='red')
plt.xlabel('Decay Distance (x)')
plt.ylabel('Probability Density')
plt.title('Cavity Decay Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('cavity_decay_distributions.png')



# Step 1: Define the null hypothesis
# Null hypothesis: The optical cavity does not modify the particles, so the decay follows the same exponential distribution as in the vacuum.
# We will test this hypothesis by comparing the likelihood of the null model (exponential) to the alternative model (mixed exponential-Gaussian).

# Step 2: Calculate the log-likelihood for the null model (exponential decay)
def null_model_log_likelihood(data, lambda_):
    """
    Log-likelihood for the null model (exponential decay).
    """
    return np.sum(np.log(vacuum_decay_probability(data, lambda_)))

# Calculate the log-likelihood for the null model using the vacuum decay constant
null_log_likelihood = null_model_log_likelihood(cavity_data, lambda_vacuum)

# Step 3: Calculate the log-likelihood for the alternative model (mixed exponential-Gaussian)
def alternative_model_log_likelihood(data, lambda_, mu, sigma, fraction):
    """
    Log-likelihood for the alternative model (mixed exponential-Gaussian decay).
    """
    return np.sum(np.log(cavity_decay_probability(data, lambda_, mu, sigma, fraction)))

# Calculate the log-likelihood for the alternative model using the fitted parameters
alternative_log_likelihood = alternative_model_log_likelihood(cavity_data, lambda_cavity, mu, sigma, fraction)

# Step 4: Perform a likelihood ratio test
# The test statistic is D = -2 * (log(L_null) - log(L_alternative))
D = -2 * (null_log_likelihood - alternative_log_likelihood)
print(f"Likelihood Ratio Test Statistic (D): {D:.4f}")

# Step 5: Determine the critical value for 95% confidence
# The test statistic D follows a chi-squared distribution with degrees of freedom equal to the difference in the number of parameters.
# The null model has 1 parameter (lambda), and the alternative model has 4 parameters (lambda, mu, sigma, fraction).
# Degrees of freedom = 4 - 1 = 3
from scipy.stats import chi2

critical_value = chi2.ppf(0.95, df=3)  # 95% confidence level, df=3
print(f"Critical Value for 95% Confidence (Chi-squared, df=3): {critical_value:.4f}")

# Step 6: Compare the test statistic to the critical value
if D > critical_value:
    print("Reject the null hypothesis: There is significant evidence that the optical cavity modifies the particles.")
else:
    print("Fail to reject the null hypothesis: There is not enough evidence to conclude that the optical cavity modifies the particles.")

# Step 7: Calculate the p-value
p_value = 1 - chi2.cdf(D, df=3)
print(f"p-value: {p_value:.4f}")

# Step 8: Plot the chi-squared distribution and the test statistic
x_values = np.linspace(0, 15, 1000)
chi2_pdf = chi2.pdf(x_values, df=3)

plt.figure(figsize=(8, 6))
plt.plot(x_values, chi2_pdf, label='Chi-squared Distribution (df=3)', color='blue')
plt.axvline(D, color='red', linestyle='--', label=f'Test Statistic (D = {D:.4f})')
plt.axvline(critical_value, color='green', linestyle='--', label=f'Critical Value (95% Confidence)')
plt.fill_between(x_values, chi2_pdf, where=(x_values >= critical_value), color='red', alpha=0.3, label='Rejection Region')
plt.xlabel('Test Statistic (D)')
plt.ylabel('Probability Density')
plt.title('Likelihood Ratio Test: Chi-squared Distribution')
plt.legend()
plt.savefig('likelihood_ratio_test.png')