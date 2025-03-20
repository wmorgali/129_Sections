import numpy as np
import matplotlib.pyplot as plt

class Quadrature:
    def __init__(self, func, a, b):
        self.func = func
        self.a = a
        self.b = b
    
    def midpoint_rule(self):
        midpoint = (self.a + self.b) / 2
        return (self.b - self.a) * self.func(midpoint)
    
    def trapezoidal_rule(self):
        return (self.b - self.a) / 2 * (self.func(self.a) + self.func(self.b))
    
    def simpsons_rule(self):
        midpoint = (self.a + self.b) / 2
        return (self.b - self.a) / 6 * (self.func(self.a) + 4 * self.func(midpoint) + self.func(self.b))
    
    def gauss_legendre(self, N):
        xi, wi = np.polynomial.legendre.leggauss(N)  # Get nodes and weights
        transformed_xi = (self.b - self.a) / 2 * xi + (self.a + self.b) / 2  # Transform to [a, b]
        return (self.b - self.a) / 2 * sum(w * self.func(x) for x, w in zip(transformed_xi, wi))

class GaussQuad(Quadrature):
    def __init__(self, func, a, b, order):
        super().__init__(func, a, b)
        self.order = order
    
    def legendre_polynomial(self, M, x):
        if M == 0:
            return 1
        elif M == 1:
            return x
        else:
            return ((2 * M - 1) * x * self.legendre_polynomial(M - 1, x) - (M - 1) * self.legendre_polynomial(M - 2, x)) / M
    
    def newton_method(self, M, tol=1e-10):
        roots = np.cos(np.pi * (np.arange(1, M + 1) - 0.25) / (M + 0.5))  # Initial guess
        for i in range(10):  # Iterate to refine roots
            Pm = np.array([self.legendre_polynomial(M, r) for r in roots])
            Pm_deriv = np.array([(M * (r * Pm[i] - self.legendre_polynomial(M - 1, r)) / (r**2 - 1)) for i, r in enumerate(roots)])
            roots -= Pm / Pm_deriv  # Newton-Raphson update
        
        weights = 2 / ((1 - roots**2) * (Pm_deriv**2))  # Compute weights
        return roots, weights

def polynomial_function(x, k):
    return x**k

def fermi_dirac_function(x, k):
    return 1 / (1 + np.exp(-k * x))

def compute_relative_error(true_value, approx_value):
    return 2 * abs(true_value - approx_value) / (abs(true_value) + abs(approx_value))

def generate_heatmap(func, k_values, N_values, true_integral_func, method):
    errors = np.zeros((len(k_values), len(N_values)))
    
    for i, k in enumerate(k_values):
        true_value = true_integral_func(k)
        for j, N in enumerate(N_values):
            quad = Quadrature(lambda x: func(x, k), 0, 1)
            if method == Quadrature.gauss_legendre:
                approx_value = method(quad, min(N, 50))  # Cap N for Gauss-Legendre
            else:
                approx_value = method(quad)
            errors[i, j] = compute_relative_error(true_value, approx_value)
    
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(N_values, k_values, errors, shading='auto', cmap='hot')
    plt.colorbar(label='Relative Error')
    plt.xlabel('N')
    plt.ylabel('k')
    plt.title(f'Heatmap for {method.__name__}')
    plt.savefig(f'heatmap_{method.__name__}.png')

def true_integral_polynomial(k):
    return 1 / (k + 1)

def true_integral_fermi_dirac(k):
    return (1 / k) * (np.log(np.exp(k) + 1) - np.log(1 + 1)) if k != 0 else 1.0

k_values = np.arange(0, 6)  # Reduced range
N_values = np.logspace(1, 4, num=8, dtype=int)  # Reduced range

# Generate heatmaps for polynomial function
generate_heatmap(polynomial_function, k_values, N_values, true_integral_polynomial, Quadrature.midpoint_rule)
generate_heatmap(polynomial_function, k_values, N_values, true_integral_polynomial, Quadrature.trapezoidal_rule)
generate_heatmap(polynomial_function, k_values, N_values, true_integral_polynomial, Quadrature.simpsons_rule)
generate_heatmap(polynomial_function, k_values, N_values, true_integral_polynomial, Quadrature.gauss_legendre)

# Generate heatmaps for Fermi-Dirac function
generate_heatmap(fermi_dirac_function, k_values, N_values, true_integral_fermi_dirac, Quadrature.midpoint_rule)
generate_heatmap(fermi_dirac_function, k_values, N_values, true_integral_fermi_dirac, Quadrature.trapezoidal_rule)
generate_heatmap(fermi_dirac_function, k_values, N_values, true_integral_fermi_dirac, Quadrature.simpsons_rule)
generate_heatmap(fermi_dirac_function, k_values, N_values, true_integral_fermi_dirac, Quadrature.gauss_legendre)
