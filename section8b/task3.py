import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad, quad, romberg

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
        xi, wi = np.polynomial.legendre.leggauss(N)
        transformed_xi = (self.b - self.a) / 2 * xi + (self.a + self.b) / 2
        return (self.b - self.a) / 2 * sum(w * self.func(x) for x, w in zip(transformed_xi, wi))

# Task 3: Harmonic Oscillator Time Period Calculation
def potential_function(x):
    return x**4

def integrand(x, a):
    return 1 / np.sqrt(potential_function(a) - potential_function(x))

def calculate_period(a, method, N=10):
    prefactor = np.sqrt(8)
    if method == "fixed_quad":
        result, _ = fixed_quad(integrand, 0, a, args=(a,), n=N)
    elif method == "quad":
        result, _ = quad(integrand, 0, a, args=(a,))
    elif method == "romberg":
        result = romberg(integrand, 0, a, args=(a,), show=True, divmax=10)
    else:
        raise ValueError("Invalid method")
    return prefactor * result

# Plotting period vs amplitude
a_values = np.linspace(0.1, 2, 20)
periods = [calculate_period(a, "quad") for a in a_values]

plt.figure(figsize=(8, 6))
plt.plot(a_values, periods, label='Period vs Amplitude')
plt.xlabel("Amplitude a")
plt.ylabel("Time Period T")
plt.title("Time Period of a Harmonic Oscillator")
plt.legend()
plt.grid()
plt.savefig("task3.png")