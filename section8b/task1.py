import numpy as np

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


# Example usage
def example_function(x):
    return x**2

quad = Quadrature(example_function, 0, 1)
print("Midpoint Rule:", quad.midpoint_rule())
print("Trapezoidal Rule:", quad.trapezoidal_rule())
print("Simpson's Rule:", quad.simpsons_rule())
print("Gauss-Legendre Quadrature (N=2):", quad.gauss_legendre(2))

gauss_quad = GaussQuad(example_function, 0, 1, 3)
roots, weights = gauss_quad.newton_method(3)
print("Gauss-Legendre Roots:", roots)
print("Gauss-Legendre Weights:", weights)

