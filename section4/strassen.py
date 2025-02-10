import numpy as np
import time
import matplotlib.pyplot as plt

def strassen_matrix_mult(A, B):
    """Recursive Strassen's Matrix Multiplication"""
    n = len(A)
    # Base case
    if n == 1:
        return A * B
    
    mid = n // 2
    
    # Split into submatrices of size n/2 x n/2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    
    # Recursively compute the products of the submatrices until n = 1
    M1 = strassen_matrix_mult(A11 + A22, B11 + B22)
    M2 = strassen_matrix_mult(A21 + A22, B11)
    M3 = strassen_matrix_mult(A11, B12 - B22)
    M4 = strassen_matrix_mult(A22, B21 - B11)
    M5 = strassen_matrix_mult(A11 + A12, B22)
    M6 = strassen_matrix_mult(A21 - A11, B11 + B12)
    M7 = strassen_matrix_mult(A12 - A22, B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    # Combine the submatrices to form the final product matrix
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C

# Example usage
n = 4  # Matrix size (should be a power of 2 for simplicity)
A = np.random.randint(1, 10, (n, n))
B = np.random.randint(1, 10, (n, n))
C = strassen_matrix_mult(A, B)

print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Product Matrix C:")
print(C)

# Master theorem: T(n) = aT(n/b) + f(n)
# T(n) = 7T(n/2) + O(n^2), 7 recursive calls, each subproblem is half the size
# T(n) = :
# 1) O(n^log_b(a)) if f(n) = O(n^c) and c < log_b(a)
# 2) O(n^c * log(n)) if f(n) = O(n^c) and c = log_b(a)
# 3) O(f(n)) if f(n) = O(n^c) and c > log_b(a)
# In this case, c = 2, log_b(a) = 2.807, so T(n) = O(n^2.807)

def measure_runtime(matrix_mult_func, sizes):
    times = []
    for n in sizes:
        A = np.random.randint(1, 10, (n, n))
        B = np.random.randint(1, 10, (n, n))
        
        start_time = time.time()
        matrix_mult_func(A, B)
        end_time = time.time()
        
        times.append(end_time - start_time)
        print(f"Size: {n}x{n}, Time: {times[-1]:.5f} sec")
    
    return sizes, times

# Define matrix sizes
sizes = [2, 4, 8, 16, 32]  # Should be powers of 2
sizes, times = measure_runtime(strassen_matrix_mult, sizes)

# Estimate the exponent using log-log fitting
log_n = np.log(sizes)
log_T = np.log(times)
slope, _ = np.polyfit(log_n, log_T, 1)

print(f"Estimated exponent from empirical data: {slope:.2f}")

# Plot results
plt.figure(figsize=(8,6))
plt.plot(sizes, times, 'o-', label="Empirical Runtime")
plt.plot(sizes, [t**2.81 for t in sizes], '--', label=r'Theoretical $O(n^{2.81})$')

plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (seconds)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.title("Comparison of Empirical and Theoretical Complexity")
plt.savefig("strassen.png")
