import numpy as np
import time
import matplotlib.pyplot as plt

def naive_matrix_mult(A, B):
    """Recursive Divide-and-Conquer Matrix Multiplication"""
    n = len(A)
    
    if n == 1:
        return A * B
    
    mid = n // 2
    
    #Split into submatrices of size n/2 x n/2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    
    # Recursively compute the products of the submatrices until n = 1
    C11 = naive_matrix_mult(A11, B11) + naive_matrix_mult(A12, B21)
    C12 = naive_matrix_mult(A11, B12) + naive_matrix_mult(A12, B22)
    C21 = naive_matrix_mult(A21, B11) + naive_matrix_mult(A22, B21)
    C22 = naive_matrix_mult(A21, B12) + naive_matrix_mult(A22, B22)
    
    # Combine the submatrices to form the final product matrix
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C

# Example usage
n = 4  # Matrix size (should be a power of 2 for simplicity)
A = np.random.randint(1, 10, (n, n))
B = np.random.randint(1, 10, (n, n))
C = naive_matrix_mult(A, B)

print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Product Matrix C:")
print(C)


# Master theorem: T(n) = aT(n/b) + f(n)
# Naive matrix multiplication: T(n) = 8T(n/2) + O(n^2)
# a = 8, b = 2, f(n) = O(n^2)
# T(n) = :
# 1) O(n^log_b(a)) if f(n) = O(n^c) and c < log_b(a)
# 2) O(n^c * log(n)) if f(n) = O(n^c) and c = log_b(a)
# 3) O(f(n)) if f(n) = O(n^c) and c > log_b(a)
# In this case, c = 2, log_b(a) = 3, so T(n) = O(n^3)
# 3 is the critical exponent.

# Test different matrix sizes
sizes = [2, 4, 8, 16, 32]  # Matrix sizes (must be powers of 2)
times = []

for n in sizes:
    A = np.random.randint(1, 10, (n, n))
    B = np.random.randint(1, 10, (n, n))
    
    start_time = time.time()
    naive_matrix_mult(A, B)
    end_time = time.time()
    
    times.append(end_time - start_time)
    print(f"Size: {n}x{n}, Time: {times[-1]:.5f} sec")

# Estimate the Exponent
log_n = np.log(sizes)
log_T = np.log(times)
slope, _ = np.polyfit(log_n, log_T, 1)  # Linear fit to find exponent

print(f"Estimated exponent from empirical data: {slope:.2f}")

# Plot the Results
plt.figure(figsize=(8,6))
plt.plot(sizes, times, 'o-', label="Empirical Runtime")
plt.plot(sizes, [t**3 for t in sizes], '--', label=r'Theoretical $O(n^3)$')

plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (seconds)")
plt.xscale("log") # Plotted on a log scale to take into account initial times.
plt.yscale("log")
plt.legend()
plt.grid()
plt.title("Comparison of Empirical and Theoretical Complexity")
plt.savefig("time_comparisons.png")

#This estimated a critical exponent of 2.89, which is close to the theoretical value of 3.