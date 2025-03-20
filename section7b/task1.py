import numpy as np
import matplotlib.pyplot as plt

def sobol_sequence(n, m1, m2, m3):
    # Initialize direction numbers as integers
    direction_numbers = np.array([m1, m2, m3], dtype=int)
    
    # Initialize the Sobol sequence
    sobol_seq = np.zeros((n, 2))
    
    # Generate the Sobol sequence
    for i in range(n):
        # Calculate the direction numbers
        v1 = direction_numbers[0]
        v2 = direction_numbers[1]
        v3 = direction_numbers[2]
        
        # Generate the Sobol point
        sobol_seq[i, 0] = v1 / (2 ** 3)
        sobol_seq[i, 1] = v2 / (2 ** 3)
        
        # Update direction numbers using bitwise operations
        direction_numbers[0] = v1 ^ (v1 >> 1)
        direction_numbers[1] = v2 ^ (v2 >> 2)
        direction_numbers[2] = v3 ^ (v3 >> 3)
    
    return sobol_seq

# Parameters
n = 50  # Number of points
m1, m2, m3 = 1, 3, 5  # Initial conditions

# Generate Sobol sequence
sobol_seq = sobol_sequence(n, m1, m2, m3)

# Plot the Sobol sequence
plt.figure(figsize=(8, 8))
plt.scatter(sobol_seq[:, 0], sobol_seq[:, 1], c='blue', label='Sobol Sequence')
plt.title('2D Sobol Sequence (First 50 Points)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('sobol_sequence.png')