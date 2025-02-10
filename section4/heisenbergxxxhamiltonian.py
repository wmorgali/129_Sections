import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt

def pauli_matrices():
    """Returns the Pauli spin matrices and raising/lowering operators."""
    Sx = np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Sp = Sx + 1j * Sy  # S+ = Sx + iSy
    Sm = Sx - 1j * Sy  # S- = Sx - iSy
    return Sp, Sm, Sz

def construct_hamiltonian(N, J=1.0):
    """Constructs the Heisenberg XXX Hamiltonian for N spins with periodic boundary conditions."""
    dim = 2**N  # Hilbert space dimension
    Sp, Sm, Sz = pauli_matrices()
    
    H = sp.lil_matrix((dim, dim), dtype=complex)  # Use sparse matrix
    
    # Basis states represented as binary numbers
    for i in range(N):
        j = (i + 1) % N  # Periodic boundary condition
        
        for state in range(dim):
            # Convert state index to binary representation
            bin_state = format(state, f'0{N}b')
            
            # Extract spin values (1 = up, 0 = down)
            si = 1 if bin_state[i] == '1' else -1
            sj = 1 if bin_state[j] == '1' else -1
            
            # Sz Sz interaction
            H[state, state] += J * (si * sj) / 4
            
            # S+ S- and S- S+ terms (hopping terms)
            if bin_state[i] != bin_state[j]:
                flipped_state = list(bin_state)
                flipped_state[i], flipped_state[j] = flipped_state[j], flipped_state[i]
                flipped_state = int("".join(flipped_state), 2)  # Convert back to integer index
                H[state, flipped_state] += -J / 2
    
    return H.tocsr()  # Convert to compressed sparse row format for efficiency

# Example usage
N = 3  # Number of spins
H = construct_hamiltonian(N)

print("Hamiltonian matrix:")
print(H.toarray())  # Convert sparse matrix to dense for visualization


def qr_algorithm(H, tol=1e-10, max_iter=1000):
    """Performs the QR algorithm to diagonalize a matrix H."""
    Hk = H.toarray().astype(complex)  # Convert sparse to dense
    for _ in range(max_iter):
        Q, R = np.linalg.qr(Hk)  # QR decomposition
        Hk_next = R @ Q  # Compute next iteration
        if np.allclose(Hk, Hk_next, atol=tol):  # Convergence check
            break
        Hk = Hk_next
    return np.diag(Hk)  # Return diagonal elements

# Example usage
N = 3  # Number of spins
H = construct_hamiltonian(N)
diagonal_H = qr_algorithm(H)

print("Diagonalized Hamiltonian:")
print(diagonal_H)

def lu_decomposition_solver(omega, H):
    """Solves (omega*I - H)G = I using LU decomposition."""
    dim = H.shape[0]
    A = omega * np.eye(dim) - H.toarray()
    lu, piv = la.lu_factor(A)
    G = la.lu_solve((lu, piv), np.eye(dim))
    return G

def cholesky_decomposition_solver(omega, H):
    """Solves (omega*I - H)G = I using Cholesky decomposition."""
    dim = H.shape[0]
    A = omega * np.eye(dim) - H.toarray()
    L = la.cholesky(A, lower=True)
    G = la.cho_solve((L, True), np.eye(dim))
    return G

def plot_greens_function(N, omega_range):
    """Plots the trace of the Green's function versus omega."""
    H = construct_hamiltonian(N)
    greens_values = []
    
    for omega in omega_range:
        G = lu_decomposition_solver(omega, H)
        greens_values.append(np.trace(G).real)  # Take the real part
    
    plt.figure(figsize=(8,6))
    plt.plot(omega_range, greens_values, label='Tr(G) vs ω')
    plt.xlabel("Frequency ω")
    plt.ylabel("Tr(G)")
    plt.title("Green's Function Resonances")
    plt.legend()
    plt.grid()
    plt.savefig("greens_function.png")

# Example usage
N = 3  # Number of spins
omega_range = np.linspace(-5, 5, 100)  # Frequency range
plot_greens_function(N, omega_range)

