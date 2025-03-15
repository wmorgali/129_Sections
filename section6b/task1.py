import numpy as np
import matplotlib.pyplot as plt

# Part a:

# Number of time steps and total time
N = 1000
T = 1.0
dt = T / N

# Generate a Wiener process (Brownian motion)
W = np.cumsum(np.sqrt(dt) * np.random.randn(N))

# Function to demonstrate Ito vs Stratonovich
def ito_integral(f, W, dt):
    ito_sum = 0
    for i in range(1, len(W)):
        ito_sum += f(W[i-1]) * (W[i] - W[i-1])
    return ito_sum

def stratonovich_integral(f, W, dt):
    strato_sum = 0
    for i in range(1, len(W)-1):
        strato_sum += 0.5 * (f(W[i-1]) + f(W[i+1])) * (W[i+1] - W[i])
    return strato_sum

# Example function
f = lambda W: W**2  # f(t, W) = W^2

# Compute integrals
ito_result = ito_integral(f, W, dt)
strato_result = stratonovich_integral(f, W, dt)

# Display results
print(f"Ito Integral Result: {ito_result}")
print(f"Stratonovich Integral Result: {strato_result}")

# Plotting the Wiener process and integrals
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, T, N), W, label="Wiener Process (W_t)")
plt.title("Wiener Process and Stochastic Integrals")
plt.xlabel("Time")
plt.ylabel("Wiener Process Value")
plt.legend()
plt.savefig("weiner_process.png")


# Part b:

# Parameters
mu = 0.1        # Drift term
sigma = 0.2    # Volatility
T = 1.0        # Total time
N = 1000       # Number of time steps
S0 = 1.0       # Initial value

# Time step size
dt = T / N

# Generate a standard Wiener process (Brownian motion)
W = np.random.randn(N) * np.sqrt(dt)
W = np.cumsum(W)  # Cumulative sum to create the Brownian motion

# Geometric Brownian Motion (GBM) formula
# Xt = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * Wt)
t = np.linspace(0, T, N)
Xt = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# Plotting the trajectory
plt.figure(figsize=(10, 5))
plt.plot(t, Xt, label="Geometric Brownian Motion")
plt.xlabel("Time")
plt.ylabel("X_t")
plt.title("Geometric Brownian Motion Simulation")
plt.legend()
plt.grid(True)
plt.savefig("geometric_brownian_motion.png")

# Part c:

# Parameters
mu = 0.1        # Drift term
sigma = 0.2    # Volatility
T = 10.0       # Total time
N = 100        # Number of time steps
S0 = 1.0       # Initial value

# Time step size
dt = T / N

# Generate a standard Wiener process (Brownian motion)
W = np.random.randn(N) * np.sqrt(dt)
W = np.cumsum(W)  # Cumulative sum to create the Brownian motion

# Geometric Brownian Motion (GBM) formula
# Xt = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * Wt)
t = np.linspace(0, T, N)
Xt = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# Ito Integral Calculation
xI_t = np.cumsum(np.diff(np.insert(Xt, 0, 0)))

# Plotting the trajectory
plt.figure(figsize=(10, 5))
plt.plot(t, Xt, label="Geometric Brownian Motion")
plt.plot(t, xI_t, label="Ito Integral")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Geometric Brownian Motion and Ito Integral")
plt.legend()
plt.grid(True)
plt.savefig("ito_integral.png")

# Part d:

# Parameters
mu = 0.1        # Drift term
sigma = 0.2    # Volatility
T = 10.0       # Total time
N = 100        # Number of time steps
S0 = 1.0       # Initial value

# Time step size
dt = T / N

# Generate a standard Wiener process (Brownian motion)
W = np.random.randn(N) * np.sqrt(dt)
W = np.cumsum(W)  # Cumulative sum to create the Brownian motion

# Geometric Brownian Motion (GBM) formula
# Xt = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * Wt)
t = np.linspace(0, T, N)
Xt = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# Ito Integral Calculation
xI_t = np.cumsum(np.diff(np.insert(Xt, 0, 0)))

# Stratonovich Integral Calculation (using midpoint approximation)
dW = np.diff(W)  # Brownian increments
xS_t = np.cumsum(0.5 * (Xt[:-1] + Xt[1:]) * dW)

# Plotting the trajectory
plt.figure(figsize=(10, 5))
plt.plot(t, Xt, label="Geometric Brownian Motion")
plt.plot(t, xI_t, label="Ito Integral")
plt.plot(t[:-1], xS_t, label="Stratonovich Integral")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Geometric Brownian Motion, Ito and Stratonovich Integrals")
plt.legend()
plt.grid(True)
plt.savefig("stratonovich_integral.png")

# Part e:

# Parameters
mu = 0.1        # Drift term
sigma = 0.2    # Volatility
T = 10.0       # Total time
N_values = np.logspace(1, 4, num=4, dtype=int)  # Log-spaced N values
S0 = 1.0       # Initial value

# Store mean and variance for each N
ito_means, ito_vars = [], []
strato_means, strato_vars = [], []

for N in N_values:
    dt = T / N

    # Generate a standard Wiener process (Brownian motion)
    W = np.random.randn(N) * np.sqrt(dt)
    W = np.cumsum(W)  # Cumulative sum to create the Brownian motion

    # Time array
    t = np.linspace(0, T, N)

    # Geometric Brownian Motion (GBM) formula
    Xt = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    # Ito Integral Calculation
    xI_t = np.cumsum(np.diff(np.insert(Xt, 0, 0)))

    # Stratonovich Integral Calculation (using midpoint approximation)
    dW = np.diff(W)
    xS_t = np.cumsum(0.5 * (Xt[:-1] + Xt[1:]) * dW)

    # Store statistics
    ito_means.append(np.mean(xI_t))
    ito_vars.append(np.var(xI_t))
    strato_means.append(np.mean(xS_t))
    strato_vars.append(np.var(xS_t))

# Plotting the statistics
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(N_values, ito_means, marker='o')
plt.xscale("log")
plt.title("Ito Mean")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(N_values, ito_vars, marker='o')
plt.xscale("log")
plt.title("Ito Variance")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(N_values, strato_means, marker='o')
plt.xscale("log")
plt.title("Stratonovich Mean")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(N_values, strato_vars, marker='o')
plt.xscale("log")
plt.title("Stratonovich Variance")
plt.grid(True)

plt.tight_layout()
plt.savefig("integrals_statistics.png")

# Part f:

# Parameters
mu = 0.1        # Drift term
sigma = 0.2    # Volatility
T = 10.0       # Total time
N_values = np.logspace(1, 4, num=4, dtype=int)  # Log-spaced N values
S0 = 1.0       # Initial value

# Store mean and variance for each N
ito_means, ito_vars = [], []
strato_means, strato_vars = [], []

for N in N_values:
    dt = T / N

    # Generate a standard Wiener process (Brownian motion)
    W = np.random.randn(N) * np.sqrt(dt)
    W = np.cumsum(W)  # Cumulative sum to create the Brownian motion

    # Time array
    t = np.linspace(0, T, N)

    # Geometric Brownian Motion (GBM) formula
    Xt = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    # Ito Integral Calculation
    xI_t = np.cumsum(np.diff(np.insert(Xt, 0, 0)))

    # Stratonovich Integral Calculation (using midpoint approximation)
    dW = np.diff(W)
    xS_t = np.cumsum(0.5 * (Xt[:-1] + Xt[1:]) * dW)

    # Functional Dynamics f(X_t) = X_t^2
    f_Xt = Xt**2

    # Ito Stochastic Integral for f(X_t)
    F_I_t = np.cumsum(np.diff(np.insert(f_Xt, 0, 0)))

    # Stratonovich Stochastic Integral for f(X_t)
    F_S_t = np.cumsum(0.5 * (f_Xt[:-1] + f_Xt[1:]) * dW)

    # Store statistics
    ito_means.append(np.mean(F_I_t))
    ito_vars.append(np.var(F_I_t))
    strato_means.append(np.mean(F_S_t))
    strato_vars.append(np.var(F_S_t))

# Plotting the statistics
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(N_values, ito_means, marker='o')
plt.xscale("log")
plt.title("Ito Functional Mean")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(N_values, ito_vars, marker='o')
plt.xscale("log")
plt.title("Ito Functional Variance")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(N_values, strato_means, marker='o')
plt.xscale("log")
plt.title("Stratonovich Functional Mean")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(N_values, strato_vars, marker='o')
plt.xscale("log")
plt.title("Stratonovich Functional Variance")
plt.grid(True)

plt.tight_layout()
plt.savefig("functional_statistics.png")

# Part g:

# Parameters
mu = 0.1        # Drift term
sigma = 0.2    # Volatility
T = 10.0       # Total time
N_values = np.logspace(1, 4, num=4, dtype=int)  # Log-spaced N values
S0 = 1.0       # Initial value

# Store mean and variance for each N
ito_means, ito_vars = [], []
strato_means, strato_vars = [], []

# Autocorrelation function
def autocorrelation(F, lag):
    n = len(F)
    F_mean = np.mean(F)
    autocorr = np.correlate(F - F_mean, F - F_mean, mode='full') / (np.var(F) * n)
    return autocorr[n - 1:n - 1 + lag]

for N in N_values:
    dt = T / N

    # Generate a standard Wiener process (Brownian motion)
    W = np.random.randn(N) * np.sqrt(dt)
    W = np.cumsum(W)  # Cumulative sum to create the Brownian motion

    # Time array
    t = np.linspace(0, T, N)

    # Geometric Brownian Motion (GBM) formula
    Xt = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    # Functional Dynamics f(X_t) = X_t^2
    f_Xt = Xt**2

    # Ito Stochastic Integral for f(X_t)
    F_I_t = np.cumsum(np.diff(np.insert(f_Xt, 0, 0)))

    # Autocorrelation for different stopping times
    for stop_time in [5, 10, 20, 30]:
        stop_index = int(stop_time * N / T)  # Convert to integer
        if stop_index < len(F_I_t):
            autocorr = autocorrelation(F_I_t[:stop_index], lag=100)
            plt.plot(autocorr, label=f"t={stop_time}")


# Plotting the autocorrelation
plt.title("Autocorrelation of Stopping Function")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.grid(True)
plt.savefig("autocorrelation.png")