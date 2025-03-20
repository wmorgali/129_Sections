# Part A:

"""
Part A Explanation:
- We are asked to draw the forward and backward computational directed graphs for a 4-layer MLP.
- The dimensions of weights and biases should be annotated near each graph node.

Forward Pass:
1. Input Layer: X_i (dimension A, e.g., 784 for MNIST images).
2. Layer 1: W1 (A x 6), b1 (6), output Z1 = X_i @ W1 + b1, activation A1 = ReLU(Z1).
3. Layer 2: W2 (6 x 4), b2 (4), output Z2 = A1 @ W2 + b2, activation A2 = ReLU(Z2).
4. Layer 3: W3 (4 x 3), b3 (3), output Z3 = A2 @ W3 + b3, activation A3 = ReLU(Z3).
5. Layer 4: W4 (3 x 2), b4 (2), output Z4 = A3 @ W4 + b4, activation A4 = Sigmoid(Z4).

Backward Pass:
- Gradients are computed using the chain rule, starting from the output layer and propagating backward.
- For each layer, compute gradients of the loss with respect to weights, biases, and inputs.
"""

# Part B: Construct a class for MLP

import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation_functions):
        """
        Initialize the MLP.
        
        Args:
            layer_sizes (list): List of integers representing the number of neurons in each layer.
            activation_functions (list): List of activation functions for each layer (e.g., 'relu', 'sigmoid').
        """
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU, Xavier for Sigmoid
            if activation_functions[i] == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros(layer_sizes[i + 1]))
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function."""
        return (x > 0).astype(float)
    
    def sigmoid_derivative(self, x):
        """Derivative of Sigmoid activation function."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def softmax_derivative(self, x):
        """Derivative of Softmax activation function."""
        s = self.softmax(x)
        return s * (1 - s)  # Simplified derivative for softmax
    
    def forward(self, X):
        """
        Perform forward propagation.
        
        Args:
            X (np.array): Input data of shape (batch_size, input_dim).
        
        Returns:
            np.array: Output of the network.
        """
        self.activations = [X]  # Store activations for each layer
        self.z_values = []      # Store pre-activation values
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if self.activation_functions[i] == 'relu':
                a = self.relu(z)
            elif self.activation_functions[i] == 'sigmoid':
                a = self.sigmoid(z)
            elif self.activation_functions[i] == 'softmax':
                a = self.softmax(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        """
        Perform backward propagation and update weights and biases.
        
        Args:
            X (np.array): Input data of shape (batch_size, input_dim).
            y (np.array): True labels of shape (batch_size, output_dim).
            learning_rate (float): Learning rate for gradient descent.
        """
        # Forward pass to compute activations
        self.forward(X)
        
        # Compute the loss gradient (assuming cross-entropy loss with softmax)
        m = X.shape[0]
        dA = self.activations[-1] - y  # Derivative of loss w.r.t. output
        
        for i in reversed(range(len(self.weights))):
            # Compute gradients for weights and biases
            if self.activation_functions[i] == 'relu':
                dZ = dA * self.relu_derivative(self.z_values[i])
            elif self.activation_functions[i] == 'sigmoid':
                dZ = dA * self.sigmoid_derivative(self.z_values[i])
            elif self.activation_functions[i] == 'softmax':
                dZ = dA  # Simplified derivative for softmax with cross-entropy loss
            
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0) / m
            dA = np.dot(dZ, self.weights[i].T)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

# Example usage:
# Define a 4-layer MLP with layer sizes [784, 128, 64, 10] and activations ['relu', 'relu', 'softmax']
mlp = MLP(layer_sizes=[784, 128, 64, 10], activation_functions=['relu', 'relu', 'softmax'])

# Forward pass example
X = np.random.randn(10, 784)  # Random input data (batch_size=10, input_dim=784)
output = mlp.forward(X)
print("Output shape:", output.shape)

# Backward pass example
y = np.random.randn(10, 10)  # Ensure labels have shape (batch_size, 10)
mlp.backward(X, y, learning_rate=0.01)