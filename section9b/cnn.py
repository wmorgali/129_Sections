# ---------------------------- 
# Task 2: Convolutional Neural Network (CNN)
# ----------------------------

import numpy as np

# Reuse the MLP code from Task 1 for the fully connected layers
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
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function."""
        return (x > 0).astype(float)
    
    def sigmoid_derivative(self, x):
        """Derivative of Sigmoid activation function."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
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
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, d_L_d_out, learning_rate=0.01):
        """
        Perform backward propagation and update weights and biases.
        
        Args:
            d_L_d_out (np.array): Gradient of the loss with respect to the output.
            learning_rate (float): Learning rate for gradient descent.
        
        Returns:
            np.array: Gradient of the loss with respect to the input.
        """
        # Compute the loss gradient (assuming mean squared error)
        dA = d_L_d_out  # Derivative of loss w.r.t. output
        
        for i in reversed(range(len(self.weights))):
            # Compute gradients for weights and biases
            if self.activation_functions[i] == 'relu':
                dZ = dA * self.relu_derivative(self.z_values[i])
            elif self.activation_functions[i] == 'sigmoid':
                dZ = dA * self.sigmoid_derivative(self.z_values[i])
            
            dW = np.dot(self.activations[i].T, dZ)
            db = np.sum(dZ, axis=0)
            dA = np.dot(dZ, self.weights[i].T)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
        
        return dA  # Return the gradient of the loss with respect to the input

# Define a Convolutional Layer
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        """
        Initialize the convolutional layer.
        
        Args:
            num_filters (int): Number of filters.
            filter_size (int): Size of each filter (filter_size x filter_size).
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Initialize filters with random values
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size ** 2)

    def iterate_regions(self, image):
        """
        Helper function to iterate over regions of the input image.
        
        Args:
            image (np.array): Input image of shape (height, width).
        
        Yields:
            im_region (np.array): Region of the image.
            i (int): Row index.
            j (int): Column index.
        """
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        """
        Perform forward propagation for the convolutional layer.
        
        Args:
            input (np.array): Input image of shape (height, width).
        
        Returns:
            np.array: Output feature maps of shape (height - filter_size + 1, width - filter_size + 1, num_filters).
        """
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learn_rate):
        """
        Perform backward propagation for the convolutional layer.
        
        Args:
            d_L_d_out (np.array): Gradient of the loss with respect to the output.
            learn_rate (float): Learning rate for gradient descent.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        self.filters -= learn_rate * d_L_d_filters
        return None  # No need to return anything for this simple example

# Define a Max Pooling Layer
class MaxPooling:
    def __init__(self, pool_size):
        """
        Initialize the max pooling layer.
        
        Args:
            pool_size (int): Size of the pooling window (pool_size x pool_size).
        """
        self.pool_size = pool_size

    def iterate_regions(self, image):
        """
        Helper function to iterate over regions of the input image.
        
        Args:
            image (np.array): Input image of shape (height, width, num_filters).
        
        Yields:
            im_region (np.array): Region of the image.
            i (int): Row index.
            j (int): Column index.
        """
        h, w, _ = image.shape
        for i in range(h // self.pool_size):
            for j in range(w // self.pool_size):
                im_region = image[
                    (i * self.pool_size):(i * self.pool_size + self.pool_size),
                    (j * self.pool_size):(j * self.pool_size + self.pool_size)
                ]
                yield im_region, i, j

    def forward(self, input):
        """
        Perform forward propagation for the max pooling layer.
        
        Args:
            input (np.array): Input feature maps of shape (height, width, num_filters).
        
        Returns:
            np.array: Output feature maps of shape (height // pool_size, width // pool_size, num_filters).
        """
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.max(im_region, axis=(0, 1))
        return output

    def backward(self, d_L_d_out):
        """
        Perform backward propagation for the max pooling layer.
        
        Args:
            d_L_d_out (np.array): Gradient of the loss with respect to the output.
        
        Returns:
            np.array: Gradient of the loss with respect to the input.
        """
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.max(im_region, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * self.pool_size + i2, j * self.pool_size + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input

# Define the CNN class
class CNN:
    def __init__(self, num_filters, filter_size, pool_size, input_shape, num_classes, activation='relu'):
        """
        Initialize the CNN.
        
        Args:
            num_filters (int): Number of filters in the convolutional layer.
            filter_size (int): Size of each filter (filter_size x filter_size).
            pool_size (int): Size of the pooling window (pool_size x pool_size).
            input_shape (tuple): Shape of the input image (height, width).
            num_classes (int): Number of output classes.
            activation (str): Activation function to use ('relu' or 'sigmoid').
        """
        self.conv = ConvLayer(num_filters, filter_size)
        self.pool = MaxPooling(pool_size)
        # Calculate the size of the flattened output after convolution and pooling
        h = (input_shape[0] - filter_size + 1) // pool_size
        w = (input_shape[1] - filter_size + 1) // pool_size
        self.mlp = MLP([h * w * num_filters, num_classes], [activation])
        self.pool_output_shape = (h, w, num_filters)  # Store the shape of the pooling output

    def forward(self, input):
        """
        Perform forward propagation for the CNN.
        
        Args:
            input (np.array): Input image of shape (height, width).
        
        Returns:
            np.array: Output of the CNN.
        """
        conv_out = self.conv.forward(input)
        print(f"Convolutional layer output shape: {conv_out.shape}")
        pool_out = self.pool.forward(conv_out)
        print(f"Pooling layer output shape: {pool_out.shape}")
        # Flatten the output for the fully connected layer
        flattened = pool_out.flatten().reshape(1, -1)  # Reshape to (1, num_features)
        print(f"Flattened output shape: {flattened.shape}")
        return self.mlp.forward(flattened)

    def backward(self, d_L_d_out, learn_rate):
        """
        Perform backward propagation for the CNN.
        
        Args:
            d_L_d_out (np.array): Gradient of the loss with respect to the output.
            learn_rate (float): Learning rate for gradient descent.
        """
        # Backprop through the MLP
        d_L_d_mlp = self.mlp.backward(d_L_d_out, learn_rate)
        print(f"Gradient from MLP shape: {d_L_d_mlp.shape}")
        # Reshape the gradient to match the pooling output
        d_L_d_pool = d_L_d_mlp.reshape(self.pool_output_shape)
        print(f"Reshaped gradient for pooling layer shape: {d_L_d_pool.shape}")
        # Backprop through the pooling layer
        d_L_d_conv = self.pool.backward(d_L_d_pool)
        print(f"Gradient from pooling layer shape: {d_L_d_conv.shape}")
        # Backprop through the convolutional layer
        self.conv.backward(d_L_d_conv, learn_rate)

# Example usage:
cnn = CNN(num_filters=8, filter_size=3, pool_size=2, input_shape=(28, 28), num_classes=10, activation='relu')
input_image = np.random.randn(28, 28)
print(f"Input image shape: {input_image.shape}")
output = cnn.forward(input_image)
print(f"CNN output shape: {output.shape}")
cnn.backward(np.random.randn(1, 10), learn_rate=0.005)