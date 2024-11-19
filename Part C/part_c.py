import numpy as np
import matplotlib.pyplot as plt

# Function definition
def target_function(x):
    return 3 * x + 0.7 * x**2

# Generate training data
np.random.seed(42)
x_train = np.linspace(-10, 10, 200).reshape(-1, 1)
y_train = target_function(x_train).ravel() + np.random.normal(scale=5, size=x_train.shape[0])

# Neural network model using forward pass
def initialize_parameters(input_size, hidden_layer_sizes, output_size):
    parameters = []
    layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
    
    for i in range(len(layer_sizes) - 1):
        weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
        bias = np.zeros((1, layer_sizes[i + 1]))
        parameters.append((weight, bias))
    
    return parameters

# Forward pass
def forward_pass(x, parameters):
    activations = [x]
    for weight, bias in parameters:
        x = np.dot(x, weight) + bias
        x = np.maximum(0, x) 
        activations.append(x)
    return activations

# Backward pass
def backward_pass(activations, parameters, y_true, learning_rate):
    gradients = []
    y_pred = activations[-1]
    error = y_pred - y_true.reshape(-1, 1)
    
    delta = error * (y_pred > 0)
    for i in reversed(range(len(parameters))):
        weight, bias = parameters[i]
        grad_weight = np.dot(activations[i].T, delta)
        grad_bias = np.sum(delta, axis=0, keepdims=True)
        gradients.append((grad_weight, grad_bias))
        
        if i > 0:
            delta = np.dot(delta, weight.T) * (activations[i] > 0)
    
    gradients = gradients[::-1]
    
    # Update parameters
    for i in range(len(parameters)):
        weight, bias = parameters[i]
        grad_weight, grad_bias = gradients[i]
        parameters[i] = (weight - learning_rate * grad_weight, bias - learning_rate * grad_bias)
    
    return parameters

# Training loop
def train_neural_network(x_train, y_train, hidden_layer_sizes=(10, 10), epochs=5000, learning_rate=0.01):
    input_size = x_train.shape[1]
    output_size = 1
    parameters = initialize_parameters(input_size, hidden_layer_sizes, output_size)
    
    for epoch in range(epochs):
        activations = forward_pass(x_train, parameters)
        parameters = backward_pass(activations, parameters, y_train, learning_rate)
        
        if epoch % 100 == 0:
            loss = np.mean((activations[-1] - y_train.reshape(-1, 1)) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return parameters

# Plotting function
def plot_results(parameters):
    x_test = np.linspace(-10, 10, 200).reshape(-1, 1)
    activations = forward_pass(x_test, parameters)
    y_pred = activations[-1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='gray', alpha=0.5, label='Training Data')
    plt.plot(x_test, target_function(x_test), color='blue', label='Target Function')
    plt.plot(x_test, y_pred, color='red', linestyle='--', label='Neural Network Prediction')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Neural Network Approximation of y=3x+0.7x^2')
    plt.show()

# Parameters
hidden_layer_sizes = (20, 10)  # Example architecture
epochs = 5000
learning_rate = 0.5

# Train and plot
parameters = train_neural_network(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes, epochs=epochs, learning_rate=learning_rate)
plot_results(parameters)

# Allow for change in parameters
def interactive_training(hidden_layer_sizes=(10, 10), epochs=1000, learning_rate=0.01):
    parameters = train_neural_network(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes, epochs=epochs, learning_rate=learning_rate)
    plot_results(parameters)
