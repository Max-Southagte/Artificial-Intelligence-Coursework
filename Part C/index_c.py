import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Generate input data (x-values) and corresponding true function output (y-values)
x_values = np.linspace(-10, 10, 100)
y_actual = 3 * x_values + 0.7 * x_values**2

# Define a simple neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases for input to hidden layer
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros(hidden_size)
        # Initialize weights and biases for hidden to output layer
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros(output_size)
        self.learning_rate = learning_rate

    # Define the sigmoid activation function
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # Define the derivative of the sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    # Perform a forward pass through the network
    def forward(self, inputs):
        # Compute the input and output of the hidden layer
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        # Compute the output of the network
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        return self.output_layer_input

    # Perform a backward pass to compute gradients and update weights
    def backward(self, inputs, targets, predictions):
        # Compute the error between predictions and targets
        error = predictions - targets

        # Compute gradients for weights and biases in the hidden-to-output layer
        grad_weights_hidden_output = np.dot(self.hidden_layer_output.T, error) / len(inputs)
        grad_bias_output = np.mean(error, axis=0)

        # Compute gradients for weights and biases in the input-to-hidden layer
        grad_hidden_layer_output = np.dot(error, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_layer_output)
        grad_weights_input_hidden = np.dot(inputs.T, grad_hidden_layer_output) / len(inputs)
        grad_bias_hidden = np.mean(grad_hidden_layer_output, axis=0)

        # Update weights and biases using gradient descent
        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.bias_output -= self.learning_rate * grad_bias_output
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden
        self.bias_hidden -= self.learning_rate * grad_bias_hidden

    # Train the network for a given number of epochs
    def train(self, inputs, targets, epochs):
        loss_history = []
        predictions_history = []
        for epoch in range(epochs):
            # Perform forward pass
            predictions = self.forward(inputs)
            # Store predictions for this epoch
            predictions_history.append(predictions.copy())
            # Compute mean squared error loss
            loss = np.mean((predictions - targets) ** 2)
            loss_history.append(loss)
            # Perform backward pass
            self.backward(inputs, targets, predictions)
        return loss_history, predictions_history

# Reshape the data to match the input dimensions expected by the network
x_train = x_values.reshape(-1, 1)
y_train = y_actual.reshape(-1, 1)

# Create a neural network instance with 1 input node, 40 hidden nodes, and 1 output node
nn = NeuralNetwork(input_size=1, hidden_size=40, output_size=1, learning_rate=0.1)

# Train the neural network and record the loss history and predictions
loss_history, predictions_history = nn.train(x_train, y_train, 5000)

# Use the trained network to make predictions
y_predictions = nn.forward(x_train)

# Plot the training loss and the function approximation
plt.figure(figsize=(12, 6))

# Plot the training loss over epochs
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.legend()

# Plot the true function and neural network predictions at different epochs
plt.subplot(1, 2, 2)
plt.plot(x_values, y_actual, color="blue", label="True Function", linestyle='dashed')

# Select specific epochs to plot (e.g., start, middle, end)
epochs_to_plot = [0, 1000, 2000, 3000, 4000, 4999]
colors = ["orange", "green", "purple", "yellow", "black", "red"]
for i, epoch in enumerate(epochs_to_plot):
    plt.plot(x_values, predictions_history[epoch].flatten(), color=colors[i], label=f"Epoch {epoch+1}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Function Approximation over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
