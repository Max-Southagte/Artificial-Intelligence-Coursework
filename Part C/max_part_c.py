import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x_values = np.linspace(-10, 10, 100)
y_actual = 3 * x_values + 0.7 * x_values**2

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros(output_size)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def forward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        return self.output_layer_input

    def backward(self, inputs, targets, predictions):
        error = predictions - targets
        grad_weights_hidden_output = np.dot(self.hidden_layer_output.T, error) / len(inputs)
        grad_bias_output = np.mean(error, axis=0)
        grad_hidden_layer_output = np.dot(error, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_layer_output)
        grad_weights_input_hidden = np.dot(inputs.T, grad_hidden_layer_output) / len(inputs)
        grad_bias_hidden = np.mean(grad_hidden_layer_output, axis=0)

        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.bias_output -= self.learning_rate * grad_bias_output
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden
        self.bias_hidden -= self.learning_rate * grad_bias_hidden

    def train(self, inputs, targets, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            predictions = self.forward(inputs)
            loss = np.mean((predictions - targets) ** 2)
            loss_history.append(loss)
            self.backward(inputs, targets, predictions)
        return loss_history

x_train = x_values.reshape(-1, 1)
y_train = y_actual.reshape(-1, 1)

nn = SimpleNeuralNetwork(input_size=1, hidden_size=40, output_size=1, learning_rate=0.1)
loss_history = nn.train(x_train, y_train, epochs=10000)

y_predictions = nn.forward(x_train)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_values, y_actual, color="blue", label="True Function", linestyle='dashed')
plt.plot(x_values, y_predictions.flatten(), color="red", label="Neural Network Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function Approximation")
plt.legend()

plt.tight_layout()
plt.show()
