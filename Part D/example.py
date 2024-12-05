import numpy as np
from random import random

class NeuralNetwork(object):

    def __init__(self, num_inputs=2, hidden_layers=[2, 2], num_outputs=2):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        layer_sizes = [num_inputs] + hidden_layers + [num_outputs]
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.rand(layer_sizes[i], layer_sizes[i + 1])
            self.weights.append(weight_matrix)
        self.weights = self.weights

        self.deltas = []
        for i in range(len(layer_sizes) - 1):
            delta_matrix = np.zeros((layer_sizes[i], layer_sizes[i + 1]))
            self.deltas.append(delta_matrix)
        self.deltas = self.deltas

        self.outputs = []
        for i in range(len(layer_sizes)):
            output_array = np.zeros(layer_sizes[i])
            self.outputs.append(output_array)
        self.outputs = self.outputs

    def forward_pass(self, inputs):
        current_output = inputs
        self.outputs[0] = inputs
        for i, weight_matrix in enumerate(self.weights):
            next_output = np.dot(current_output, weight_matrix)
            current_output = self.sigmoid(next_output)
            self.outputs[i + 1] = current_output
        return current_output

    def backpropagation(self, error):
        for i in reversed(range(len(self.deltas))):
            layer_output = self.outputs[i + 1]
            delta = error * self.sigmoid_derivative(layer_output)
            delta = delta.reshape(delta.shape[0], -1).T
            previous_layer_output = self.outputs[i]
            previous_layer_output = previous_layer_output.reshape(previous_layer_output.shape[0], -1)
            self.deltas[i] = np.dot(previous_layer_output, delta)
            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for i, input_data in enumerate(inputs):
                target = targets[i]
                output = self.forward_pass(input_data)
                error = target - output
                self.backpropagation(error)
                self.gradient_descent(learning_rate)
                total_error += self.mean_squared_error(target, output)

    def gradient_descent(self, learning_rate=0.05):
        for i in range(len(self.weights)):
            self.weights[i] += self.deltas[i] * learning_rate

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def mean_squared_error(self, target, output):
        return np.average((target - output) ** 2)

if __name__ == "__main__":
    training_inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] * i[1]] for i in training_inputs])

    nn = NeuralNetwork(2, [5, 5], 1)
    nn.train(training_inputs, targets, 10, 0.1)

    test_input = np.array([0.3, 0.2])
    expected_output = np.array([0.06])
    nn_output = nn.forward_pass(test_input)

    print("=============== Testing the Network Output ===============")
    print("Test input: ", test_input)
    print("Expected output: ", expected_output)
    print("Neural Network actual output: ", nn_output, "Error: ", expected_output - nn_output)
    print("=========================================================")
