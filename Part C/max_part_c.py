import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(-10, 10, 100)
y_true = 3 * x + 0.7 * x**2

class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.learning_rate = learning_rate

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2

    def backward(self, x, y, y_pred):
        error = y_pred - y
        d_w2 = np.dot(self.a1.T, error) / len(x)
        d_b2 = np.mean(error, axis=0)
        d_a1 = np.dot(error, self.w2.T) * (1 - np.tanh(self.z1)**2)
        d_w1 = np.dot(x.T, d_a1) / len(x)
        d_b1 = np.mean(d_a1, axis=0)

        self.w2 -= self.learning_rate * d_w2
        self.b2 -= self.learning_rate * d_b2
        self.w1 -= self.learning_rate * d_w1
        self.b1 -= self.learning_rate * d_b1

    def train(self, x, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            self.backward(x, y, y_pred)
        return losses

x_train = x.reshape(-1, 1)
y_train = y_true.reshape(-1, 1)

nn = SimpleNeuralNetwork(input_dim=1, hidden_dim=10, output_dim=1, learning_rate=0.01)
losses = nn.train(x_train, y_train, epochs=5000)

y_pred = nn.forward(x_train)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y_true, color="blue", label="True Function", linestyle='dashed')
plt.plot(x, y_pred.flatten(), color="red", label="Neural Network Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function Approximation")
plt.legend()

plt.tight_layout()
plt.show()
