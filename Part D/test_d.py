import pandas as pd
import numpy as np

# Load dataset
diamonds = pd.read_csv("diamonds.csv")

# Preprocessing: Convert categorical variables to numerical
diamonds['cut'] = diamonds['cut'].astype('category').cat.codes
diamonds['color'] = diamonds['color'].astype('category').cat.codes
diamonds['clarity'] = diamonds['clarity'].astype('category').cat.codes

# Separate features and target
X = diamonds[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']].values
y = diamonds['price'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Initialize neural network parameters
np.random.seed(42)
input_size = X.shape[1]
hidden_size = 10
output_size = 1

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.001  # Scale down initialization
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.001 
b2 = np.zeros((1, output_size))

# Activation function (ReLU) and its derivative
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(np.float32)

# Loss function (Mean Squared Error)
def mse_loss(y_true, y_pred):
    y_pred = y_pred.flatten()  # Flatten the predicted values to match y's shape
    return np.mean((y_true - y_pred) ** 2)

# Forward pass
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z1, A1, Z2

# Backward pass
def backward_propagation(X, y, Z1, A1, Z2, W2):
    m = X.shape[0]
    dZ2 = Z2 - y.reshape(-1, 1)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# Training the neural network
learning_rate = 0.01
epochs = 5000

for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2 = forward_propagation(X, W1, b1, W2, b2)

    # Compute loss
    loss = mse_loss(y, Z2)

    # Backward pass
    dW1, db1, dW2, db2 = backward_propagation(X, y, Z1, A1, Z2, W2)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Prediction function
def predict(user_input):
    user_input = np.array(user_input).reshape(1, -1)
    user_input = (user_input - X.mean(axis=0)) / X.std(axis=0)
    _, _, Z2 = forward_propagation(user_input, W1, b1, W2, b2)
    return Z2[0, 0]

# User input
print("Enter the following details about the diamond:")
carat = float(input("Carat: "))
cut = int(input("Cut (0-4): "))
color = int(input("Color (0-6): "))
clarity = int(input("Clarity (0-7): "))
depth = float(input("Depth: "))
table = float(input("Table: "))
x = float(input("x: "))
y = float(input("y: "))
z = float(input("z: "))

user_input = [carat, cut, color, clarity, depth, table, x, y, z]
predicted_price = predict(user_input)
print(f"Predicted price of the diamond: ${predicted_price:.2f}")
