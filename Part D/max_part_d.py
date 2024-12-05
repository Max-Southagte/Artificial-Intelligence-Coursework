import numpy as np
import pandas as pd

# Load the dataset (replace 'diamonds.csv' with your dataset file path)
data = pd.read_csv('diamonds.csv')

# Select features and target
categorical_columns = ['cut', 'color', 'clarity']
numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
target_column = 'price'

# One-hot encode categorical features
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Normalize numerical features
for column in numerical_columns:
    data[column] = (data[column] - data[column].mean()) / data[column].std()

# Separate features and target
features = data.drop(target_column, axis=1).values
target = data[target_column].values

# Normalize the target
target_mean = target.mean()
target_std = target.std()
target = (target - target_mean) / target_std

# Manual train-test split (80-20 split)
split_index = int(0.8 * len(features))
features_train, features_test = features[:split_index], features[split_index:]
target_train, target_test = target[:split_index], target[split_index:]

# Neural Network parameters
input_layer_size = features_train.shape[1]
hidden_layer_size = 10
output_layer_size = 1
learning_rate = 0.01
num_epochs = 5000

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size) * 0.01
bias_hidden = np.zeros((1, hidden_layer_size))
weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size) * 0.01
bias_output = np.zeros((1, output_layer_size))

# Activation function and its derivative
def relu_activation(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    hidden_layer_input = np.dot(features_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu_activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predictions = output_layer_input

    # Compute loss (Mean Squared Error)
    loss = np.mean((predictions - target_train.reshape(-1, 1))**2)

    # Backpropagation
    error_output_layer = predictions - target_train.reshape(-1, 1)
    gradient_weights_hidden_output = np.dot(hidden_layer_output.T, error_output_layer) / features_train.shape[0]
    gradient_bias_output = np.sum(error_output_layer, axis=0, keepdims=True) / features_train.shape[0]
    error_hidden_layer = np.dot(error_output_layer, weights_hidden_output.T)
    gradient_hidden_layer_input = error_hidden_layer * relu_derivative(hidden_layer_input)
    gradient_weights_input_hidden = np.dot(features_train.T, gradient_hidden_layer_input) / features_train.shape[0]
    gradient_bias_hidden = np.sum(gradient_hidden_layer_input, axis=0, keepdims=True) / features_train.shape[0]

    # Update weights and biases
    weights_input_hidden -= learning_rate * gradient_weights_input_hidden
    bias_hidden -= learning_rate * gradient_bias_hidden
    weights_hidden_output -= learning_rate * gradient_weights_hidden_output
    bias_output -= learning_rate * gradient_bias_output

    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

# Denormalize predictions and target for comparison
def predict_price(input_features):
    input_features = np.array(input_features).reshape(1, -1)
    hidden_layer_input = np.dot(input_features, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu_activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_price = output_layer_input[0, 0]
    print(predicted_price * target_std + target_mean)
    return predicted_price * target_std + target_mean

# User input for prediction
def get_user_input():
    # Example input for all features
    print("Enter the following details for the diamond:")
    carat = float(input("Carat: "))
    depth = float(input("Depth: "))
    table = float(input("Table: "))
    x = float(input("X (length): "))
    y = float(input("Y (width): "))
    z = float(input("Z (depth): "))
    cut = input("Cut (Fair, Good, Very Good, Premium, Ideal): ")
    color = input("Color (D, E, F, G, H, I, J): ")
    clarity = input("Clarity (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF): ")
    
    # One-hot encode the categorical inputs
    categorical_data = pd.get_dummies(
        pd.DataFrame({'cut': [cut], 'color': [color], 'clarity': [clarity]}),
        columns=['cut', 'color', 'clarity'],
        drop_first=True
    )
    
    # Align the columns with training data
    categorical_data = categorical_data.reindex(columns=data.columns[len(numerical_columns):-1], fill_value=0)
    
    # Normalize numerical inputs
    numerical_data = np.array([carat, depth, table, x, y, z])
    for i, column in enumerate(numerical_columns):
        numerical_data[i] = (numerical_data[i] - data[column].mean()) / data[column].std()
    
    # Combine numerical and categorical inputs
    input_data = np.hstack((numerical_data, categorical_data.values.flatten()))
    return input_data

# Predict and print the result
user_input = get_user_input()
predicted_price = predict_price(user_input)
print(f"The predicted price of the diamond is: ${predicted_price:.2f}")
