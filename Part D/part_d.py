# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("diamonds.csv")

# Display basic information about the dataset
print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 1: Drop unnecessary columns (optional)
# You can exclude up to 3 features as per the instructions. Let's drop x, y, and z.
df = df.drop(columns=["x", "y", "z"])

# Step 2: Encode categorical features
# Convert 'cut', 'color', and 'clarity' into numeric values using mapping
cut_mapping = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
color_mapping = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_mapping = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

df["cut"] = df["cut"].map(cut_mapping)
df["color"] = df["color"].map(color_mapping)
df["clarity"] = df["clarity"].map(clarity_mapping)

# Step 3: Separate features (X) and target variable (y)
X = df.drop("price", axis=1).values  # Convert features to a NumPy array
y = df["price"].values  # Target variable as a NumPy array

# Step 4: Normalize the features
# Scale each feature to a 0–1 range
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)

# Step 5: Split the data into training and testing sets (80% train, 20% test)
split_index = int(0.8 * len(X_normalized))
X_train, X_test = X_normalized[:split_index], X_normalized[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 6: Build the Neural Network
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),  # First hidden layer
    Dense(32, activation="relu"),  # Second hidden layer
    Dense(16, activation="relu"),  # Third hidden layer
    Dense(1, activation="linear")  # Output layer for regression (price prediction)
])

# Step 7: Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # Use MSE loss and MAE metric

# Step 8: Train the model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=50, 
    batch_size=32, 
    verbose=1
)

# Step 9: Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Evaluation:")
print(f"Mean Absolute Error (MAE) on test data: {mae}")
print(f"Mean Squared Error (MSE) on test data: {loss}")

# Step 10: Visualize training history
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

# Step 11: Predict on test data
y_pred = model.predict(X_test)

# Scatter plot to compare actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()

# Step 12: Save the trained model
model.save("diamond_price_predictor.h5")
print("\nModel saved as 'diamond_price_predictor.h5'.")

# Step 13: Example prediction (optional)
example_input = X_test[0].reshape(1, -1)  # Reshape for a single input
predicted_price = model.predict(example_input)
print(f"\nPredicted price for example input: ${predicted_price[0][0]:.2f}")
