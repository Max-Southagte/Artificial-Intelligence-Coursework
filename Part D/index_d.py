import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import PolynomialFeatures

class Full_NN(object):
    def __init__(self, X, HL, Y, lambda_l2=0.01):
        self.X = X
        self.HL = HL
        self.Y = Y
        self.lambda_l2 = lambda_l2  # Regularization parameter
        L = [X] + HL + [Y]

        W = []
        for i in range(len(L) - 1):
            w = np.random.randn(L[i], L[i+1]) * 0.1  # Larger random values
            W.append(w)
        self.W = W

        B = []
        for i in range(len(L) - 1):
            b = np.zeros(L[i+1])
            B.append(b)
        self.B = B

        Der = []
        for i in range(len(L) - 1):
            d = np.zeros((L[i], L[i+1]))
            Der.append(d)
        self.Der = Der

        out = []
        for i in range(len(L)):
            o = np.zeros(L[i])
            out.append(o)
        self.out = out

    def FF(self, x):
        out = np.array(x, dtype=float)
        self.out[0] = out
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            Xnext = np.dot(out, w) + b
            out = self.relu(Xnext) if i < len(self.W) - 1 else self.relu(Xnext)  # ReLU for the output layer
            self.out[i+1] = out
        return out

    def BP(self, Er):
        for i in reversed(range(len(self.Der))):
            out = self.out[i+1]
            D = Er * self.relu_Der(out)
            D_fixed = D.reshape(D.shape[0], -1).T
            this_out = self.out[i]
            this_out = this_out.reshape(this_out.shape[0], -1)
            self.Der[i] = np.dot(this_out, D_fixed)
            Er = np.dot(D, self.W[i].T)

    def train_nn(self, x, target, epochs, lr, batch_size):
        num_samples = x.shape[0]
        for i in range(epochs):  # Increased epochs
            print(f"Epoch {i+1}/{epochs}")
            S_errors = 0
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                x_batch = x[batch_indices]
                y_batch = target[batch_indices]
                batch_errors = 0
                for j, input in enumerate(x_batch):
                    t = y_batch[j]
                    output = self.FF(input)
                    e = t - output
                    self.BP(e)
                    S_errors += self.msqe(t, output)
                    batch_errors += self.msqe(t, output)
                self.GD(lr)
                print(f"Batch {start_idx // batch_size + 1}, Error: {batch_errors / batch_size}")
            print(f"Total Error after epoch {i+1}: {S_errors / num_samples}")

    def GD(self, lr=0.01):  # Decreased learning rate
        for i in range(len(self.W)):
            W = self.W[i]
            Der = self.Der[i]
            B = self.B[i]
            W += Der * lr - self.lambda_l2 * W  # L2 regularization
            B += np.mean(Der, axis=0) * lr

    def relu(self, x):
        return np.maximum(0, x)

    def relu_Der(self, x):
        return np.where(x > 0, 1, 0)

    def msqe(self, t, output):
        t = np.array(t, dtype=float)
        output = np.array(output, dtype=float)
        msq = np.average((t - output) ** 2)
        return msq

def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def predict_price(nn, new_diamond_features):
    predicted_price = nn.FF(new_diamond_features[0])
    return predicted_price[0]

def on_predict():
    new_diamond = {
        'carat': float(entry_carat.get()),
        'cut': entry_cut.get(),
        'color': entry_color.get(),
        'clarity': entry_clarity.get(),
        'depth': float(entry_depth.get()),
        'table': float(entry_table.get()),
        'x': float(entry_x.get()),
        'y': float(entry_y.get()),
        'z': float(entry_z.get())
    }
    new_diamond_df = pd.DataFrame([new_diamond])
    new_diamond_encoded = pd.get_dummies(new_diamond_df, columns=['cut', 'color', 'clarity'])
    numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
    for col in numerical_columns:
        new_diamond_encoded[col] = (new_diamond_encoded[col] - data[col].mean()) / data[col].std()
    for col in data.drop(columns=[target]).columns:
        if col not in new_diamond_encoded.columns:
            new_diamond_encoded[col] = 0
    new_diamond_encoded = new_diamond_encoded.reindex(columns=data.drop(columns=[target]).columns, fill_value=0)
    
    # Apply polynomial features transformation
    poly_features = poly.transform(new_diamond_encoded)
    predicted_price = predict_price(nn, poly_features)
    predicted_price_original_scale = predicted_price * target_std + target_mean  # Transform back to original scale
    messagebox.showinfo("Predicted Price", f"The predicted price for the new diamond is: ${predicted_price_original_scale:.2f}")

if __name__ == "__main__":
    print("Reading data...")
    data = pd.read_csv('diamonds.csv')
    features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    target = 'price'
    print("Preprocessing data...")
    data = pd.get_dummies(data, columns=['cut', 'color', 'clarity'])
    data[['carat', 'depth', 'table', 'x', 'y', 'z']] = data[['carat', 'depth', 'table', 'x', 'y', 'z']].apply(lambda x: (x - x.mean()) / x.std())
    X = data.drop(columns=[target]).values
    y = data[target].values

    target_mean = y.mean()
    target_std = y.std()
    y = (y - target_mean) / target_std

    # Apply polynomial features transformation
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)
    print("Training the neural network...")
    nn = Full_NN(X_train.shape[1], [30, 30], 1)  # Increased number of neurons
    nn.train_nn(X_train, y_train, epochs=50, lr=0.01, batch_size=64)  # Increased epochs
    print("Evaluating the neural network...")
    predictions = np.array([nn.FF(x) for x in X_test])
    mse = np.mean((y_test - predictions)**2)
    print(f'Test MSE: {mse}')

    root = tk.Tk()
    root.title("Diamond Price Predictor")

    tk.Label(root, text="Carat").grid(row=0)
    tk.Label(root, text="Cut").grid(row=1)
    tk.Label(root, text="Color").grid(row=2)
    tk.Label(root, text="Clarity").grid(row=3)
    tk.Label(root, text="Depth").grid(row=4)
    tk.Label(root, text="Table").grid(row=5)
    tk.Label(root, text="X").grid(row=6)
    tk.Label(root, text="Y").grid(row=7)
    tk.Label(root, text="Z").grid(row=8)






    entry_carat = tk.Entry(root)
    entry_cut = tk.Entry(root)
    entry_color = tk.Entry(root)
    entry_clarity = tk.Entry(root)
    entry_depth = tk.Entry(root)
    entry_table = tk.Entry(root)
    entry_x = tk.Entry(root)
    entry_y = tk.Entry(root)
    entry_z = tk.Entry(root)

    entry_carat.grid(row=0, column=1)
    entry_cut.grid(row=1, column=1)
    entry_color.grid(row=2, column=1)
    entry_clarity.grid(row=3, column=1)
    entry_depth.grid(row=4, column=1)
    entry_table.grid(row=5, column=1)
    entry_x.grid(row=6, column=1)
    entry_y.grid(row=7, column=1)
    entry_z.grid(row=8, column=1)

    tk.Button(root, text="Predict Price", command=on_predict).grid(row=9, columnspan=2)

    root.mainloop()
