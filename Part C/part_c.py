import numpy as np
import matplotlib.pyplot as plt

class NN:
    # The constructor: X = input neurons count, HL = hidden layers, Y = output neurons
    def __init__(self, X=1, HL=[10], Y=1):
        self.X = X  
        self.HL = HL  
        self.Y = Y  
        L = [X] + HL + [Y]
        
        # weights for the layers (i to i+1)
        W = []  
        for i in range(len(L) - 1):
            w = np.random.rand(L[i], L[i + 1])
            W.append(w)
        self.W = W
        
        # back propagation derivatives (inspired by session 5 code)
        Der = []  
        for i in range(len(L) - 1):
            d = np.zeros((L[i], L[i + 1]))
            Der.append(d)
        self.Der = Der 
        
        # outputs
        out = []  
        for i in range(len(L)):
            o = np.zeros(L[i])  
            out.append(o)
        self.out = out 

    # Feedforward
    def FF(self, x):
        out = x  
        self.out[0] = x  
        for i, w in enumerate(self.W):  
            Xnext = np.dot(out, w) 
            out = self.sigmoid(Xnext)  
            self.out[i + 1] = out  
        return out  

    # Back propagation
    def BP(self, Er):
        for i in reversed(range(len(self.Der))):
            out = self.out[i + 1]
            D = Er * self.sigmoid_Der(out) 
            D_fixed = D.reshape(D.shape[0], -1).T
            this_out = self.out[i].reshape(self.out[i].shape[0], -1)
            self.Der[i] = np.dot(this_out, D_fixed)  
            Er = np.dot(D, self.W[i].T)  

    # Training
    def train_nn(self, x, target, epochs, lr):
        for i in range(epochs):
            S_errors = 0  
            for j, input in enumerate(x):
                t = target[j]
                output = self.FF(input)
                e = t - output  
                self.BP(e)
                self.GD(lr)  
                S_errors += self.msqe(t, output) 
        
    # Gradient descent
    def GD(self, lr=0.05):
        for i in range(len(self.W)):
            self.W[i] += self.Der[i] * lr  

    # Sigmoid activation
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    def sigmoid_Der(self, x):
        return x * (1.0 - x)

    # Mean square error
    def msqe(self, t, output):
        return np.average((t - output) ** 2)

if __name__ == "__main__":
    # Generate data
    x_values = np.linspace(-10, 10, 1000).reshape(-1, 1) 
    y_values = 3 * x_values + 0.7 * x_values ** 2

    nn = NN(1, [10], 1)
    nn.train_nn(x_values, y_values, epochs=1000, lr=0.01)

    # Testing stuff here (inspired by the testing format in session 5 as it looked cool)
    test_input = np.array([5]) 
    test_target = 3 * test_input + 0.7 * test_input ** 2
    NN_output = nn.FF(test_input)
    print("=================Test Test Test====================")
    print("Test input is ", test_input)
    print("Target output is ", test_target)
    print("Neural Network actual output is ", NN_output, " there is an error (not MSQE) of ", test_target - NN_output)
    print("===================================================")

    # Visualize it baby (used matplotlib) This may take a lil min
    predicted_values = np.array([nn.FF(x) for x in x_values])

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Actual Function (y = 3x + 0.7x^2)', color='b')
    plt.plot(x_values, predicted_values, label='Neural Network Approximation', color='r', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Neural Network Approximation of y = 3x + 0.7x^2')
    plt.legend()
    plt.grid()
    plt.show()
