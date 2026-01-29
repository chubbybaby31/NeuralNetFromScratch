import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate, batch_size):
        pass

    def adam_update(self, learning_rate, beta1, beta2, eps, t):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(self.output_size, self.input_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((self.output_size, 1))

        # Adam state
        self.m_W = np.zeros_like(self.weights)
        self.v_W = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)

    def forward(self, input):
        self.input = input
        return self.weights @ self.input + self.biases
        

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[1]

        self.dW = output_gradient @ self.input.T / batch_size
        self.dB = np.sum(output_gradient, axis=1, keepdims=True) / batch_size
        dX = self.weights.T @ output_gradient

        return dX

    def adam_update(self, learning_rate, beta1, beta2, eps, t):
        
        # Update moments for weights
        
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW ** 2)

        # Bias correction
        
        m_W_hat = self.m_W / (1 - beta1**t)
        v_W_hat = self.v_W / (1 - beta2**t)

        # Update weights

        self.weights -= learning_rate * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        # Update Biases
        
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.dB
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.dB ** 2)

        m_b_hat = self.m_b / (1 - beta1**t)
        v_b_hat = self.v_b / (1 - beta2**t)

        self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)
        
class Activation(Layer):
    def __init__(self, activation_function, activation_prime):
        self.activation_function = activation_function
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation_function(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.fmax(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)

class Softmax(Activation):
    def __init__(self):
        softmax_prime = lambda x: 1
        super().__init__(self.softmax, softmax_prime)

    def softmax(self, x):
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, network):
        self.network = network

    def predict(self, data):
        output = data
        for layer in self.network:
            output = layer.forward(output)
        return output

    def score(self, X, y):
        X = X.squeeze(-1).T
        y = y.squeeze(-1).T
        preds = np.argmax(self.predict(X), axis=0)
        labels = np.argmax(y, axis=0)
        return np.sum(preds == labels) / y.shape[1]

    def train(self, X, y, learning_rate=0.1, epoch=16, batch_size=100):
        self.t = 0
        for e in range(epoch):
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
            error = 0
            correct = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size].squeeze(-1).T
                y_batch = y[i:i + batch_size].squeeze(-1).T
                output = self.predict(X_batch)
                loss = self.cross_entropy(y_batch, output)
                error += loss
                preds = np.argmax(output, axis=0)
                labels = np.argmax(y_batch, axis=0)
                correct += np.sum(preds == labels)
                self.t += 1
                gradient = self.cross_entropy_prime(y_batch, output)
                for layer in reversed(self.network):
                    gradient = layer.backward(gradient, learning_rate)
                    layer.adam_update(
                        learning_rate=learning_rate,
                        beta1=0.9,
                        beta2=0.999,
                        eps=1e-8,
                        t=self.t
                    )
            print(f"Epoch: {e + 1}/{epoch} | Cross Entropy Loss: {error / y.shape[0]} | Accuracy: {correct / y.shape[0]}")
        print(f"Model Accuracy on Training Data: {self.score(X, y)}")

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    def cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-9))

    def cross_entropy_prime(self, y_true, y_pred):
        return y_pred - y_true
        
    def one_hot_encode(self, data):
        one_hot = []
        for d in data:
            one_hot.append(np.zeros((10, 1)))
            one_hot[-1][int(d)] = 1
        return np.array(one_hot)