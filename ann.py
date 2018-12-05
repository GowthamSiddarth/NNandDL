import numpy as np


def sigmoid(z): return 1 / (1 + np.exp(-z))


def sigmoid_prime(z): return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, layers):
        self.num_of_layers = len(layers)
        self.layers = layers
        self.bias = [np.random.randn(nodes, 1) for nodes in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[1:], layers[:-1])]

    def feed_forward(self, x):
        """Return output of the nn, given the input"""
        for bias, weights in zip(self.bias, self.weights):
            x = sigmoid(np.dot(weights, x) + bias)

        return x

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b, nabla_w = [np.zeros(b.shape) for b in self.bias], [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.bias, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def back_propagation(self, x, y):
        nabla_b, nabla_w = [np.zeros(b.shape) for b in self.bias], [np.zeros(w.shape) for w in self.weights]
        activations, zs, activation = [x], [], x

        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        delta = Network.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_of_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def stochastic_gradient_descent(self, train_data, learning_rate, epochs, mini_batch_size, test_data=None):
        if test_data:
            test_data_len = len(test_data)

        train_data_len = len(train_data)

        for epoch in range(epochs):
            mini_batches = [train_data[k: k + mini_batch_size] for k in range(0, train_data_len, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), test_data_len))
            else:
                print("Epoch {0} complete".format(epoch))

    @staticmethod
    def cost_derivative(a, y):
        return a - y
