import numpy as np


def sigmoid(z): return 1 / (1 + np.exp(-z))


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
        pass

    def evaluate(self, test_data):
        pass

    def SGD(self, train_data, learning_rate, epochs, mini_batch_size, test_data=None):
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
