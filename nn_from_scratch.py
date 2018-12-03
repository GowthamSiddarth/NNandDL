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
