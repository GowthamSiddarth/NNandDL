import numpy as np


class Network(object):
    def __init__(self, layers):
        self.num_of_layers = len(layers)
        self.layers = layers
        self.bias = [np.random.randn(nodes, 1) for nodes in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[1:], layers[:-1])]
