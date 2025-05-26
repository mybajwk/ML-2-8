import numpy as np


class ScratchDense:
    def __init__(self, weights, biases, activation=None):
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def forward(self, x):
        out = np.dot(x, self.weights) + self.biases
        if self.activation == 'relu':
            return np.maximum(0, out)
        elif self.activation == 'softmax':
            exps = np.exp(out - np.max(out, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)
        return out