import numpy as np
from models.cnn.tensor_activation import activation_functions_np

class ScratchDense:
    def __init__(self, weights, biases, activation=None):
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def forward(self, x):
        output = np.dot(x, self.weights) + self.biases
        if self.activation in activation_functions_np:
            output = activation_functions_np[self.activation](output)
        return output