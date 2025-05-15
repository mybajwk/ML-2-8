import numpy as np

class ScratchDense:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        z = x @ self.W + self.b
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)