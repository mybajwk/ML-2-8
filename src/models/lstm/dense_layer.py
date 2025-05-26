import numpy as np

class ScratchDense:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.last_input = x
        z = x @ self.W + self.b
        e = np.exp(z - z.max(axis=1, keepdims=True))
        self.out = e / e.sum(axis=1, keepdims=True)
        return self.out

    def backward(self, dL_dout):
        x = self.last_input
        dL_dz = self.out * (dL_dout - (dL_dout * self.out).sum(axis=1, keepdims=True))
        self.dW = x.T @ dL_dz
        self.db = dL_dz.sum(axis=0)
        dL_dx = dL_dz @ self.W.T
        return dL_dx