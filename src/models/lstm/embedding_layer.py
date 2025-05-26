import numpy as np

class ScratchEmbedding:
    def __init__(self, W: np.ndarray):
        self.W = W  # shape (vocab, dim)

    def forward(self, x_int):
        self.x_int = x_int
        return self.W[x_int]

    def backward(self, dL_dout):
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.x_int, dL_dout)
        return None  # No gradient w.r.t. input ids