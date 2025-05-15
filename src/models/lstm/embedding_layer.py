import numpy as np

class ScratchEmbedding:
    def __init__(self, W: np.ndarray):
        self.W = W  # (vocab, dim)

    def forward(self, x_int):
        return self.W[x_int]