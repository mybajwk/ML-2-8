import torch as tc
import numpy as np

class EmbeddingLayer:
    def __init__(self, weights: np.ndarray):
        self.weights = tc.tensor(weights, dtype=tc.float32)

    def forward(self, x: tc.Tensor):
        return self.weights[x] 