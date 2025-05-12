import torch as tc
import numpy as np
from models.nn.activations import activation_functions

class SimpleRNNLayer:
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, activation='tanh'):
        self.Wx = tc.tensor(Wx, dtype=tc.float32)
        self.Wh = tc.tensor(Wh, dtype=tc.float32)
        self.b = tc.tensor(b, dtype=tc.float32)
        self.activation, _, _ = activation_functions[activation]

    def forward(self, x: tc.Tensor):
        batch_size, seq_len, _ = x.shape
        h = tc.zeros(batch_size, self.b.shape[0])
        for t in range(seq_len):
            h = self.activation(x[:, t, :] @ self.Wx + h @ self.Wh + self.b)
        return h  # (batch, hidden)