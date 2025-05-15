import numpy as np

class ScratchLSTM:
    def __init__(self, W, U, b):
        self.W, self.U, self.b = W, U, b
        self.units = U.shape[0]

    def forward(self, x, return_sequences=False):
        B, T, _ = x.shape
        H = np.zeros((B, self.units), np.float32)
        C = np.zeros((B, self.units), np.float32)
        outputs = []
        for t in range(T):
            xt = x[:, t, :]
            z = xt @ self.W + H @ self.U + self.b
            i, f, g, o = np.split(z, 4, axis=1)
            i = 1 / (1 + np.exp(-i))
            f = 1 / (1 + np.exp(-f))
            g = np.tanh(g)
            o = 1 / (1 + np.exp(-o))
            C = f * C + i * g
            H = o * np.tanh(C)
            if return_sequences:
                outputs.append(H[:, None, :])
        return np.concatenate(outputs, axis=1) if return_sequences else H