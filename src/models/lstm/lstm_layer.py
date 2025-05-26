import numpy as np

class ScratchLSTM:
    def __init__(self, W, U, b):
        self.W, self.U, self.b = W, U, b
        self.units = U.shape[0]

    def forward(self, x, return_sequences=False):
        self.x = x
        B, T, _ = x.shape
        H = np.zeros((B, self.units), np.float32)
        C = np.zeros((B, self.units), np.float32)
        self.cache = []

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
            self.cache.append((xt, H.copy(), C.copy(), i, f, g, o))
            if return_sequences:
                outputs.append(H[:, None, :])
        self.return_sequences = return_sequences
        return np.concatenate(outputs, axis=1) if return_sequences else H

    def backward(self, dL_dH):
        B, T, D = self.x.shape
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        dX_all = np.zeros_like(self.x)
        dH_next = np.zeros((B, self.units))
        dC_next = np.zeros((B, self.units))

        range_T = range(T - 1, -1, -1) if self.return_sequences else [T - 1]

        for t in range_T:
            xt, H, C, i, f, g, o = self.cache[t]
            dH = dL_dH[:, t, :] if self.return_sequences else dL_dH
            dH += dH_next

            tanhC = np.tanh(C)
            dO = dH * tanhC * o * (1 - o)
            dC = dH * o * (1 - tanhC ** 2) + dC_next

            dF = dC * self.cache[t - 1][2] * f * (1 - f) if t > 0 else dC * 0
            dI = dC * g * i * (1 - i)
            dG = dC * i * (1 - g ** 2)

            dz = np.concatenate([dI, dF, dG, dO], axis=1)

            dW += xt.T @ dz
            dU += self.cache[t - 1][1].T @ dz if t > 0 else 0
            db += dz.sum(axis=0)

            dX_step = dz @ self.W.T
            dH_next = dz @ self.U.T
            dC_next = dC * f

            dX_all[:, t, :] = dX_step

        self.dW = dW
        self.dU = dU
        self.db = db
        return dX_all