import numpy as np


class ScratchMaxPooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def forward(self, x):
        n, h, w, c = x.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        out = np.zeros((n, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i * sh, i * sh + ph
                w_start, w_end = j * sw, j * sw + pw
                window = x[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(window, axis=(1, 2))
        return out