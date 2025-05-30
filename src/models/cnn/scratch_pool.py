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
    
class ScratchAveragePooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def forward(self, x):
        n_samples, height, width, n_channels = x.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides
        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1
        output = np.zeros((n_samples, out_height, out_width, n_channels))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + pool_height
                w_start = j * stride_width
                w_end = w_start + pool_width
                window = x[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.mean(window, axis=(1, 2))
        return output
    
class ScratchGlobalAveragePooling2D:
    def forward(self, x):
        return np.mean(x, axis=(1, 2))


class ScratchGlobalMaxPooling2D:
    def forward(self, x):
        return np.max(x, axis=(1, 2))