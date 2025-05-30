import numpy as np
from models.cnn.tensor_activation import activation_functions_np

class ScratchConv2D:
    def __init__(self, weights, biases, padding='valid', strides=(1, 1), activation=None):
        self.weights = weights
        self.biases = biases
        self.padding = padding
        self.strides = strides
        self.activation = activation

    def forward(self, x):
        n_samples, height, width, n_channels = x.shape
        filter_height, filter_width, _, n_filters = self.weights.shape
        if self.padding == 'valid':
            out_height = (height - filter_height) // self.strides[0] + 1
            out_width = (width - filter_width) // self.strides[1] + 1
            padded_x = x
        elif self.padding == 'same':
            out_height = int(np.ceil(height / self.strides[0]))
            out_width = int(np.ceil(width / self.strides[1]))
            pad_h = max((out_height - 1) * self.strides[0] + filter_height - height, 0)
            pad_w = max((out_width - 1) * self.strides[1] + filter_width - width, 0)
            pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
            pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
            padded_x = np.pad(x, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')

        output = np.zeros((n_samples, out_height, out_width, n_filters))
        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end = i * self.strides[0], i * self.strides[0] + filter_height
                w_start, w_end = j * self.strides[1], j * self.strides[1] + filter_width
                patch = padded_x[:, h_start:h_end, w_start:w_end, :]
                for k in range(n_filters):
                    output[:, i, j, k] = np.sum(
                        patch * self.weights[..., k][None, :, :, :], axis=(1, 2, 3)
                    ) + self.biases[k]

        if self.activation in activation_functions_np:
            func = activation_functions_np[self.activation]
            output = func(output)
        return output
