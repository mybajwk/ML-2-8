import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from models.cnn.scratch_convolution import ScratchConv2D
from models.cnn.scratch_dense import ScratchDense
from models.cnn.scratch_flatten import ScratchFlatten
from models.cnn.scratch_pool import ScratchMaxPooling2D, ScratchAveragePooling2D, ScratchGlobalAveragePooling2D, ScratchGlobalMaxPooling2D
from models.cnn.scratch_dropout import ScratchDropout
class ScratchModel:
    def __init__(self, keras_model_path):
        keras_model = load_model(keras_model_path)
        self.layers = []
        for layer in keras_model.layers:
            if isinstance(layer, Conv2D):
                w, b = layer.get_weights()
                self.layers.append(ScratchConv2D(w, b, layer.padding, layer.strides, layer.activation.__name__))
            elif isinstance(layer, MaxPooling2D):
                self.layers.append(ScratchMaxPooling2D(layer.pool_size, layer.strides))
            elif isinstance(layer, AveragePooling2D):
                self.layers.append(ScratchAveragePooling2D(layer.pool_size, layer.strides))

            elif isinstance(layer, GlobalAveragePooling2D):
                self.layers.append(ScratchGlobalAveragePooling2D())

            elif isinstance(layer, GlobalMaxPooling2D):
                self.layers.append(ScratchGlobalMaxPooling2D())
            elif isinstance(layer, Flatten):
                self.layers.append(ScratchFlatten())
            elif isinstance(layer, Dense):
                w, b = layer.get_weights()
                self.layers.append(ScratchDense(w, b, layer.activation.__name__))
            elif isinstance(layer, Dropout):
                self.layers.append(ScratchDropout())

    def predict(self, x, batch_size=None):
        if batch_size is None:
            return self._predict_batch(x)
        else:
            preds = []
            for start in range(0, len(x), batch_size):
                print(f"Processing batch from index {start}")
                end = start + batch_size
                batch = x[start:end]
                batch_pred = self._predict_batch(batch)
                preds.append(batch_pred)
            return np.concatenate(preds, axis=0)
    
    def _predict_batch(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out