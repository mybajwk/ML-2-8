from .simple_rnn_layer import SimpleRNNLayer
from .bidirectional_rnn_layer import BidirectionalRNNLayer
from .embedding_layer import EmbeddingLayer
import tensorflow as tf
from models.nn.ffnn import FFNN
import torch as tc
import numpy as np

class SimpleRNNManual:
    def __init__(self):
        self.embedding = None
        self.rnn_layers = []
        self.dense = None

    def load_from_keras(self, keras_model_path: str, bidirectional=True, rnn_layers=1, dense_activations=None, rnn_activation='tanh'):
        model = tf.keras.models.load_model(keras_model_path)
        weights = model.get_weights()
        idx = 0

        self.embedding = EmbeddingLayer(weights[idx]); idx += 1

        self.rnn_layers = []
        for _ in range(rnn_layers):
            if bidirectional:
                Wx_f, Wh_f, b_f = weights[idx:idx+3]; idx += 3
                Wx_b, Wh_b, b_b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(
                    BidirectionalRNNLayer((Wx_f, Wh_f, b_f), (Wx_b, Wh_b, b_b), activation=rnn_activation)
                )
            else:
                Wx, Wh, b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(SimpleRNNLayer(Wx, Wh, b, activation=rnn_activation))

        dense_weights = []
        dense_biases = []
        while idx < len(weights):
            W = weights[idx]; idx += 1
            b = weights[idx]; idx += 1
            dense_weights.append(W)
            dense_biases.append(b.reshape(1, -1))

        layer_sizes = [dense_weights[0].shape[0]] + [w.shape[1] for w in dense_weights]

        if dense_activations is None:
            dense_activations = ['softmax'] if len(layer_sizes) == 2 else ['relu'] * (len(layer_sizes) - 2) + ['softmax']

        self.dense = FFNN(layer_sizes, activations_list=dense_activations, loss_function='cce')
        for i, layer in enumerate(self.dense.layers):
            layer.weights = tc.tensor(dense_weights[i], dtype=tc.float32)
            layer.biases = tc.tensor(dense_biases[i], dtype=tc.float32)

    def forward(self, x_token_ids: tc.Tensor):
        x = self.embedding.forward(x_token_ids)
        for rnn in self.rnn_layers:
            x = rnn.forward(x)
        return self.dense.forward(x)

    def predict(self, x_token_ids: tc.Tensor):
        logits = self.forward(x_token_ids)
        return tc.argmax(logits, dim=1)

    def evaluate(self, x_token_ids: tc.Tensor, y_true: tc.Tensor):
        from sklearn.metrics import f1_score
        y_pred = self.predict(x_token_ids).cpu().numpy()
        y_true = y_true.cpu().numpy()
        return f1_score(y_true, y_pred, average='macro')
    
    def load_full_npz(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        config = data['config']
        weights = data['weights']
        self.load_from_config_and_weights(config, weights)

    def load_from_config_and_weights(self, config: dict, weights: list):
        idx = 0
        self.embedding = EmbeddingLayer(weights[idx]); idx += 1

        self.rnn_layers = []
        for _ in range(len(config["rnn_units"])):
            if config["bidirectional"]:
                Wx_f, Wh_f, b_f = weights[idx:idx+3]; idx += 3
                Wx_b, Wh_b, b_b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(
                    BidirectionalRNNLayer((Wx_f, Wh_f, b_f), (Wx_b, Wh_b, b_b), activation=config.get("rnn_activation", "tanh"))
                )
            else:
                Wx, Wh, b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(SimpleRNNLayer(Wx, Wh, b, activation=config.get("rnn_activation", "tanh")))

        dense_weights, dense_biases = [], []
        while idx < len(weights):
            W = weights[idx]; idx += 1
            b = weights[idx]; idx += 1
            dense_weights.append(W)
            dense_biases.append(b.reshape(1, -1))

        layer_sizes = [dense_weights[0].shape[0]] + [w.shape[1] for w in dense_weights]
        dense_activations = config.get("dense_activations")
        self.dense = FFNN(
            layer_sizes,
            activations_list=dense_activations,
            loss_function='cce',
            weight_inits=['manual'] * (len(layer_sizes) - 1)
        )
        for i, layer in enumerate(self.dense.layers):
            layer.weights = tc.tensor(dense_weights[i], dtype=tc.float32)
            layer.biases = tc.tensor(dense_biases[i], dtype=tc.float32)