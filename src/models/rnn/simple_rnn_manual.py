from .simple_rnn_layer import SimpleRNNLayer
from .bidirectional_rnn_layer import BidirectionalRNNLayer
from .embedding_layer import EmbeddingLayer
import tensorflow as tf
from models.nn.ffnn import FFNN
import torch as tc
import numpy as np
import matplotlib.pyplot as plt

class SimpleRNNManual:
    def __init__(self):
        self.embedding = None
        self.rnn_layers = []
        self.dense = None


    def forward(self, x_token_ids: tc.Tensor):
        x = self.embedding.forward(x_token_ids)
    
        for i, rnn_layer_instance in enumerate(self.rnn_layers): 
            return_sequences = (i < len(self.rnn_layers) - 1)
            x = rnn_layer_instance.forward(x, return_sequences=return_sequences)
            
        if self.dense is not None:
            return self.dense.forward(x)
        return x


    def predict(self, x_token_ids: tc.Tensor):
        logits = self.forward(x_token_ids)
        return tc.argmax(logits, dim=1)
    
    def plot_loss(self, history):
        if not history:
            print("History kosong.")
            return

        train_loss = history['train_loss']
        val_loss = history.get('val_loss', [])

        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label='Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss per Epoch')
        plt.grid(True)
        plt.show()

        if val_loss:
            plt.figure(figsize=(6, 4))
            plt.plot(val_loss, label='Validation Loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Loss per Epoch')
            plt.grid(True)
            plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label='Train Loss', color='blue')
        if val_loss:
            plt.plot(val_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

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

        rnn_activations_from_config = config.get("rnn_activations")
        
        if rnn_activations_from_config is None:
            global_rnn_activation = config.get("rnn_activation", "tanh")
            rnn_activations_from_config = [global_rnn_activation] * len(config["rnn_units"])
        elif len(rnn_activations_from_config) != len(config["rnn_units"]):
             print(f"Warning: Length mismatch between rnn_activations ({len(rnn_activations_from_config)}) and rnn_units ({len(config['rnn_units'])}). Using first activation or tanh.")
             if rnn_activations_from_config:
                 rnn_activations_from_config = [rnn_activations_from_config[0]] * len(config["rnn_units"])
             else:
                 rnn_activations_from_config = ['tanh'] * len(config["rnn_units"])


        for i in range(len(config["rnn_units"])):
            current_rnn_activation = rnn_activations_from_config[i]
            if config["bidirectional"]:
                Wx_f, Wh_f, b_f = weights[idx:idx+3]; idx += 3
                Wx_b, Wh_b, b_b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(
                    BidirectionalRNNLayer((Wx_f, Wh_f, b_f), (Wx_b, Wh_b, b_b), activation=current_rnn_activation)
                )
            else:
                Wx, Wh, b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(SimpleRNNLayer(Wx, Wh, b, activation=current_rnn_activation))

        dense_weights, dense_biases = [], []
        while idx < len(weights):
            if idx + 1 < len(weights):
                W = weights[idx]; idx += 1
                b = weights[idx]; idx += 1
                dense_weights.append(W)
                dense_biases.append(b.reshape(1, -1))
            else:
                break
        
        if not dense_weights:
            self.dense = None
            return

        layer_sizes = [dense_weights[0].shape[0]] + [w.shape[1] for w in dense_weights]
        dense_activations = config.get("dense_activations")
        
        if dense_activations is None:
            if len(layer_sizes) - 1 == 1:
                 dense_activations = ['softmax']
            elif len(layer_sizes) - 1 > 1:
                dense_activations = ['relu'] * (len(layer_sizes) - 2) + ['softmax']
            else:
                dense_activations = []

        self.dense = FFNN(
            layer_sizes,
            activations_list=dense_activations,
            loss_function='cce',
            weight_inits=['manual'] * (len(layer_sizes) - 1) if layer_sizes and len(layer_sizes) > 1 else []
        )
        for i, layer in enumerate(self.dense.layers):
            layer.weights = tc.tensor(dense_weights[i], dtype=tc.float32)
            layer.biases = tc.tensor(dense_biases[i], dtype=tc.float32)
    
    def load_from_keras_h5_auto_config(self, keras_h5_path: str):
        model = tf.keras.models.load_model(keras_h5_path)
        weights = model.get_weights()
        keras_layer_configs = model.get_config()['layers']

        idx = 0 
        self.embedding = EmbeddingLayer(weights[idx]); idx += 1

        self.rnn_layers = []

        keras_rnn_layer_idx_in_weights = 0 
        
        for layer_conf in keras_layer_configs:
            if layer_conf['class_name'] == 'SimpleRNN':
                rnn_activation = layer_conf['config']['activation']
                Wx, Wh, b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(SimpleRNNLayer(Wx, Wh, b, activation=rnn_activation))
                keras_rnn_layer_idx_in_weights +=1

            elif layer_conf['class_name'] == 'Bidirectional':
                wrapped_rnn_conf = layer_conf['config']['layer']['config']
                rnn_activation = wrapped_rnn_conf['activation']

                Wx_f, Wh_f, b_f = weights[idx:idx+3]; idx += 3
                Wx_b, Wh_b, b_b = weights[idx:idx+3]; idx += 3
                self.rnn_layers.append(
                    BidirectionalRNNLayer((Wx_f, Wh_f, b_f), (Wx_b, Wh_b, b_b), activation=rnn_activation)
                )
                keras_rnn_layer_idx_in_weights +=1

        dense_weights, dense_biases = [], []
        dense_activations = []
        
        for layer_conf in keras_layer_configs:
            if layer_conf['class_name'] == 'Dense':
                if idx + 1 < len(weights):
                    W = weights[idx]; idx += 1
                    b = weights[idx]; idx += 1
                    dense_weights.append(W)
                    dense_biases.append(b.reshape(1, -1))
                    dense_activations.append(layer_conf['config']['activation'])
                else:
                    print(f"Warning: Not enough weights for all Dense layers in H5 config. Processed {len(dense_weights)} Dense layers.")
                    break


        if not dense_weights:
            self.dense = None
            return

        layer_sizes = [dense_weights[0].shape[0]] + [w.shape[1] for w in dense_weights]
        self.dense = FFNN(layer_sizes, activations_list=dense_activations, loss_function='cce',
                          weight_inits=['manual'] * len(dense_weights) if dense_weights else [])
        for i, layer_obj in enumerate(self.dense.layers):
            layer_obj.weights = tc.tensor(dense_weights[i], dtype=tc.float32)
            layer_obj.biases = tc.tensor(dense_biases[i], dtype=tc.float32)