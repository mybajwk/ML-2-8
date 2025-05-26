import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Embedding, Dropout, Dense, SimpleRNN, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class SimpleRNNKeras:
    def __init__(
        self,
        max_vocab=10000,
        max_len=100,
        embedding_dim=128,
        rnn_units=[64],
        rnn_activations=None,
        dense_units=[3],
        dense_activations=['softmax'],
        bidirectional=True,
        dropout=0.5,
        learning_rate=1e-3
    ):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units

        if rnn_activations is None:
            self.rnn_activations = ['tanh'] * len(rnn_units)
        elif isinstance(rnn_activations, str):
            self.rnn_activations = [rnn_activations] * len(rnn_units)
        elif isinstance(rnn_activations, list):
            if len(rnn_activations) != len(rnn_units):
                raise ValueError("Length of rnn_activations must match length of rnn_units.")
            self.rnn_activations = rnn_activations
        else:
            raise ValueError("rnn_activations must be a string, a list of strings, or None.")

        self.dense_units = dense_units
        self.dense_activations = dense_activations
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_classes = self.dense_units[-1] if self.dense_units else None

        self.model = None

    def set_vectorized_data(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test  = X_test
        self.y_test  = y_test

    def build_model(self):
        layers = []
    
        layers.append(Embedding(input_dim=self.max_vocab, output_dim=self.embedding_dim))
    
        for i, units in enumerate(self.rnn_units):
            current_rnn_activation = self.rnn_activations[i]
            print(f"Adding RNN layer {i+1} with {units} units and activation {current_rnn_activation}")
            rnn_layer_instance = SimpleRNN(units, activation=current_rnn_activation, return_sequences=(i < len(self.rnn_units) - 1))
            if self.bidirectional:
                rnn_layer_instance = Bidirectional(rnn_layer_instance) 
            layers.append(rnn_layer_instance)
    
        layers.append(Dropout(self.dropout))
    
        for units, activation in zip(self.dense_units, self.dense_activations):
            layers.append(Dense(units, activation=activation))
    
        self.model = Sequential(layers)
        
        metrics_to_use = ['accuracy']
        # from tensorflow.keras.metrics import F1Score
        # if self.num_classes and self.num_classes > 1:
        #     f1_macro_metric = F1Score(
        #         num_classes=self.num_classes, 
        #         average='macro', 
        #         name='f1_macro'
        #     )
        #     metrics_to_use.append(f1_macro_metric)

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=metrics_to_use
        )

    def train(self, epochs=10, batch_size=64, shuffle_data=True):
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle_data
        )

    def evaluate(self):
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        return f1_score(self.y_test, y_pred, average='macro')
    
    def plot_loss(self):
        if not hasattr(self, 'history'):
            print("Model belum dilatih.")
            return

        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label='Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss per Epoch')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(val_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss per Epoch')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label='Train Loss', color='blue')
        plt.plot(val_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save(self, path="model_simple_rnn.h5"):
        self.model.save(path)

    def save_full_npz(self, path="model_simple_rnn.npy"):
        data = {
            "weights": self.model.get_weights(),
            "config": {
                "rnn_units": self.rnn_units,
                "rnn_activations": self.rnn_activations, 
                "dense_units": self.dense_units,
                "dense_activations": self.dense_activations,
                "embedding_dim": self.embedding_dim,
                "max_vocab": self.max_vocab,
                "max_len": self.max_len,
                "bidirectional": self.bidirectional,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate
            }
        }
        np.save(path, data, allow_pickle=True)
        print(f"Saved full model to {path}")