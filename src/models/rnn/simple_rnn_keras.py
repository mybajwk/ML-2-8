import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Embedding, Dropout, Dense, SimpleRNN, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class SimpleRNNKeras:
    def __init__(
        self,
        max_vocab=10000,
        max_len=100,
        embedding_dim=128,
        rnn_units=[64],
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
        self.dense_units = dense_units
        self.dense_activations = dense_activations
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = None

    def set_vectorized_data(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test  = X_test
        self.y_test  = y_test

    def build_model(self):
        layers = [Embedding(input_dim=self.max_vocab, output_dim=self.embedding_dim, input_length=self.max_len)]

        for i, units in enumerate(self.rnn_units):
            rnn_layer = SimpleRNN(units, return_sequences=(i < len(self.rnn_units) - 1))
            if self.bidirectional:
                rnn_layer = Bidirectional(rnn_layer)
            layers.append(rnn_layer)

        layers.append(Dropout(self.dropout))

        for units, activation in zip(self.dense_units, self.dense_activations):
            layers.append(Dense(units, activation=activation))

        self.model = Sequential(layers)
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

    def train(self, epochs=10, batch_size=64):
        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self):
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        return f1_score(self.y_test, y_pred, average='macro')

    def save(self, path="model_simple_rnn.h5"):
        self.model.save(path)

    def save_full_npz(self, path="model_simple_rnn.npz"):
        weights = self.model.get_weights()
        config = {
            "rnn_units": self.rnn_units,
            "dense_units": self.dense_units,
            "dense_activations": self.dense_activations,
            "embedding_dim": self.embedding_dim,
            "max_vocab": self.max_vocab,
            "max_len": self.max_len,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate
        }

        np.savez_compressed(path, weights=weights, config=config)
        print(f"Saved full model to {path}")
