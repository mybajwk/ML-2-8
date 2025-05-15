import numpy as np
from models.lstm.embedding_layer import ScratchEmbedding
from models.lstm.lstm_layer import ScratchLSTM
from models.lstm.dense_layer import ScratchDense

class ScratchLSTMClassifier:
    def __init__(self, emb_w=None, lstm_specs=None, d_w=None, d_b=None):
        if emb_w is not None and lstm_specs is not None and d_w is not None:
            self.embedding = ScratchEmbedding(emb_w)
            self.dense = ScratchDense(d_w, d_b)
            self.lstm_specs = lstm_specs

    def forward(self, x):
        H = self.embedding.forward(x)
        for spec in self.lstm_specs:
            typ, return_seq, *weights = spec
            if typ == 'unidir':
                lstm = ScratchLSTM(*weights[0])
                H = lstm.forward(H, return_sequences=return_seq)
            else:
                f_lstm = ScratchLSTM(*weights[0])
                b_lstm = ScratchLSTM(*weights[1])
                out_f = f_lstm.forward(H, return_sequences=return_seq)
                out_b = b_lstm.forward(H[:, ::-1, :], return_sequences=return_seq)
                if return_seq:
                    out_b = out_b[:, ::-1, :]
                    H = np.concatenate([out_f, out_b], axis=2)
                else:
                    H = np.concatenate([out_f, out_b], axis=1)
        if H.ndim == 3:
            H = H[:, -1, :]
        return self.dense.forward(H)

    def save_npy(self, path):
        data = {
            "embedding": self.embedding.W,
            "dense_W": self.dense.W,
            "dense_b": self.dense.b,
            "lstm_specs": self.lstm_specs
        }
        np.save(path, data)

    def load_npy(self, path):
        data = np.load(path, allow_pickle=True).item()
        self.embedding = ScratchEmbedding(data["embedding"])
        self.dense = ScratchDense(data["dense_W"], data["dense_b"])
        self.lstm_specs = data["lstm_specs"]