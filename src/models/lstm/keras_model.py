from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense

def build_lstm_model(n_layers, units, bidirectional, max_len, max_tokens, embed_dim, num_classes):
    inp = Input(shape=(max_len,), dtype='int32')
    x = Embedding(input_dim=max_tokens, output_dim=embed_dim)(inp)
    for i in range(n_layers):
        lstm_cls = LSTM if not bidirectional else lambda *a,**kw: Bidirectional(LSTM(*a, **kw))
        return_seq = i < n_layers - 1
        x = lstm_cls(units[i], return_sequences=return_seq)(x)
        x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[])
    return model