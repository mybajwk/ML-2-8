import pickle
from tensorflow.keras.layers import TextVectorization

class TextPreprocessor:
    def __init__(self, max_vocab=10000, max_len=100):
        self.vectorizer = TextVectorization(
            max_tokens=max_vocab,
            output_sequence_length=max_len
        )

    def adapt(self, texts):
        self.vectorizer.adapt(texts)

    def transform(self, texts):
        return self.vectorizer(texts)

    def save(self, path='vectorizer.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    @staticmethod
    def load(path='vectorizer.pkl'):
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
        p = TextPreprocessor()
        p.vectorizer = vectorizer
        return p
