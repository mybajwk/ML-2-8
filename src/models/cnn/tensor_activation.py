import numpy as np

def linear(x): return x
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def softsign(x): return x / (1 + np.abs(x))
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)

activation_functions_np = {
    'linear': linear,
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softsign': softsign,
    'leaky_relu': leaky_relu,
}