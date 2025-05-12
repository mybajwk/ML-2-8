from .activations import activation_functions
from .initializers import WeightInitializer
import torch as tc

class Layer:
    def __init__(self, input_dim, output_dim, activation_name,
                 weight_init='random_uniform', init_params=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation_name

        self.weights = WeightInitializer.initialize(
            input_dim, output_dim, method=weight_init,
            init_params=init_params, activation=activation_name
        )
        self.biases = WeightInitializer.initialize(
            1, output_dim, method=weight_init,
            init_params=init_params, activation=activation_name
        )

        self.activation, self.d_activation, self.d_activation_times_vector = activation_functions[activation_name]

        self.input = None
        self.net = None
        self.out = None

    def forward(self, X):
        self.input = X  # shape: (batch, input_dim)
        self.net = X @ self.weights + self.biases  # shape: (batch, output_dim)
        self.out = self.activation(self.net)
        return self.out

    def backward(self, dO):
        m = self.input.shape[0]

        if self.activation_name == 'softmax':
            error_term = self.d_activation_times_vector(self.out, dO)
        else:
            error_term = dO * self.d_activation(self.net)

        self.grad_weights = (self.input.t() @ error_term) / m
        self.grad_biases = tc.sum(error_term, dim=0, keepdims=True) / m

        return error_term @ self.weights.t()  # shape: (batch, input_dim)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases