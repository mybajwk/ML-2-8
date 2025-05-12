
import torch as tc
from .simple_rnn_layer import SimpleRNNLayer

class BidirectionalRNNLayer:
    def __init__(self, forward_weights, backward_weights, activation='tanh'):
        self.rnn_f = SimpleRNNLayer(*forward_weights, activation=activation)
        self.rnn_b = SimpleRNNLayer(*backward_weights, activation=activation)

    def forward(self, x: tc.Tensor):
        h_f = self.rnn_f.forward(x)
        h_b = self.rnn_b.forward(tc.flip(x, dims=[1])) 
        return tc.cat([h_f, h_b], dim=1)