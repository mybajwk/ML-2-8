
import torch as tc
from .simple_rnn_layer import SimpleRNNLayer

class BidirectionalRNNLayer:
    def __init__(self, forward_weights, backward_weights, activation='tanh'):
        self.rnn_f = SimpleRNNLayer(*forward_weights, activation=activation)
        self.rnn_b = SimpleRNNLayer(*backward_weights, activation=activation)

    def forward(self, x: tc.Tensor, return_sequences=False):
        h_f_out = self.rnn_f.forward(x, return_sequences=return_sequences)

        h_b_out_for_flipped_input = self.rnn_b.forward(tc.flip(x, dims=[1]), return_sequences=return_sequences)

        if return_sequences:
            h_b_aligned = tc.flip(h_b_out_for_flipped_input, dims=[1])
            h_f_final = h_f_out
            h_b_final = h_b_aligned
        else:
            h_f_final = h_f_out.unsqueeze(1)
            h_b_final = h_b_out_for_flipped_input.unsqueeze(1)

        h = tc.cat([h_f_final, h_b_final], dim=-1)  

        if not return_sequences:
            h = h.squeeze(1)

        return h

