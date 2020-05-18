import torch
import numpy as np
from torch import nn


class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """

    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        super(GRUCellV2, self).__init__()
        self.activation = activation

        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.hidden_size = hidden_size

    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell


        Returns the current hidden state h_t for every datapoint in batch.

        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """

        g_input = torch.mm(x, self.w_ih.t()) + self.b_ih
        g_hidden = torch.mm(h, self.w_hh.t()) + self.b_hh

        input_reset, input_input, input_new = g_input.chunk(3, 1)
        hidden_reset, hidden_input, hidden_new = g_hidden.chunk(3, 1)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        input_gate = torch.sigmoid(input_input + hidden_input)
        new_gate = torch.tanh(input_new + reset_gate * hidden_new)

        h_t = ((1 - input_gate) * new_gate) + (input_gate * h[0])
        return h_t


torch.manual_seed(100304343)
x = torch.randn(5, 3, 10)
gru = nn.GRU(10, 20, bidirectional=False, batch_first=True)
outputs, h = gru(x)



rnn = nn.GRUCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    print(np.shape(input[i]))
    hx = rnn(input[i], hx)
    output.append(hx)
print(output)

gnn = GRUCellV2(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = gnn(input[i], hx)
    output.append(hx)

print(output)

#
# torch.manual_seed(100304343)
# x = torch.randn(5, 3, 10)
# gru2 = GRU2(10, 20, bidirectional=False)
# outputs, h_fw = gru2(x)
#
# print("Checking the unidirectional GRU implementation")
# print("Same hidden states of the forward cell?\t\t{}".format(
#     is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
# ))
#
# torch.manual_seed(100304343)
# x = torch.randn(5, 3, 10)
# gru = GRU2(10, 20, bidirectional=True)
# outputs, h_fw, h_bw = gru(x)
#
# torch.manual_seed(100304343)
# x = torch.randn(5, 3, 10)
# gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
# outputs, h = gru2(x)
#
# print("Checking the bidirectional GRU implementation")
# print("Same hidden states of the forward cell?\t\t{}".format(
#     is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
# ))
# print("Same hidden states of the backward cell?\t{}".format(
#     is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
# ))