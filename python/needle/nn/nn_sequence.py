"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(ops.add_scalar(ops.exp(ops.negate(x)), 1.0), -1.0)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        k = 1.0 / hidden_size
        bound = np.sqrt(k)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            out = out + ops.broadcast_to(ops.reshape(self.bias_ih, (1, self.hidden_size)), out.shape)
            out = out + ops.broadcast_to(ops.reshape(self.bias_hh, (1, self.hidden_size)), out.shape)
        if self.nonlinearity == 'relu':
            return ops.relu(out)
        else:
            return ops.tanh(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.rnn_cells: List[RNNCell] = []
        in_size = input_size
        for _ in range(num_layers):
            cell = RNNCell(in_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
            self.rnn_cells.append(cell)
            in_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        # split sequence along time dimension into list of (bs, input_size)
        xs = [t for t in ops.split(X, axis=0)]
        if h0 is None:
            h_prev = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        else:
            h_prev = [t for t in ops.split(h0, axis=0)]

        outputs = []
        for x_t in xs:
            h_next = []
            for l, cell in enumerate(self.rnn_cells):
                h_t = cell(x_t, h_prev[l])
                h_next.append(h_t)
                x_t = h_t
            outputs.append(x_t)
            h_prev = h_next

        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_prev, axis=0)
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        k = 1.0 / hidden_size
        bound = np.sqrt(k)
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h

        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            gates = gates + ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)), gates.shape)
            gates = gates + ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)), gates.shape)

        parts = [t for t in ops.split(gates, axis=1)]
        H = self.hidden_size
        i = ops.stack(parts[0:H], axis=1)
        f = ops.stack(parts[H:2 * H], axis=1)
        g = ops.stack(parts[2 * H:3 * H], axis=1)
        o = ops.stack(parts[3 * H:4 * H], axis=1)

        sigm = Sigmoid()
        i = sigm(i)
        f = sigm(f)
        o = sigm(o)
        g = ops.tanh(g)

        c1 = f * c0 + i * g
        h1 = o * ops.tanh(c1)
        return h1, c1
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.lstm_cells: List[LSTMCell] = []
        in_size = input_size
        for _ in range(num_layers):
            cell = LSTMCell(in_size, hidden_size, bias=bias, device=device, dtype=dtype)
            self.lstm_cells.append(cell)
            in_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        xs = [t for t in ops.split(X, axis=0)]
        if h is None:
            h_prev = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
            c_prev = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        else:
            h0, c0 = h
            h_prev = [t for t in ops.split(h0, axis=0)]
            c_prev = [t for t in ops.split(c0, axis=0)]

        outputs = []
        for x_t in xs:
            h_next = []
            c_next = []
            for l, cell in enumerate(self.lstm_cells):
                h_t, c_t = cell(x_t, (h_prev[l], c_prev[l]))
                h_next.append(h_t)
                c_next.append(c_t)
                x_t = h_t
            outputs.append(x_t)
            h_prev, c_prev = h_next, c_next

        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_prev, axis=0)
        c_n = ops.stack(c_prev, axis=0)
        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # x: (seq_len, bs) containing word indices (float32 but integral values)
        S, B = x.shape
        idx = x.numpy().astype(np.int32).reshape(-1)
        # one-hot: (S*B, V) on same device as embedding weights
        oh = init.one_hot(
            self.num_embeddings,
            Tensor(idx, device=self.weight.device, dtype="float32"),
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        emb2d = oh @ self.weight  # (S*B, D)
        return ops.reshape(emb2d, (S, B, self.embedding_dim))
        ### END YOUR SOLUTION