import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
from needle import ops
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

        C1, C2, C3, C4 = 16, 32, 64, 128

        def conv_bn_relu(in_c, out_c, k, stride):
            return ndl.nn.Sequential(
                ndl.nn.Conv(in_c, out_c, k, stride=stride,
                            bias=True, device=device, dtype=dtype),
                ndl.nn.BatchNorm2d(out_c, device=device, dtype=dtype),
                ndl.nn.ReLU(),
            )

        def residual_block(c):
            return ndl.nn.Residual(
                ndl.nn.Sequential(
                    ndl.nn.Conv(c, c, 3, stride=1,
                                bias=True, device=device, dtype=dtype),
                    ndl.nn.BatchNorm2d(c, device=device, dtype=dtype),
                    ndl.nn.ReLU(),
                    ndl.nn.Conv(c, c, 3, stride=1,
                                bias=True, device=device, dtype=dtype),
                    ndl.nn.BatchNorm2d(c, device=device, dtype=dtype),
                    ndl.nn.ReLU(),
                )
            )


        self.features = ndl.nn.Sequential(
            conv_bn_relu(3, C1, 7, 4),    
            conv_bn_relu(C1, C2, 3, 2),   
            residual_block(C2),          
            conv_bn_relu(C2, C3, 3, 2),   
            conv_bn_relu(C3, C4, 3, 2),   
            residual_block(C4),           
        )

        self.classifier = ndl.nn.Sequential(
            ndl.nn.Flatten(),
            ndl.nn.Linear(C4, 128, device=device, dtype=dtype), 
            ndl.nn.ReLU(),
            ndl.nn.Linear(128, 10, device=device, dtype=dtype),  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.seq_model_type = seq_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, bias=True, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bias=True, device=device, dtype=dtype)
        else:
            raise ValueError("Unknown seq_model: %s" % seq_model)
        self.proj = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        emb = self.embedding(x)
        out, h_out = self.seq_model(emb, h)
        S, B, H = out.shape
        logits = self.proj(ops.reshape(out, (S * B, H)))
        return logits, h_out
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
