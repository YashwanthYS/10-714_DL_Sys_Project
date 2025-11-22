"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = kernel_size // 2

        k = self.kernel_size
        fan_in = in_channels * k * k
        fan_out = out_channels * k * k

        w_shape = (k, k, in_channels, out_channels)
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in,
                fan_out,
                shape=w_shape,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            bound = 1.0 / np.sqrt(in_channels * k * k)
            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_nhwc = x.transpose((1, 2)).transpose()

        y = ops.conv(
            x_nhwc,
            self.weight,
            stride=self.stride,
            padding=self.padding,
        )

        if self.bias is not None:
            b = self.bias.reshape((1, 1, 1, self.out_channels))
            b = ops.broadcast_to(b, y.shape)
            y = y + b

        y_nchw = y.transpose().transpose((1, 2))
        return y_nchw
        ### END YOUR SOLUTION