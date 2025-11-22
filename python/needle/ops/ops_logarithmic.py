from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 
import numpy

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if isinstance(Z, numpy.ndarray):
            Z_np = Z
            to_backend = lambda x: x
        else:
            Z_np = Z.numpy()
            to_backend = lambda x: array_api.array(x, device=Z.device)

        last = Z_np.ndim - 1
        m = numpy.max(Z_np, axis=last, keepdims=True)
        lse = numpy.log(numpy.sum(numpy.exp(Z_np - m), axis=last, keepdims=True)) + m
        out_np = Z_np - lse
        out_np = out_np.astype("float32")
        return to_backend(out_np)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        last = len(Z.shape) - 1

        lse = logsumexp(Z, axes=(last,))
        lse_b = broadcast_to(reshape(lse, tuple(list(Z.shape[:-1]) + [1])), Z.shape)
        sm = exp(Z - lse_b)

        gsum = summation(out_grad, axes=(last,))
        gsum_b = broadcast_to(reshape(gsum, tuple(list(Z.shape[:-1]) + [1])), Z.shape)
        return out_grad - sm * gsum_b
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if isinstance(Z, numpy.ndarray):
            Z_np = Z
            to_backend = lambda x: x
        else:
            Z_np = Z.numpy()
            to_backend = lambda x: array_api.array(x, device=Z.device)

        axes = self.axes
        if axes is None:
            m = numpy.max(Z_np)
            out_np = numpy.log(numpy.sum(numpy.exp(Z_np - m))) + m
        else:
            if isinstance(axes, tuple):
                axes_np = axes
            else:
                axes_np = (axes,)
            m = numpy.max(Z_np, axis=axes_np, keepdims=True)
            s = numpy.sum(numpy.exp(Z_np - m), axis=axes_np, keepdims=True)
            out_np = numpy.log(s) + m
            out_np = numpy.squeeze(out_np, axis=axes_np)

        out_np = out_np.astype("float32")
        return to_backend(out_np)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes is None:
            lse = logsumexp(Z, axes=None)
            lse_b = broadcast_to(reshape(lse, (1,) * len(Z.shape)), Z.shape)
            og_b = broadcast_to(reshape(out_grad, (1,) * len(Z.shape)), Z.shape)
        else:
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            shape = list(Z.shape)
            for ax in axes:
                shape[ax] = 1
            lse = logsumexp(Z, axes=axes)
            lse_b = broadcast_to(reshape(lse, tuple(shape)), Z.shape)
            og_b = broadcast_to(reshape(out_grad, tuple(shape)), Z.shape)

        return og_b * exp(Z - lse_b)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)