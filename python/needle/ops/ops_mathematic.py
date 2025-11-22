"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return (
            out_grad * b * power(a, b - 1),
            out_grad * power(a, b) * log(a),
        )
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad * self.scalar * power_scalar(x, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b * b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        nd = len(a.shape)
        order = list(range(nd))
        if self.axes is None:
            order[-1], order[-2] = order[-2], order[-1]
        else:
            i, j = self.axes
            order[i], order[j] = order[j], order[i]

        if isinstance(a, numpy.ndarray):
            return numpy.transpose(a, axes=order)
        else:
            return a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return reshape(out_grad, x.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        if len(x.shape) < len(out_grad.shape):
            pad = (1,) * (len(out_grad.shape) - len(x.shape))
            x_shape_aligned = pad + x.shape
        else:
            x_shape_aligned = x.shape
        axes = []
        for i, (sx, so) in enumerate(zip(x_shape_aligned, out_grad.shape)):
            if sx == 1 and so != 1:
                axes.append(i)
        if axes:
            reduced = summation(out_grad, axes=tuple(axes))
        else:
            reduced = out_grad
        return reshape(reduced, x.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = self.axes

        if axes is None:
            return array_api.sum(a, axis=None)

        if isinstance(axes, int):
            return array_api.sum(a, axis=axes)

        ndim = len(a.shape)
        norm_axes = []
        for ax in axes:
            if ax < 0:
                ax += ndim
            norm_axes.append(ax)

        res = a
        for ax in sorted(set(norm_axes), reverse=True):
            res = array_api.sum(res, axis=ax)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = self.axes

        if axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        grad_shape = list(out_grad.shape)
        for ax in sorted(axes):
            grad_shape.insert(ax, 1)

        out_grad = reshape(out_grad, tuple(grad_shape))
        return broadcast_to(out_grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        gA = matmul(out_grad, transpose(B))
        gB = matmul(transpose(A), out_grad)

        def _unbroadcast(grad, shape):
            if grad.shape == shape:
                return grad
            if len(shape) < len(grad.shape):
                padded_shape = (1,) * (len(grad.shape) - len(shape)) + shape
            else:
                padded_shape = shape
            axes = tuple(
                i for i, (gs, ss) in enumerate(zip(grad.shape, padded_shape))
                if ss == 1 and gs != 1
            )
            if axes:
                grad = summation(grad, axes=axes)
            return reshape(grad, shape)

        return _unbroadcast(gA, A.shape), _unbroadcast(gB, B.shape)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad / x
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad * exp(x)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        try:
            data = x.realize_cached_data()
        except Exception:
            data = x.cached_data
        mask_nd = array_api.where(data > 0, 1.0, 0.0)
        mask = Tensor(mask_nd, device=x.device, dtype=x.dtype, requires_grad=False)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = node  
        return out_grad * add_scalar(negate(power_scalar(y, 2)), 1.0)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        arrays = list(args)
        if len(arrays) == 0:
            raise ValueError("stack expects at least one array")

        if isinstance(arrays[0], numpy.ndarray):
            return numpy.stack(arrays, axis=self.axis)

        device = arrays[0].device
        np_arrays = [a.numpy() for a in arrays]
        stacked = numpy.stack(np_arrays, axis=self.axis)
        return array_api.array(stacked, device=device)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        if isinstance(A, numpy.ndarray):
            n = A.shape[self.axis]
            parts = numpy.split(A, n, axis=self.axis)
            return tuple(numpy.squeeze(p, axis=self.axis) for p in parts)

        A_np = A.numpy()
        n = A_np.shape[self.axis]
        parts_np = numpy.split(A_np, n, axis=self.axis)
        squeezed_np = [numpy.squeeze(p, axis=self.axis) for p in parts_np]
        return tuple(array_api.array(p, device=A.device) for p in squeezed_np)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack([t for t in out_grad], self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        if isinstance(axes, list):
            axes = tuple(axes)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a

        in_shape = a.shape
        ndim = len(in_shape)

        norm_axes = set()
        for ax in self.axes:
            if ax < 0:
                ax += ndim
            if 0 <= ax < ndim:
                norm_axes.add(ax)

        out_shape = list(in_shape)
        step = self.dilation + 1
        for ax in norm_axes:
            out_shape[ax] = in_shape[ax] * step

        out = array_api.full(tuple(out_shape), 0.0, device=a.device)

        idx = []
        for i, s in enumerate(out_shape):
            idx.append(slice(0, s, step) if i in norm_axes else slice(0, s))
        out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a

        ndim = len(a.shape)
        norm_axes = set()
        for ax in self.axes:
            if ax < 0:
                ax += ndim
            if 0 <= ax < ndim:
                norm_axes.add(ax)

        step = self.dilation + 1
        idx = []
        for i, s in enumerate(a.shape):
            idx.append(slice(0, s, step) if i in norm_axes else slice(0, s))
        return a[tuple(idx)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        if isinstance(a, numpy.ndarray):
            return numpy.transpose(a, self.axes)
        else:
            return a.compact().permute(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        index = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            index[ax] = i
        return permute(out_grad, tuple(index))
        ### END YOUR SOLUTION


def permute(a, axes):
    return Permute(axes)(a)



class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        assert len(A.shape) == 4, "The input tensor should be 4D"
        assert len(B.shape) == 4, "The kernel tensor should be 4D"

        A = A.compact()
        B = B.compact()

        batch_size, in_height, in_width, in_channel = A.shape
        bs, hs, ws, cs = A.strides
        kernel_height, kernel_width, in_channel_B, out_channel = B.shape
        assert in_channel == in_channel_B

        pad_A = A.pad(
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            )
        ).compact()

        batch_size, in_height, in_width, in_channel = pad_A.shape
        bs, hs, ws, cs = pad_A.strides

        out_height = (in_height - kernel_height) // self.stride + 1
        out_width = (in_width - kernel_width) // self.stride + 1

        receptive_field_shape = (
            batch_size,
            out_height,
            out_width,
            kernel_height,
            kernel_width,
            in_channel,
        )
        receptive_field_strides = (
            bs,
            hs * self.stride,
            ws * self.stride,
            hs,
            ws,
            cs,
        )

        receptive_field = pad_A.as_strided(
            receptive_field_shape, receptive_field_strides
        ).compact()

        rf_vec = receptive_field.reshape(
            (
                receptive_field.size // (kernel_height * kernel_width * in_channel),
                kernel_height * kernel_width * in_channel,
            )
        ).compact()

        kernel_vec = B.reshape(
            (kernel_height * kernel_width * in_channel, out_channel)
        ).compact()

        out = rf_vec @ kernel_vec
        out = out.reshape(
            (batch_size, out_height, out_width, out_channel)
        ).compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        k, _, _, _ = W.shape

        # grad w.r.t. X
        W_flipped = flip(W, (0, 1))
        W_flipped_permuted = transpose(W_flipped, (2, 3))
        outgrad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        X_grad = conv(
            outgrad_dilated,
            W_flipped_permuted,
            padding=k - 1 - self.padding,
        )

        outgrad_dilated_permuted = permute(outgrad_dilated, (1, 2, 0, 3))
        X_permuted = permute(X, (3, 1, 2, 0))
        W_grad = conv(
            X_permuted,
            outgrad_dilated_permuted,
            padding=self.padding,
        )
        W_grad = permute(W_grad, (1, 2, 0, 3))

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
