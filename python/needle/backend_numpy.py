"""This file defies specific implementations of devices when using numpy as NDArray backend.
"""
import numpy


class Device:
    """Baseclass of all device"""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    def __repr__(self):
        return "needle.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.randn(*shape)

    def rand(self, *shape):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        # Efficient one-hot without building full identity
        idx = numpy.asarray(i, dtype=numpy.int64).reshape(-1)
        m = idx.shape[0]
        out = numpy.zeros((m, n), dtype=dtype)
        out[numpy.arange(m, dtype=numpy.int64), idx] = 1
        return out

    def empty(self, shape, dtype="float32"):
        return numpy.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return numpy.full(shape, fill_value, dtype=dtype)


def cpu():
    """Return cpu device"""
    return CPUDevice()


def default_device():
    return cpu()


def all_devices():
    """return a list of all available devices"""
    return [cpu()]
