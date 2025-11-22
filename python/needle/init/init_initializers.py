import math
from .init_basic import *
from typing import Any, Optional, Tuple


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    gain: float = 1.0,
    shape: Optional[Tuple[int, ...]] = None,
    **kwargs: Any,
) -> "Tensor":
    """
    Xavier uniform initialization.

    If `shape` is provided, draw a tensor of that shape; otherwise draw a
    (fan_in, fan_out) tensor. The scale is still determined by fan_in/fan_out.
    """
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    if shape is None:
        return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    else:
        return rand(*shape, low=-a, high=a, **kwargs)


def xavier_normal(
    fan_in: int,
    fan_out: int,
    gain: float = 1.0,
    shape: Optional[Tuple[int, ...]] = None,
    **kwargs: Any,
) -> "Tensor":
    """
    Xavier normal initialization.
    """
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    if shape is None:
        return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    else:
        return randn(*shape, mean=0.0, std=std, **kwargs)


def kaiming_uniform(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    shape: Optional[Tuple[int, ...]] = None,
    **kwargs: Any,
) -> "Tensor":
    """
    Kaiming (He) uniform initialization for ReLU.

    `fan_in` and `fan_out` determine the scale; `shape` (if given)
    determines the output tensor shape.
    """
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2.0)
    bound = gain * math.sqrt(3.0 / fan_in)  # sqrt(6 / fan_in)

    if shape is None:
        return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    else:
        return rand(*shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    shape: Optional[Tuple[int, ...]] = None,
    **kwargs: Any,
) -> "Tensor":
    """
    Kaiming (He) normal initialization for ReLU.
    """
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_in)

    if shape is None:
        return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    else:
        return randn(*shape, mean=0.0, std=std, **kwargs)
