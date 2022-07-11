from math import prod
from typing import Any, TypeVar

from numpy import dtype, ndarray

FrameDType = TypeVar("FrameDType", bound=dtype)
FrameShape = TypeVar("FrameShape", bound=Any)


def average_all_frames(
    frames: ndarray[FrameShape, FrameDType]
) -> ndarray[FrameShape, FrameDType]:
    """Average all frames over the leading axis.

    Args:
        frames (ndarray[FrameShape, FrameDType]): A stack of frames to be averaged.

    Returns:
        ndarray[FrameShape, FrameDType]: A frame containing the average pixel values
            of all frames in the stack.
    """
    return frames.reshape(
        [frames.size // prod(frames.shape[-2:]), *frames.shape[-2:]]
    ).sum(axis=-3)
