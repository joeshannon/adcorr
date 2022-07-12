from typing import Any

from numpy import bool_, broadcast_to, dtype, ndarray
from numpy.ma import MaskedArray, masked_where

from .utils.typing import FrameDType, StackShape


def mask_frames(
    frames: ndarray[StackShape, FrameDType], mask: ndarray[Any, dtype[bool_]]
) -> MaskedArray[StackShape, FrameDType]:
    """Replaces masked elemenets of frames in a stack with zero.

    Args:
        frames: A stack of frames to be masked.
        mask: The boolean mask to apply to each frame.

    Returns:
        A stack of frames where pixels.
    """
    return masked_where(broadcast_to(mask, frames.shape), frames)
