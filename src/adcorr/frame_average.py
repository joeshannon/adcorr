from typing import Any, TypeVar

from numpy import dtype
from numpy.ma import MaskedArray, masked_array

FrameDType = TypeVar("FrameDType", bound=dtype)
FrameShape = TypeVar("FrameShape", bound=Any)


def average_frames(
    frames: MaskedArray[FrameShape, FrameDType]
) -> MaskedArray[FrameShape, FrameDType]:
    """Average all frames over the leading axis.

    Args:
        frames (MaskedArray[FrameShape, FrameDType]): A stack of frames to be averaged.

    Returns:
        MaskedArray[FrameShape, FrameDType]: A frame containing the average pixel values
            of all frames in the stack.
    """
    return masked_array(frames.sum(axis=0) / frames.shape[0], mask=frames.mask[0])
