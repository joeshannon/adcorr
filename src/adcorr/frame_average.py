from math import prod

from .utils.typing import Frames


def average_all_frames(frames: Frames) -> Frames:
    """Average all frames over the leading axis.

    Args:
        frames: A stack of frames to be averaged.

    Returns:
        A frame containing the average pixel values of all frames in the stack.
    """
    return frames.reshape(
        [frames.size // prod(frames.shape[-2:]), *frames.shape[-2:]]
    ).sum(axis=-3)
