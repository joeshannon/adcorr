from typing import Tuple, TypeVar

from numpy import dtype, ndarray

FrameDType = TypeVar("FrameDType", bound=dtype)
NumFrames = TypeVar("NumFrames", bound=int)
FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)


def subtract_background(
    foreground_frames: ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType],
    background_frame: ndarray[Tuple[FrameWidth, FrameHeight], FrameDType],
) -> ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]:
    """Subtract a background frame from a sequence of foreground frames.

    Subtract a background frame from a sequence of foreground frames, as detailed in
    section 3.4.6 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        foreground_frames (ndarray[Tuple[NumFrames, FrameWidth, FrameHeight],
            FrameDType]): A sequence of foreground frames to be corrected.
        background_frame (ndarray[Tuple[FrameWidth, FrameHeight], FrameDType]): The
            background which is to be corrected for.

    Returns:
        ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]: A sequence of
            corrected frames.
    """
    return foreground_frames - background_frame
