from typing import Any, Tuple, TypeVar

from numpy import dtype
from numpy.ma import MaskedArray, masked_array

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Any)


def subtract_background(
    foreground_frames: MaskedArray[FramesShape, FrameDType],
    background_frame: MaskedArray[Tuple[int, int], FrameDType],
) -> MaskedArray[FramesShape, FrameDType]:
    """Subtract a background frame from a sequence of foreground frames.

    Subtract a background frame from a sequence of foreground frames, as detailed in
    section 3.4.6 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        foreground_frames (MaskedArray[FramesShape, FrameDType]): A sequence of
            foreground frames to be corrected.
        background_frame (MaskedArray[Tuple[int, int], FrameDType]): The background
            which is to be corrected for.

    Returns:
        MaskedArray[FramesShape, FrameDType]: A sequence of corrected frames.
    """
    return masked_array(
        foreground_frames.data - background_frame.data,
        foreground_frames.mask,
    )
