from typing import Tuple, TypeVar

from numpy import dtype
from numpy.ma import MaskedArray, masked_array

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def normalize_thickness(
    frames: MaskedArray[FramesShape, FrameDType],
    sample_thickness: float,
) -> MaskedArray[FramesShape, FrameDType]:
    """Normailizes pixel intensities by dividing by the sample thickness.

    Normailizes pixel intensities by dividing by the sample thickness, as detailed in
    section 3.4.3 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[FramesShape, FrameDType]): A stack of uncertain frames to be
            corrected.
        sample_thickness (float): The thickness of the exposed sample.

    Returns:
        MaskedArray[FramesShape, FrameDType]: The normalized stack of frames.
    """
    return masked_array(frames.data / sample_thickness, frames.mask)
