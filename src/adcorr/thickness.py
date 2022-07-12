from typing import Tuple, TypeVar, cast

from numpy import dtype, ndarray

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def normalize_thickness(
    frames: ndarray[FramesShape, FrameDType],
    sample_thickness: float,
) -> ndarray[FramesShape, FrameDType]:
    """Normailizes pixel intensities by dividing by the sample thickness.

    Normailizes pixel intensities by dividing by the sample thickness, as detailed in
    section 3.4.3 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (ndarray[FramesShape, FrameDType]): A stack of frames to be corrected.
        sample_thickness (float): The thickness of the exposed sample.

    Returns:
        ndarray[FramesShape, FrameDType]: The normalized stack of frames.
    """
    if sample_thickness <= 0:
        raise ValueError("Sample Thickness must be positive.")

    return cast(ndarray[FramesShape, FrameDType], frames / sample_thickness)
