from typing import Any, TypeVar

from numpy import atleast_1d, dtype, expand_dims, floating, ndarray
from numpy.ma import MaskedArray, masked_array

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Any)
TimesShape = TypeVar("TimesShape", bound=Any)


def normalize_frame_time(
    frames: MaskedArray[FramesShape, FrameDType],
    count_times: ndarray[TimesShape, dtype[floating]],
) -> MaskedArray[FramesShape, FrameDType]:
    """Normalize for detector frame rate by scaling photon counts according to count time.

    Normalize for detector frame rate by scaling photon counts according to count time,
    as detailed in section 3.4.3 of 'Everything SAXS: small-angle scattering pattern
    collection and correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[FramesShape, FrameDType]): A stack of frames to be
            normalized.
        count_times (ndarray[TimesShape, dtype[floating]]): The period over which
            photons are counted for each frame.

    Returns:
        MaskedArray[FramesShape, FrameDType]: The normalized stack of frames.
    """
    times = expand_dims(atleast_1d(count_times), (-2, -1))
    return masked_array(frames.data / times, frames.mask)
