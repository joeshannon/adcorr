from typing import Literal, Tuple, TypeVar, Union

from numpy import atleast_1d, dtype, expand_dims, floating, ndarray

FrameDType = TypeVar("FrameDType", bound=dtype)
NumFrames = TypeVar("NumFrames", bound=int)
FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)


def normalize_frame_time(
    frames: ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType],
    count_times: ndarray[Tuple[Union[NumFrames, Literal[1]]], dtype[floating]],
) -> ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]:
    """Normalize for detector frame rate by scaling photon counts according to count time.

    Normalize for detector frame rate by scaling photon counts according to count time,
    as detailed in section 3.4.3 of 'Everything SAXS: small-angle scattering pattern
    collection and correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]): A
            stack of frames to be normalized.
        count_times (ndarray[Tuple[Union[TimesShape, Literal[1]]], dtype[floating]]):
            The period over which photons are counted for each frame.

    Returns:
        ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]: The normalized
            stack of frames.
    """
    if (count_times <= 0).any():
        raise ValueError("Count times must be positive.")

    times = expand_dims(atleast_1d(count_times), (1, 2))
    return frames / times
