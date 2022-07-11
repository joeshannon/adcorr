from typing import Tuple, TypeVar

from numpy import dtype, floating, ndarray

FrameDType = TypeVar("FrameDType", bound=dtype)
NumFrames = TypeVar("NumFrames", bound=int)
FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)


def correct_flatfield(
    frames: ndarray[Tuple[NumFrames, FrameHeight, FrameWidth], FrameDType],
    flatfield: ndarray[Tuple[FrameHeight, FrameWidth], dtype[floating]],
) -> ndarray[Tuple[NumFrames, FrameHeight, FrameWidth], FrameDType]:
    """Apply multiplicative flatfield correction, to correct for inter-pixel sensitivity.

    Apply multiplicative flatfield correction, to correct for inter-pixel sensitivity,
    as described in section 3.xii of 'The modular small-angle X-ray scattering data
    correction sequence' [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (ndarray[Tuple[NumFrames, FrameHeight, FrameWidth], FrameDType]): A
            stack of frames to be corrected.
        flatfield (ndarray[Tuple[FrameHeight, FrameWidth], dtype[floating]]): The
            multiplicative flatfield correction to be applied.

    Returns:
        ndarray[Tuple[NumFrames, FrameHeight, FrameWidth], FrameDType]: The corrected
            stack of frames.
    """
    return frames * flatfield
