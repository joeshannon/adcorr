from typing import Tuple, TypeVar, cast

from numpy import dtype, ndarray

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def correct_displaced_volume(
    frames: ndarray[FramesShape, FrameDType],
    displaced_fraction: float,
) -> ndarray[FramesShape, FrameDType]:
    """Correct for displaced volume of solvent by multiplying signal by retained fraction.

    Correct for displaced volume of solvent by multiplying signal by the retained
    fraction, as detailed in section 3.xviii and appendix B of `The modular small-angle
    X-ray scattering data correction sequence'
    [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (ndarray[FramesShape, FrameDType]):  A stack of frames to be corrected.
        displaced_fraction (float): The fraction of solvent displaced by the analyte.

    Returns:
        ndarray[FramesShape, FrameDType]: The corrected stack of frames.
    """
    if displaced_fraction < 0.0 or displaced_fraction > 1.0:
        raise ValueError("Displaced Fraction must be in interval [0, 1].")

    return cast(ndarray[FramesShape, FrameDType], frames * (1.0 - displaced_fraction))
