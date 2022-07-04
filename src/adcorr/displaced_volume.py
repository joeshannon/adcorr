from typing import Tuple, TypeVar

from numpy import dtype
from numpy.ma import MaskedArray, masked_array

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def correct_displaced_volume(
    frames: MaskedArray[FramesShape, FrameDType],
    displaced_fraction: float,
) -> MaskedArray[FramesShape, FrameDType]:
    """Correct for displaced volume of solvent by multiplying signal by retained fraction.

    Correct for displaced volume of solvent by multiplying signal by the retained
    fraction, as detailed in section 3.xviii and appendix B of `The modular small-angle
    X-ray scattering data correction sequence'
    [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (MaskedArray[FramesShape, FrameDType]):  A stack of frames to be
            corrected.
        displaced_fraction (float): The fraction of solvent displaced by the analyte.

    Returns:
        MaskedArray[FramesShape, FrameDType]: The corrected stack of frames.
    """
    retained_fraction = 1 - displaced_fraction
    return masked_array(frames.data * retained_fraction, mask=frames.mask)
