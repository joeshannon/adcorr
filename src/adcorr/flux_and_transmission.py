from typing import Any, TypeVar

from numpy import dtype, expand_dims, sum
from numpy.ma import MaskedArray, masked_array

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Any)


def normalize_transmitted_flux(
    frames: MaskedArray[FramesShape, FrameDType]
) -> MaskedArray[FramesShape, FrameDType]:
    """Normalize for incident flux and transmissibility by scaling photon counts.

    Normalize for incident flux and transmissibility by scaling photon counts with
    respect to the total observed flux, as detailed in section 4 of `The modular small-
    angle X-ray scattering data correction sequence'
    [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (MaskedArray[FramesShape, FrameDType]): A stack of frames to be
            normalized.

    Returns:
        MaskedArray[FramesShape, FrameDType]: The normalized stack of frames.
    """
    frame_flux = expand_dims(sum(frames.filled(0), axis=(1, 2)), (1, 2))
    return masked_array(frames.data / frame_flux, frames.mask)
