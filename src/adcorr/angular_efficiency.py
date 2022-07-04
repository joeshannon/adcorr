from typing import Any, Tuple, TypeVar

from numpy import cos, dtype, exp
from numpy.ma import MaskedArray, masked_array

from .utils.geometry import scattering_angles

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Any)


def correct_angular_efficincy(
    frames: MaskedArray[FramesShape, FrameDType],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    absorbtion_coefficient: float,
    thickness: float,
) -> MaskedArray[FramesShape, FrameDType]:
    """Corrects for loss due to the angular efficiency of the detector head.

    Corrects for loss due to the angular efficiency of the detector head, as described
    in section 3.xiii and appendix C of 'The modular small-angle X-ray scattering data
    correction sequence' [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (MaskedArray[FramesShape, FrameDType]): A stack of frames to be
            corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample head.
        absorbtion_coefficient (float): The coefficient of absorbtion for a given
            material at a given photon energy.
        thickness (float): The thickness of the detector head material.

    Returns:
        MaskedArray[FramesShape, FrameDType]: The corrected stack of frames.
    """
    absorbtion_efficiency = 1 - exp(
        -absorbtion_coefficient
        * thickness
        / cos(scattering_angles(frames[0].shape, beam_center, pixel_sizes, distance))
    )
    return masked_array(frames / absorbtion_efficiency, frames.mask)
