from typing import Any, Tuple, TypeVar, cast

from numpy import cos, dtype, exp, ndarray

from .utils.geometry import scattering_angles

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Any)


def correct_angular_efficiency(
    frames: ndarray[FramesShape, FrameDType],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    absorption_coefficient: float,
    thickness: float,
) -> ndarray[FramesShape, FrameDType]:
    """Corrects for loss due to the angular efficiency of the detector head.

    Corrects for loss due to the angular efficiency of the detector head, as described
    in section 3.xiii and appendix C of 'The modular small-angle X-ray scattering data
    correction sequence' [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (ndarray[FramesShape, FrameDType]): A stack of frames to be
            corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample head.
        absorption_coefficient (float): The coefficient of absorption for a given
            material at a given photon energy.
        thickness (float): The thickness of the detector head material.

    Returns:
        ndarray[FramesShape, FrameDType]: The corrected stack of frames.
    """
    if absorption_coefficient <= 0.0:
        raise ValueError("absorption coefficient must positive.")
    if thickness <= 0.0:
        raise ValueError("Thickness must be positive.")

    absorption_efficiency = 1.0 - exp(
        -absorption_coefficient
        * thickness
        / cos(
            scattering_angles(
                cast(tuple[int, int], frames.shape[-2:]),
                beam_center,
                pixel_sizes,
                distance,
            )
        )
    )
    return frames / absorption_efficiency
