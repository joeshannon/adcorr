from typing import Tuple, TypeVar

from numpy import cos, dtype, sin, square
from numpy.ma import MaskedArray, masked_array

from .utils.geometry import azimuthal_angles, scattering_angles

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def correct_polarization(
    frames: MaskedArray[FramesShape, FrameDType],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    horizontal_poarization: float = 0.5,
) -> MaskedArray[FramesShape, FrameDType]:
    """Corrects for the effect of polarization of the incident beam.

    Corrects for the effect of polarization of the incident beam, as detailed in
    section 3.4.1 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[FramesShape, FrameDType]): A stack of frames to be
            corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.
        horizontal_poarization (float, optional): The fraction of incident radiation
            polarized in the horizontal plane, where 0.5 signifies an unpolarized
            source. Defaults to 0.5.

    Returns:
        MaskedArray[FramesShape, FrameDType]: The corrected stack of frames.
    """
    scattering = scattering_angles(
        frames.shape[-2:], beam_center, pixel_sizes, distance
    )
    azimuths = azimuthal_angles(frames.shape[-2:], beam_center)
    correction_factor = horizontal_poarization * (
        1 - square(sin(azimuths) * sin(scattering))
    ) + (1 - horizontal_poarization) * (1 - square(cos(azimuths) * sin(scattering)))
    return masked_array(frames * correction_factor, frames.mask)
