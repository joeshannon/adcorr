from typing import Tuple, TypeVar, cast

from numpy import cos, dtype, ndarray, sin, square

from .utils.geometry import azimuthal_angles, scattering_angles

FrameDType = TypeVar("FrameDType", bound=dtype)
FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def correct_polarization(
    frames: ndarray[FramesShape, FrameDType],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    horizontal_poarization: float = 0.5,
) -> ndarray[FramesShape, FrameDType]:
    """Corrects for the effect of polarization of the incident beam.

    Corrects for the effect of polarization of the incident beam, as detailed in
    section 3.4.1 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (ndarray[FramesShape, FrameDType]): A stack of frames to be corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.
        horizontal_poarization (float, optional): The fraction of incident radiation
            polarized in the horizontal plane, where 0.5 signifies an unpolarized
            source. Defaults to 0.5.

    Returns:
        ndarray[FramesShape, FrameDType]: The corrected stack of frames.
    """
    if horizontal_poarization < 0.0 or horizontal_poarization > 1.0:
        raise ValueError("Horizontal Polarization must be within the interval [0, 1].")

    scattering = scattering_angles(
        cast(tuple[int, int], frames.shape[-2:]), beam_center, pixel_sizes, distance
    )
    azimuths = azimuthal_angles(
        cast(tuple[int, int], frames.shape[-2:]), beam_center, pixel_sizes
    )
    correction_factors = horizontal_poarization * (
        1.0 - square(sin(azimuths) * sin(scattering))
    ) + (1.0 - horizontal_poarization) * (1.0 - square(cos(azimuths) * sin(scattering)))
    return frames * correction_factors
