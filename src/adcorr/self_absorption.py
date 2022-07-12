from math import exp
from typing import Tuple, cast

from numpy import (
    cos,
    divide,
    dtype,
    floating,
    log,
    logical_and,
    ndarray,
    ones_like,
    power,
)

from .utils.geometry import scattering_angles
from .utils.typing import Frames


def correct_self_absorption(
    frames: Frames,
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    absorption_coefficient: float,
    thickness: float,
) -> Frames:
    """Correct for transmission loss due to differences in observation angle.

    Correct for transmission loss due to differences in observation angle, as detailed
    in section 3.4.7 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames: A stack of frames to be corrected.
        beam_center: The center position of the beam in pixels.
        pixel_sizes: The real space size of a detector pixel.
        distance: The distance between the detector and the sample.
        absorption_coefficient: The coefficient of absorption for a given
            material at a given photon energy.
        thickness: The thickness of the detector head material.

    Returns:
        The corrected stack of frames.
    """
    if absorption_coefficient < 0.0:
        raise ValueError("absorption coefficient must non-negative.")
    if thickness < 0.0:
        raise ValueError("Thickness must be non-negative.")

    angles = scattering_angles(
        cast(Tuple[int, int], frames.shape[-2:]), beam_center, pixel_sizes, distance
    )
    transmissibility = exp(-absorption_coefficient * thickness)
    secangle: ndarray[Tuple[int, int], dtype[floating]] = 1 / cos(angles)
    correction_factors: ndarray[Tuple[int, int], dtype[floating]] = divide(
        1 - power(transmissibility, secangle - 1),
        log(transmissibility) * (1 - secangle),
        out=ones_like(secangle),
        where=logical_and(secangle != 1.0, transmissibility != 1.0),
    )

    return frames * correction_factors
