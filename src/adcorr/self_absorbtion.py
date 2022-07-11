from math import exp
from typing import Tuple, TypeVar, cast

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

FrameDType = TypeVar("FrameDType", bound=dtype)
NumFrames = TypeVar("NumFrames", bound=int)
FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)


def correct_self_absorbtion(
    frames: ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    absorbtion_coefficient: float,
    thickness: float,
) -> ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]:
    """Correct for transmission loss due to differences in observation angle.

    Correct for transmission loss due to differences in observation angle, as detailed
    in section 3.4.7 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (ndarray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]): A
            stack of frames to be corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.
        absorbtion_coefficient (float): The coefficient of absorbtion for a given
            material at a given photon energy.
        thickness (float): The thickness of the detector head material.

    Returns:
        ndarray[FramesShape, FrameDType]: The corrected stack of frames.
    """
    if absorbtion_coefficient < 0.0:
        raise ValueError("Absorbtion coefficient must non-negative.")
    if thickness < 0.0:
        raise ValueError("Thickness must be non-negative.")

    angles = scattering_angles(
        cast(Tuple[int, int], frames.shape[-2:]), beam_center, pixel_sizes, distance
    )
    transmissibility = exp(-absorbtion_coefficient * thickness)
    secangle: ndarray[Tuple[int, int], dtype[floating]] = 1 / cos(angles)
    correction_factors: ndarray[Tuple[int, int], dtype[floating]] = divide(
        1 - power(transmissibility, secangle - 1),
        log(transmissibility) * (1 - secangle),
        out=ones_like(secangle),
        where=logical_and(secangle != 1.0, transmissibility != 1.0),
    )

    return frames * correction_factors
