from typing import Tuple, TypeVar

from numpy import array, dtype, expand_dims, floating, ndarray
from numpy.ma import MaskedArray, masked_array

FrameDType = TypeVar("FrameDType", bound=dtype)
NumFrames = TypeVar("NumFrames", bound=int)
FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)


def correct_dark_current(
    frames: MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType],
    count_times: ndarray[Tuple[NumFrames], dtype[floating]],
    base_dark_current: float,
    temporal_dark_current: float,
    flux_dependant_dark_current: float,
) -> MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]:
    """Correct by subtracting base, temporal and flux-dependant dark currents.

    Correct for incident dark current by subtracting a baselike, time dependant and a
    flux dependant count rate from frames, as detailed in section 3.3.6 of 'Everything
    SAXS: small-angle scattering pattern collection and correction'
    [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]): A
            stack of frames to be corrected.
        atemporal_dark_current (float): The dark current flux, irrespective of time.
        temporal_dark_current (float): The dark current flux, as a factor of time.
        flux_dependant_dark_current (float): The dark current flux, as a factor of
            incident flux.
        count_times (ndarray[Tuple[NumFrames], dtype[floating]]): The period over which
            photons are counted for each frame.

    Returns:
        MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], FrameDType]: The
            corrected stack of frames.
    """
    base = array([(base_dark_current, 0.0)])
    temporal = expand_dims(temporal_dark_current * count_times, (-2, -1))
    flux_dependant = flux_dependant_dark_current * frames
    return masked_array(frames.data - base - temporal - flux_dependant, frames.mask)
