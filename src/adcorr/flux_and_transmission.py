from numpy import expand_dims, sum

from .utils.typing import Frames


def normalize_transmitted_flux(frames: Frames) -> Frames:
    """Normalize for incident flux and transmissibility by scaling photon counts.

    Normalize for incident flux and transmissibility by scaling photon counts with
    respect to the total observed flux, as detailed in section 4 of 'The modular small-
    angle X-ray scattering data correction sequence'
    [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames: A stack of frames to be normalized.

    Returns:
        The normalized stack of frames.
    """
    frame_flux = expand_dims(sum(frames, axis=(-1, -2)), (-1, -2))
    return frames / frame_flux
