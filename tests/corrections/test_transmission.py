from numpy import Inf, allclose, array
from numpy.ma import masked_where

from adcorr.corrections import normalize_transmitted_flux


def test_normalize_transmitted_flux_typical_2x2():
    assert allclose(
        array([[0.1, 0.2], [0.3, 0.4]]),
        normalize_transmitted_flux(array([[1.0, 2.0], [3.0, 4.0]]), array([10.0])),
    )


def test_normalize_transmitted_flux_typical_3x3():
    assert allclose(
        array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        normalize_transmitted_flux(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), array([10.0])
        ),
    )


def test_normalize_transmitted_flux_typical_2x2x2():
    assert allclose(
        array([[[1.0, 2.0], [3.0, 4.0]], [[2.5, 3.0], [3.5, 4.0]]]),
        normalize_transmitted_flux(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([1.0, 2.0]),
        ),
    )


def test_normalize_transmitted_flux_masked_2x2():
    assert allclose(
        array([[Inf, 2.0], [3.0, Inf]]),
        normalize_transmitted_flux(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            array([1.0]),
        ).filled(Inf),
    )
