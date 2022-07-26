from numpy import Inf, array, isclose
from numpy.ma import masked_where

from adcorr.corrections import normalize_transmitted_flux


def test_normalize_transmitted_flux_typical_2x2():
    assert isclose(
        array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        ),
        normalize_transmitted_flux(array([[1.0, 2.0], [3.0, 4.0]])),
    ).all()


def test_normalize_transmitted_flux_typical_3x3():
    assert isclose(
        array(
            [
                [0.022222, 0.044444, 0.066666],
                [0.088888, 0.111111, 0.133333],
                [0.155555, 0.177777, 0.200000],
            ]
        ),
        normalize_transmitted_flux(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        ),
    ).all()


def test_normalize_transmitted_flux_typical_2x2x2():
    assert isclose(
        array(
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
                [
                    [0.192308, 0.230769],
                    [0.269231, 0.307692],
                ],
            ]
        ),
        normalize_transmitted_flux(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ),
    ).all()


def test_normalize_transmitted_flux_masked_2x2():
    assert isclose(
        array(
            [
                [Inf, 0.4],
                [0.6, Inf],
            ]
        ),
        normalize_transmitted_flux(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            )
        ).filled(Inf),
    ).all()
