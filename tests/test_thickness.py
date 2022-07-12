from numpy import Inf, array, isclose
from numpy.ma import masked_where
from pytest import raises

from adcorr import normalize_thickness


def test_normalize_thickness_typical_2x2():
    assert isclose(
        array([[0.5, 1.0], [1.5, 2.0]]),
        normalize_thickness(array([[1.0, 2.0], [3.0, 4.0]]), 2.0),
    ).all()


def test_normalize_thickness_typical_3x3():
    assert isclose(
        array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]]),
        normalize_thickness(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), 2.0
        ),
    ).all()


def test_normalize_thickness_typical_2x2x2():
    assert isclose(
        array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]),
        normalize_thickness(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), 2.0
        ),
    ).all()


def test_normalize_thickness_masked_2x2():
    assert isclose(
        array([[Inf, 1.0], [1.5, Inf]]),
        normalize_thickness(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            2.0,
        ).filled(Inf),
    ).all()


def test_normalize_thickness_thickess_zero():
    with raises(ValueError):
        normalize_thickness(array([[1.0, 2.0], [3.0, 4.0]]), 0.0)


def test_normalize_thickness_thickess_negative():
    with raises(ValueError):
        normalize_thickness(array([[1.0, 2.0], [3.0, 4.0]]), -1.0)


def test_normalize_thickness_thickess_small():
    assert isclose(
        array([[1e6, 2e6], [3e6, 4e6]]),
        normalize_thickness(array([[1.0, 2.0], [3.0, 4.0]]), 1e-6),
    ).all()


def test_normalize_thickness_thickess_large():
    assert isclose(
        array([[1e-6, 2e-6], [3e-6, 4e-6]]),
        normalize_thickness(array([[1.0, 2.0], [3.0, 4.0]]), 1e6),
    ).all()
