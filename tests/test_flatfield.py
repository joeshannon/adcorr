from numpy import Inf, array, isclose
from numpy.ma import masked_where

from adcorr import correct_flatfield


def test_correct_flatfield_typical_2x2():
    assert isclose(
        array([[1.0, 4.0], [9.0, 16.0]]),
        correct_flatfield(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([[1.0, 2.0], [3.0, 4.0]]),
        ),
    ).all()


def test_correct_flatfield_typical_3x3():
    assert isclose(
        array(
            [
                [1.0, 4.0, 9.0],
                [16.0, 25.0, 36.0],
                [49.0, 64.0, 81.0],
            ]
        ),
        correct_flatfield(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ),
    ).all()


def test_correct_flatfield_typical_2x2x2():
    assert isclose(
        array(
            [
                [[1.0, 4.0], [9.0, 16.0]],
                [[25.0, 36.0], [49.0, 64.0]],
            ]
        ),
        correct_flatfield(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        ),
    ).all()


def test_correct_flatfield_masked_2x2():
    assert isclose(
        array([[Inf, 4.0], [9.0, Inf]]),
        correct_flatfield(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            array([[1.0, 2.0], [3.0, 4.0]]),
        ).filled(Inf),
    ).all()
