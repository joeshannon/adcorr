from numpy import Inf, array, isclose
from numpy.ma import masked_where

from adcorr import subtract_background


def test_background_subtraction_typical_2x2():
    assert isclose(
        array([[0.9, 1.8], [2.7, 3.6]]),
        subtract_background(
            array([[1.0, 2.0], [3.0, 4.0]]), array([[0.1, 0.2], [0.3, 0.4]])
        ),
    ).all()


def test_background_subtraction_typical_3x3():
    assert isclose(
        array([[0.9, 1.8, 2.7], [3.6, 4.5, 5.4], [6.3, 7.2, 8.1]]),
        subtract_background(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        ),
    ).all()


def test_background_subtraction_typical_2x2x2():
    assert isclose(
        array([[[0.9, 1.8], [2.7, 3.6]], [[4.9, 5.8], [6.7, 7.6]]]),
        subtract_background(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([[0.1, 0.2], [0.3, 0.4]]),
        ),
    ).all()


def test_correct_deadtime_masked_2x2():
    assert isclose(
        array(
            [
                [Inf, 1.8],
                [2.7, Inf],
            ]
        ),
        subtract_background(
            masked_where(
                array([[True, False], [False, True]]), array([[1.0, 2.0], [3.0, 4.0]])
            ),
            array([[0.1, 0.2], [0.3, 0.4]]),
        ).filled(Inf),
    ).all()
