from numpy import Inf, array, isclose
from numpy.ma import masked_where
from pytest import raises

from adcorr import correct_dark_current


def test_correct_dark_current_typical_2x2():
    assert isclose(
        array(
            [
                [0.79, 1.69],
                [2.59, 3.49],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.1, 0.1, 0.1
        ),
    ).all()


def test_correct_dark_current_typical_3x3():
    assert isclose(
        array(
            [
                [0.79, 1.69, 2.59],
                [3.49, 4.39, 5.29],
                [6.19, 7.09, 7.99],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            array([0.1]),
            0.1,
            0.1,
            0.1,
        ),
    ).all()


def test_correct_dark_current_typical_2x2x2():
    assert isclose(
        array(
            [
                [
                    [0.79, 1.69],
                    [2.59, 3.49],
                ],
                [
                    [4.39, 5.29],
                    [6.19, 7.09],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.1]),
            0.1,
            0.1,
            0.1,
        ),
    ).all()


def test_correct_dark_current_masked_2x2():
    assert isclose(
        array(
            [
                [Inf, 1.69],
                [2.59, Inf],
            ]
        ),
        correct_dark_current(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            array([0.1]),
            0.1,
            0.1,
            0.1,
        ).filled(Inf),
    ).all()


def test_correct_dark_current_count_times_singular():
    assert isclose(
        array(
            [
                [
                    [0.79, 1.69],
                    [2.59, 3.49],
                ],
                [
                    [4.39, 5.29],
                    [6.19, 7.09],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1]),
            0.1,
            0.1,
            0.1,
        ),
    ).all()


def test_correct_dark_current_count_times_vector():
    assert isclose(
        array(
            [
                [
                    [0.79, 1.69],
                    [2.59, 3.49],
                ],
                [
                    [4.38, 5.28],
                    [6.18, 7.08],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.2]),
            0.1,
            0.1,
            0.1,
        ),
    ).all()


def test_correct_dark_current_count_times_zero():
    with raises(ValueError):
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.0]),
            0.1,
            0.1,
            0.1,
        )


def test_correct_dark_current_count_times_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, -0.1]),
            0.1,
            0.1,
            0.1,
        )


def test_correct_dark_current_base_dark_current_zero():
    assert isclose(
        array(
            [
                [0.89, 1.79],
                [2.69, 3.59],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.0, 0.1, 0.1
        ),
    ).all()


def test_correct_dark_current_base_dark_current_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), -0.1, 0.1, 0.1
        )


def test_correct_dark_current_temporal_dark_current_zero():
    assert isclose(
        array(
            [
                [0.80, 1.70],
                [2.60, 3.50],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.1, 0.0, 0.1
        ),
    ).all()


def test_correct_dark_current_temporal_dark_current_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.1, -0.1, 0.1
        )


def test_correct_dark_current_flux_dependant_dark_current_zero():
    assert isclose(
        array(
            [
                [0.89, 1.89],
                [2.89, 3.89],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.1, 0.1, 0.0
        ),
    ).all()


def test_correct_dark_current_flux_dependant_dark_current_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.1, 0.1, -0.1
        )
