from numpy import Inf, array, isclose
from numpy.ma import masked_where
from pytest import raises

from adcorr import correct_deadtime


def test_correct_deadtime_typical_2x2():
    assert isclose(
        array(
            [
                [1.00005, 2.00020],
                [3.00045, 4.00080],
            ]
        ),
        correct_deadtime(array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 3e-6, 2e-6),
    ).all()


def test_correct_deadtime_typical_3x3():
    assert isclose(
        array(
            [
                [1.00005, 2.00020, 3.00045],
                [4.00080, 5.00125, 6.00180],
                [7.00245, 8.00320, 9.00405],
            ]
        ),
        correct_deadtime(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            array([0.1]),
            3e-6,
            2e-6,
        ),
    ).all()


def test_correct_deadtime_typical_2x2x2():
    assert isclose(
        array(
            [
                [
                    [1.00005, 2.00020],
                    [3.00045, 4.00080],
                ],
                [
                    [5.00125, 6.00180],
                    [7.00245, 8.00320],
                ],
            ]
        ),
        correct_deadtime(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.1]),
            3e-6,
            2e-6,
        ),
    ).all()


def test_correct_deadtime_masked_2x2():
    assert isclose(
        array(
            [
                [Inf, 2.00020],
                [3.00045, Inf],
            ]
        ),
        correct_deadtime(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            array([0.1]),
            3e-6,
            2e-6,
        ).filled(Inf),
    ).all()


def test_correct_deadtime_count_times_singular():
    assert isclose(
        array(
            [
                [
                    [1.00005, 2.00020],
                    [3.00045, 4.00080],
                ],
                [
                    [5.00125, 6.00180],
                    [7.00245, 8.00320],
                ],
            ]
        ),
        correct_deadtime(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1]),
            3e-6,
            2e-6,
        ),
    ).all()


def test_correct_deadtime_count_times_vector():
    assert isclose(
        array(
            [
                [
                    [1.00005, 2.00020],
                    [3.00045, 4.00080],
                ],
                [
                    [5.00063, 6.00090],
                    [7.00123, 8.00160],
                ],
            ]
        ),
        correct_deadtime(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.2]),
            3e-6,
            2e-6,
        ),
    ).all()


def test_correct_deadtime_count_times_zero():
    with raises(ValueError):
        correct_deadtime(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.0]),
            3e-6,
            2e-6,
        )


def test_correct_deadtime_count_times_negative():
    with raises(ValueError):
        correct_deadtime(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, -0.1]),
            3e-6,
            2e-6,
        )


def test_correct_deadtime_minimum_pulse_separation_zero():
    assert isclose(
        array(
            [
                [1.00002, 2.00008],
                [3.00018, 4.00032],
            ]
        ),
        correct_deadtime(array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.0, 2e-6),
    ).all()


def test_correct_deadtime_minimum_pulse_separation_negative():
    with raises(ValueError):
        correct_deadtime(array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), -3e-6, 2e-6)


def test_correct_deadtime_minimum_arrival_separation_zero():
    assert isclose(
        array(
            [
                [1.00002, 2.00008],
                [3.00018, 4.00032],
            ]
        ),
        correct_deadtime(array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 2e-6, 0.0),
    ).all()


def test_correct_deadtime_minimum_arrival_separation_negative():
    with raises(ValueError):
        correct_deadtime(array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 3e-6, -2e-6)


def test_correct_deadtime_minimum_separations_zero():
    assert isclose(
        array([[1.0, 2.0], [3.0, 4.0]]),
        correct_deadtime(array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), 0.0, 0.0),
    ).all()
