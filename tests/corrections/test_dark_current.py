from numpy import Inf, array, isclose
from numpy.ma import masked_where
from pytest import raises

from adcorr.corrections import correct_dark_current


def test_correct_dark_current_typical_2x2():
    assert isclose(
        array(
            [
                [0.88, 1.88],
                [2.88, 3.88],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([0.1]),
            array([10.0]),
            0.1,
            0.1,
            0.001,
        ),
    ).all()


def test_correct_dark_current_typical_3x3():
    assert isclose(
        array(
            [
                [0.88, 1.88, 2.88],
                [3.88, 4.88, 5.88],
                [6.88, 7.88, 8.88],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            array([0.1]),
            array([100.0]),
            0.1,
            0.1,
            0.0001,
        ),
    ).all()


def test_correct_dark_current_typical_2x2x2():
    assert isclose(
        array(
            [
                [
                    [0.88, 1.88],
                    [2.88, 3.88],
                ],
                [
                    [4.79, 5.79],
                    [6.79, 7.79],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.1]),
            array([5.0, 50.0]),
            0.1,
            0.1,
            0.002,
        ),
    ).all()


def test_correct_dark_current_masked_2x2():
    assert isclose(
        array(
            [
                [Inf, 1.88],
                [2.88, Inf],
            ]
        ),
        correct_dark_current(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            array([0.1]),
            array([10.0]),
            0.1,
            0.1,
            0.001,
        ).filled(Inf),
    ).all()


def test_correct_dark_current_count_times_singular():
    assert isclose(
        array(
            [
                [
                    [0.88, 1.88],
                    [2.88, 3.88],
                ],
                [
                    [4.88, 5.88],
                    [6.88, 7.88],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1]),
            array([10.0]),
            0.1,
            0.1,
            0.001,
        ),
    ).all()


def test_correct_dark_current_count_times_vector():
    assert isclose(
        array(
            [
                [
                    [0.88, 1.88],
                    [2.88, 3.88],
                ],
                [
                    [4.87, 5.87],
                    [6.87, 7.87],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.2]),
            array([10.0]),
            0.1,
            0.1,
            0.001,
        ),
    ).all()


def test_correct_dark_current_count_times_zero():
    with raises(ValueError):
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.0]),
            array([10.0]),
            0.1,
            0.1,
            0.001,
        )


def test_correct_dark_current_count_times_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, -0.1]),
            array([10.0]),
            0.1,
            0.1,
            0.001,
        )


def test_correct_dark_current_transmitted_flux_singular():
    assert isclose(
        array(
            [
                [
                    [0.88, 1.88],
                    [2.88, 3.88],
                ],
                [
                    [4.88, 5.88],
                    [6.88, 7.88],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1]),
            array([10.0]),
            0.1,
            0.1,
            0.001,
        ),
    ).all()


def test_correct_dark_current_transmitted_flux_vector():
    assert isclose(
        array(
            [
                [
                    [0.88, 1.88],
                    [2.88, 3.88],
                ],
                [
                    [4.87, 5.87],
                    [6.87, 7.87],
                ],
            ]
        ),
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1]),
            array([10.0, 20.0]),
            0.1,
            0.1,
            0.001,
        ),
    ).all()


def test_correct_dark_current_transmitted_flux_zero():
    with raises(ValueError):
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1]),
            array([0.0]),
            0.1,
            0.1,
            0.001,
        )


def test_correct_dark_current_transmitted_flux_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1]),
            array([10.0, -10.0]),
            0.1,
            0.1,
            0.001,
        )


def test_correct_dark_current_base_dark_current_zero():
    assert isclose(
        array(
            [
                [0.98, 1.98],
                [2.98, 3.98],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([0.1]),
            array([10.0]),
            0.0,
            0.1,
            0.001,
        ),
    ).all()


def test_correct_dark_current_base_dark_current_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([0.1]),
            array([10.0]),
            -0.1,
            0.1,
            0.001,
        )


def test_correct_dark_current_temporal_dark_current_zero():
    assert isclose(
        array(
            [
                [0.89, 1.89],
                [2.89, 3.89],
            ]
        ),
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([0.1]),
            array([10.0]),
            0.1,
            0.0,
            0.001,
        ),
    ).all()


def test_correct_dark_current_temporal_dark_current_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([0.1]),
            array([10.0]),
            0.1,
            -0.1,
            0.001,
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
            array([[1.0, 2.0], [3.0, 4.0]]), array([0.1]), array([10.0]), 0.1, 0.1, 0.0
        ),
    ).all()


def test_correct_dark_current_flux_dependant_dark_current_negative():
    with raises(ValueError):
        correct_dark_current(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([0.1]),
            array([10.0]),
            0.1,
            0.1,
            -0.001,
        )
