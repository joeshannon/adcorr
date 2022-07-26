from numpy import Inf, array, isclose
from numpy.ma import masked_where
from pytest import raises

from adcorr.corrections import normalize_frame_time


def test_normalize_frame_time_typical_2x2():
    assert isclose(
        array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        ),
        normalize_frame_time(array([[1.0, 2.0], [3.0, 4.0]]), array([0.1])),
    ).all()


def test_normalize_frame_time_typical_3x3():
    assert isclose(
        array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0],
            ]
        ),
        normalize_frame_time(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), array([0.1])
        ),
    ).all()


def test_normalize_frame_time_typical_2x2x2():
    assert isclose(
        array(
            [
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ],
                [
                    [50.0, 60.0],
                    [70.0, 80.0],
                ],
            ]
        ),
        normalize_frame_time(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.1]),
        ),
    ).all()


def test_normalize_frame_time_masked_2x2():
    assert isclose(
        array(
            [
                [Inf, 20.0],
                [30.0, Inf],
            ]
        ),
        normalize_frame_time(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            array([0.1]),
        ).filled(Inf),
    ).all()


def test_normalize_frame_time_count_times_singular():
    assert isclose(
        array(
            [
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ],
                [
                    [50.0, 60.0],
                    [70.0, 80.0],
                ],
            ]
        ),
        normalize_frame_time(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), array([0.1])
        ),
    ).all()


def test_normalize_frame_time_count_times_vector():
    assert isclose(
        array(
            [
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ],
                [
                    [25.0, 30.0],
                    [35.0, 40.0],
                ],
            ]
        ),
        normalize_frame_time(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, 0.2]),
        ),
    ).all()


def test_normalize_frame_time_count_times_zero():
    with raises(ValueError):
        normalize_frame_time(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), array([0.0])
        )


def test_normalize_frame_time_count_times_negative():
    with raises(ValueError):
        normalize_frame_time(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([0.1, -0.1]),
        )
