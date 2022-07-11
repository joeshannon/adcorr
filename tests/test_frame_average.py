from numpy import Inf, array, isclose
from numpy.ma import masked_where

from adcorr import average_all_frames


def test_average_all_frames_typical_2x2():
    assert isclose(
        array([[1.0, 2.0], [3.0, 4.0]]),
        average_all_frames(array([[1.0, 2.0], [3.0, 4.0]])),
    ).all()


def test_average_all_frames_typical_3x3():
    assert isclose(
        array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        average_all_frames(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ),
    ).all()


def test_average_all_frames_typical_2x2x2():
    assert isclose(
        array([[6.0, 8.0], [10.0, 12.0]]),
        average_all_frames(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        ),
    ).all()


def test_average_all_frames_typical_3x2x2():
    assert isclose(
        array([[15.0, 18.0], [21.0, 24.0]]),
        average_all_frames(
            array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                ]
            ),
        ),
    ).all()


def test_average_all_frames_masked_2x2():
    assert isclose(
        array([[Inf, 2.0], [3.0, Inf]]),
        average_all_frames(
            masked_where(
                array([[True, False], [False, True]]), array([[1.0, 2.0], [3.0, 4.0]])
            )
        ),
    ).all()


def test_average_all_frames_masked_2x2x2():
    assert isclose(
        array([[Inf, 8.0], [10.0, Inf]]),
        average_all_frames(
            masked_where(
                array([[[True, False], [False, True]], [[True, False], [False, True]]]),
                array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            )
        ).filled(Inf),
    ).all()


def test_average_all_frames_masked_diag_2x2x2():
    assert isclose(
        array([[5.0, 2.0], [3.0, 8.0]]),
        average_all_frames(
            masked_where(
                array([[[True, False], [False, True]], [[False, True], [True, False]]]),
                array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            )
        ).filled(Inf),
    ).all()
