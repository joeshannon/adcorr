from numpy import Inf, allclose, array
from numpy.ma import masked_where

from adcorr.corrections import average_all_frames


def test_average_all_frames_typical_2x2():
    assert allclose(
        array([[1.0, 2.0], [3.0, 4.0]]),
        average_all_frames(array([[1.0, 2.0], [3.0, 4.0]])),
    )


def test_average_all_frames_typical_3x3():
    assert allclose(
        array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        average_all_frames(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ),
    )


def test_average_all_frames_typical_2x2x2():
    assert allclose(
        array([[3.0, 4.0], [5.0, 6.0]]),
        average_all_frames(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        ),
    )


def test_average_all_frames_typical_3x2x2():
    assert allclose(
        array([[5.0, 6.0], [7.0, 8.0]]),
        average_all_frames(
            array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                ]
            ),
        ),
    )


def test_average_all_frames_masked_2x2():
    assert allclose(
        array([[Inf, 2.0], [3.0, Inf]]),
        average_all_frames(
            masked_where(
                array([[True, False], [False, True]]), array([[1.0, 2.0], [3.0, 4.0]])
            )
        ),
    )


def test_average_all_frames_masked_2x2x2():
    assert allclose(
        array([[Inf, 4.0], [5.0, Inf]]),
        average_all_frames(
            masked_where(
                array([[[True, False], [False, True]], [[True, False], [False, True]]]),
                array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            )
        ).filled(Inf),
    )


def test_average_all_frames_masked_diag_2x2x2():
    assert allclose(
        array([[5.0, 2.0], [3.0, 8.0]]),
        average_all_frames(
            masked_where(
                array([[[True, False], [False, True]], [[False, True], [True, False]]]),
                array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            )
        ).filled(Inf),
    )
