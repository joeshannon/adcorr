from numpy import Inf, array, isclose
from numpy.ma import masked_where
from pytest import raises

from adcorr import correct_displaced_volume


def test_correct_displaced_volume_typical_2x2():
    assert isclose(
        array([[0.5, 1.0], [1.5, 2.0]]),
        correct_displaced_volume(array([[1.0, 2.0], [3.0, 4.0]]), 0.5),
    ).all()


def test_correct_displaced_volume_typical_3x3():
    assert isclose(
        array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]]),
        correct_displaced_volume(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), 0.5
        ),
    ).all()


def test_correct_displaced_volume_typical_2x2x2():
    assert isclose(
        array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]),
        correct_displaced_volume(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), 0.5
        ),
    ).all()


def test_correct_displaced_volume_masked_2x2():
    assert isclose(
        array([[Inf, 1.0], [1.5, Inf]]),
        correct_displaced_volume(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            0.5,
        ).filled(Inf),
    ).all()


def test_correct_displaced_volume_thickess_zero():
    assert isclose(
        array([[1.0, 2.0], [3.0, 4.0]]),
        correct_displaced_volume(array([[1.0, 2.0], [3.0, 4.0]]), 0.0),
    ).all()


def test_correct_displaced_volume_thickess_negative():
    with raises(ValueError):
        correct_displaced_volume(array([[1.0, 2.0], [3.0, 4.0]]), -1.0)


def test_correct_displaced_volume_thickess_one():
    assert isclose(
        array([[0.0, 0.0], [0.0, 0.0]]),
        correct_displaced_volume(array([[1.0, 2.0], [3.0, 4.0]]), 1.0),
    ).all()


def test_correct_displaced_volume_thickess_greater_one():
    with raises(ValueError):
        correct_displaced_volume(array([[1.0, 2.0], [3.0, 4.0]]), 1.1)


def test_correct_displaced_volume_thickess_small():
    assert isclose(
        array([[0.9999, 1.9998], [2.9997, 3.9996]]),
        correct_displaced_volume(array([[1.0, 2.0], [3.0, 4.0]]), 0.0001),
    ).all()


def test_correct_displaced_volume_thickess_large():
    assert isclose(
        array([[0.0001, 0.0002], [0.0003, 0.0004]]),
        correct_displaced_volume(array([[1.0, 2.0], [3.0, 4.0]]), 0.9999),
    ).all()
