from numpy import Inf, array
from pytest import raises

from adcorr.corrections.masking import mask_frames


def test_masking_typical_3x3():
    assert (
        array([[Inf, 2.0], [3.0, Inf]])
        == mask_frames(
            array([[1.0, 2.0], [3.0, 4.0]]), array([[True, False], [False, True]])
        ).filled(Inf)
    ).all()


def test_masking_typical_2x3x3():
    assert (
        array([[[Inf, 2.0], [3.0, Inf]], [[Inf, 6.0], [7.0, Inf]]])
        == mask_frames(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            array([[True, False], [False, True]]),
        ).filled(Inf)
    ).all()


def test_masking_non_broadcastable_mask_raises():
    with raises(ValueError):
        mask_frames(
            array([[1.0, 2.0], [3.0, 4.0]]),
            array([[True, False, True], [False, True, False]]),
        )
