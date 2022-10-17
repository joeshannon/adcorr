import pytest
from numpy import Inf, array
from pytest import raises

from adcorr.corrections.masking import mask_frames

from ..compat import numcertain


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


@pytest.mark.usefixtures(numcertain.__name__)
def test_masking_numcertain(numcertain):
    assert (
        array(
            [
                [
                    numcertain.uncertain(Inf, 0.0),
                    numcertain.uncertain(2.0, 0.1),
                ],
                [
                    numcertain.uncertain(3.0, 0.1),
                    numcertain.uncertain(Inf, 0.0),
                ],
            ]
        )
        == mask_frames(
            array(
                [
                    [
                        numcertain.uncertain(1.0, 0.1),
                        numcertain.uncertain(2.0, 0.1),
                    ],
                    [
                        numcertain.uncertain(3.0, 0.1),
                        numcertain.uncertain(4.0, 0.1),
                    ],
                ]
            ),
            array([[True, False], [False, True]]),
        ).filled(numcertain.uncertain(Inf, 0.0))
    ).all()
