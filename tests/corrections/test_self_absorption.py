from unittest.mock import MagicMock, patch

from numpy import Inf, array, isclose
from numpy.ma import masked_where
from pytest import raises

from adcorr.corrections import correct_self_absorption

from ..inaccessable_mock import AccessedError, inaccessable_mock


def test_correct_self_absorption_typical_2x2():
    assert isclose(
        array([[0.999988, 1.999980], [2.999960, 3.999950]]),
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]), (1.0, 1.0), (0.1, 0.1), 1.0, 0.1, 0.1
        ),
    ).all()


def test_correct_self_absorption_typical_3x3():
    assert isclose(
        array(
            [
                [0.99995, 1.99995, 2.99985],
                [3.99990, 5.0, 5.99985],
                [6.99965, 7.99980, 8.99955],
            ]
        ),
        correct_self_absorption(
            array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            (1.5, 1.5),
            (0.1, 0.1),
            1.0,
            0.1,
            0.1,
        ),
    ).all()


def test_correct_self_absorption_typical_2x2x2():
    assert isclose(
        array(
            [
                [[0.99999, 1.99998], [2.99996, 3.99995]],
                [[4.99994, 5.99993], [6.99991, 7.99990]],
            ]
        ),
        correct_self_absorption(
            array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            (1.0, 1.0),
            (0.1, 0.1),
            1.0,
            0.1,
            0.1,
        ),
    ).all()


def test_correct_self_absorption_masked_2x2():
    assert isclose(
        array(
            [
                [Inf, 1.999980],
                [2.999960, Inf],
            ]
        ),
        correct_self_absorption(
            masked_where(
                array([[True, False], [False, True]]),
                array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            (1.0, 1.0),
            (0.1, 0.1),
            1.0,
            0.1,
            0.1,
        ).filled(Inf),
    ).all()


def test_correct_self_absorption_passes_beam_center_to_scattering_angles_only():
    with raises(AccessedError):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]),
            inaccessable_mock(tuple[float, float]),
            (0.1, 0.1),
            1.0,
            0.1,
            0.1,
        )
    with patch(
        "adcorr.corrections.self_absorption.scattering_angles",
        MagicMock(return_value=array([[0.5, 0.5], [0.5, 0.5]])),
    ):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]),
            inaccessable_mock(tuple[float, float]),
            (0.1, 0.1),
            1.0,
            0.1,
            0.1,
        )


def test_correct_self_absorption_passes_pixel_sizes_to_scattering_angles_only():
    with raises(AccessedError):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]),
            (1.0, 1.0),
            inaccessable_mock(tuple[float, float]),
            1.0,
            0.1,
            0.1,
        )
    with patch(
        "adcorr.corrections.self_absorption.scattering_angles",
        MagicMock(return_value=array([[0.5, 0.5], [0.5, 0.5]])),
    ):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]),
            (1.0, 1.0),
            inaccessable_mock(tuple[float, float]),
            1.0,
            0.1,
            0.1,
        )


def test_correct_self_absorption_passes_distance_to_scattering_angles_only():
    with raises(AccessedError):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]),
            (1.0, 1.0),
            (0.1, 0.1),
            inaccessable_mock(float),
            0.1,
            0.1,
        )
    with patch(
        "adcorr.corrections.self_absorption.scattering_angles",
        MagicMock(return_value=array([[0.5, 0.5], [0.5, 0.5]])),
    ):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]),
            (1.0, 1.0),
            (0.1, 0.1),
            inaccessable_mock(float),
            0.1,
            0.1,
        )


def test_correct_self_absorption_absorption_coefficient_zero():
    assert isclose(
        array([[1.0, 2.0], [3.0, 4.0]]),
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]), (1.0, 1.0), (0.1, 0.1), 1.0, 0.0, 0.1
        ),
    ).all()


def test_correct_self_absorption_absorption_coefficient_negative():
    with raises(ValueError):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]), (1.0, 1.0), (0.1, 0.1), 1.0, -0.5, 0.1
        )


def test_correct_self_absorption_thickness_zero():
    assert isclose(
        array([[1.0, 2.0], [3.0, 4.0]]),
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]), (1.0, 1.0), (0.1, 0.1), 1.0, 0.1, 0.0
        ),
    ).all()


def test_correct_self_absorption_thickness_negative():
    with raises(ValueError):
        correct_self_absorption(
            array([[1.0, 2.0], [3.0, 4.0]]), (1.0, 1.0), (0.1, 0.1), 1.0, 0.1, -0.5
        )
