from numpy import array, isclose

from adcorr.utils.geometry import azimuthal_angles, scattering_angles


def test_scattering_angles_typical_2x2():
    assert isclose(
        array([[0.0705931793, 0.0705931793], [0.0705931793, 0.0705931793]]),
        scattering_angles((2, 2), (1.0, 1.0), (0.1, 0.1), 1.0),
    ).all()


def test_scattering_angles_typical_3x3():
    assert isclose(
        array(
            [
                [0.140489702, 0.0996686525, 0.140489702],
                [0.0996686525, 0.0, 0.0996686525],
                [0.140489702, 0.0996686525, 0.140489702],
            ]
        ),
        scattering_angles((3, 3), (1.5, 1.5), (0.1, 0.1), 1.0),
    ).all()


def test_scattering_angles_typical_3x2():
    assert isclose(
        array(
            [
                [0.111341014, 0.111341014],
                [0.0499583957, 0.0499583957],
                [0.111341014, 0.111341014],
            ]
        ),
        scattering_angles((3, 2), (1.5, 1.0), (0.1, 0.1), 1.0),
    ).all()


def test_scattering_angles_zero_pixels():
    assert 0 == scattering_angles((0, 0), (1.0, 1.0), (0.1, 0.1), 1.0).size


def test_scattering_angles_center_corner():
    assert isclose(
        array([[0.209033299, 0.156815685], [0.156815685, 0.0705931793]]),
        scattering_angles((2, 2), (2.0, 2.0), (0.1, 0.1), 1.0),
    ).all()


def test_scattering_angles_center_outside():
    assert isclose(
        array([[0.140489702, 0.219987977], [0.219987977, 0.275642799]]),
        scattering_angles((2, 2), (-0.5, -0.5), (0.1, 0.1), 1.0),
    ).all()


def test_scattering_angles_pixels_small():
    assert isclose(
        array([[7.07106781e-10, 7.07106781e-10], [7.07106781e-10, 7.07106781e-10]]),
        scattering_angles((2, 2), (1.0, 1.0), (1e-9, 1e-9), 1.0),
    ).all()


def test_scattering_angles_pixels_large():
    assert isclose(
        array([[1.57079633, 1.57079633], [1.57079633, 1.57079633]]),
        scattering_angles((2, 2), (1.0, 1.0), (1e9, 1e9), 1.0),
    ).all()


def test_scattering_angles_pixels_rectangular():
    assert isclose(
        array([[0.111341014, 0.111341014], [0.111341014, 0.111341014]]),
        scattering_angles((2, 2), (1.0, 1.0), (0.1, 0.2), 1.0),
    ).all()


def test_scattering_angles_distance_small():
    assert isclose(
        array([[1.57079631, 1.57079631], [1.57079631, 1.57079631]]),
        scattering_angles((2, 2), (1.0, 1.0), (0.1, 0.1), 1e-9),
    ).all()


def test_scattering_angles_distance_large():
    assert isclose(
        array([[7.07106781e-11, 7.07106781e-11], [7.07106781e-11, 7.07106781e-11]]),
        scattering_angles((2, 2), (1.0, 1.0), (0.1, 0.1), 1e9),
    ).all()


def test_scattering_angles_distance_negative():
    assert isclose(
        array([[-0.0705931793, -0.0705931793], [-0.0705931793, -0.0705931793]]),
        scattering_angles((2, 2), (1.0, 1.0), (0.1, 0.1), -1.0),
    ).all()


def test_azimuthal_angles_typical_2x2():
    assert isclose(
        array([[0.785398163, -0.785398163], [-0.785398163, 0.785398163]]),
        azimuthal_angles((2, 2), (1.0, 1.0), (0.1, 0.1)),
    ).all()


def test_azimuthal_angles_typical_3x3():
    assert isclose(
        array(
            [
                [0.78539816, 1.57079633, -0.78539816],
                [0, 0, 0],
                [-0.78539816, 1.57079633, 0.78539816],
            ]
        ),
        azimuthal_angles((3, 3), (1.5, 1.5), (0.1, 0.1)),
    ).all()


def test_azimuthal_angles_typical_3x2():
    assert isclose(
        array([[1.10714872, -1.10714872], [0, 0], [-1.10714872, 1.10714872]]),
        azimuthal_angles((3, 2), (1.5, 1.0), (0.1, 0.1)),
    ).all()


def test_azimuthal_angles_zero_pixels():
    assert 0 == azimuthal_angles((0, 0), (1.0, 1.0), (0.1, 0.1)).size


def test_azimuthal_angles_center_corner():
    assert isclose(
        array([[0.785398163, 1.24904577], [0.321750554, 0.785398163]]),
        azimuthal_angles((2, 2), (2.0, 2.0), (0.1, 0.1)),
    ).all()


def test_azimuthal_angles_center_outside():
    assert isclose(
        array([[0.785398163, 0.463647609], [1.10714872, 0.785398163]]),
        azimuthal_angles((2, 2), (-0.5, -0.5), (0.1, 0.1)),
    ).all()


def test_azimuthal_angles_pixels_small():
    assert isclose(
        array([[0.785398163, -0.785398163], [-0.785398163, 0.785398163]]),
        azimuthal_angles((2, 2), (1.0, 1.0), (1e-9, 1e-9)),
    ).all()


def test_azimuthal_angles_pixels_large():
    assert isclose(
        array([[0.785398163, -0.785398163], [-0.785398163, 0.785398163]]),
        azimuthal_angles((2, 2), (1.0, 1.0), (1e9, 1e9)),
    ).all()


def test_azimuthal_angles_pixels_rectangular():
    assert isclose(
        array([[0.463647609, -0.463647609], [-0.463647609, 0.463647609]]),
        azimuthal_angles((2, 2), (1.0, 1.0), (0.1, 0.2)),
    ).all()
