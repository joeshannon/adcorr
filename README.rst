adcorr
======

|code_ci| |docs_ci| |coverage| |pypi_version| |license|

This package provides a set of pure python functions for performing corrections on area
detector data.

Install via PyPI with:

.. code:: bash 

    pip install adcorr

Useful Links
------------

============== ==============================================
PyPI           https://pypi.org/project/adcorr/
Source code    https://github.com/DiamondLightSource/adcorr
Documentation  https://DiamondLightSource.github.io/adcorr
Releases       https://github.com/DiamondLightSource/adcorr/releases
============== ==============================================

Brief Example
-------------

A brief example of performing corrections using the library is presented below:

.. code:: python

    frames = load_my_frames()
    mask = load_my_mask()
    count_times = load_count_times()

    frames = mask_frames(frames, mask)
    frames = correct_deadtime(
        frames,
        count_times,
        DETECTOR_MINIMUM_PULSE_SEPARATION,
        DETECTOR_MINIMUM_ARRIVAL_SEPARATION,
    )
    frames = correct_dark_current(
        frames,
        count_times,
        BASE_DARK_CURRENT,
        TEMPORAL_DARK_CURRENT,
        FLUX_DEPENDANT_DARK_CURRENT,
    )
    ...

Library Compatibility
---------------------

================================================================ ================== =====================
Library                                                          Tests              Coverage
---------------------------------------------------------------- ------------------ ---------------------
`numcertain <https://github.com/DiamondLightSource/numcertain>`_ |tests_numcertain| |coverage_numcertain|
`Pint <https://pint.readthedocs.io/en/stable/>`_                 |tests_pint|       |coverage_pint|
================================================================ ================== =====================

.. |code_ci| image:: https://github.com/DiamondLightSource/adcorr/workflows/Code%20CI/badge.svg?branch=main
    :target: https://github.com/DiamondLightSource/adcorr/actions?query=workflow%3ACode+branch%3Amain
    :alt: Code CI

.. |docs_ci| image:: https://github.com/DiamondLightSource/adcorr/workflows/Docs%20CI/badge.svg?branch=main
    :target: https://github.com/DiamondLightSource/adcorr/actions?query=workflow%3ACode+branch%3Amain
    :alt: Docs CI

.. |coverage| image:: https://codecov.io/gh/DiamondLightSource/adcorr/branch/main/graph/badge.svg?flag=core
    :target: https://codecov.io/gh/DiamondLightSource/adcorr
    :alt: Test Coverage

.. |tests_numcertain| image:: https://raw.githubusercontent.com/DiamondLightSource/adcorr/gh-badges/main/ubuntu-latest_3.9_false_numcertain.svg
    :target: https://github.com/DiamondLightSource/adcorr/actions?query=workflow%3ACode+branch%3Amain
    :alt: Numcertain Compatibility Test Outcome

.. |coverage_numcertain| image:: https://codecov.io/gh/DiamondLightSource/adcorr/branch/main/graph/badge.svg?flag=numcertain
    :target: https://codecov.io/gh/DiamondLightSource/adcorr
    :alt: Numcertain Compatibility Test Coverage

.. |tests_pint| image:: https://raw.githubusercontent.com/DiamondLightSource/adcorr/gh-badges/main/ubuntu-latest_3.9_false_pint.svg
    :target: https://github.com/DiamondLightSource/adcorr/actions?query=workflow%3ACode+branch%3Amain
    :alt: Pint Compatibility Test Outcome

.. |coverage_pint| image:: https://codecov.io/gh/DiamondLightSource/adcorr/branch/main/graph/badge.svg?flag=pint
    :target: https://codecov.io/gh/DiamondLightSource/adcorr
    :alt: Pint Compatibility Test Coverage

.. |pypi_version| image:: https://img.shields.io/pypi/v/adcorr.svg
    :target: https://pypi.org/project/adcorr
    :alt: Latest PyPI version

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: Apache License

..
    Anything below this line is used when viewing README.rst and will be replaced
    when included in index.rst

See https://DiamondLightSource.github.io/adcorr for more detailed documentation.
