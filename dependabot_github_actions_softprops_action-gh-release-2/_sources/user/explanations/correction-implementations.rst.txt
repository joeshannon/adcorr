Correction Implementations
==========================

The mathematical implementations and the academic reference of each of the implemented
corrections are tabulated below for the convenience of those utilizing the package.

In all of the below operations, :math:`I_j` denotes the uncorrected photon count of
detector pixel :math:`j` whilst :math:`I_{j,c}` denotes the corrected photon
count.

.. list-table:: Correction Implementations
    :align: center
    :widths: 20 30 25 25
    :header-rows: 1

    * - Correction
      - Operation
      - Parameters
      - Reference

    * - `correct_deadtime`
      - .. math::

            I_{j,c} &= \frac{W_0(-I_jT)}{T} \\[8pt]
            T &= \frac{\tau_1+\tau_2}{\delta t}

      - - :math:`\delta t`: Count time
        - :math:`\tau_1`: Minimum pulse separation
        - :math:`\tau_2`: Minimum arrival separation
      - `Everything SAXS: small-angle scattering pattern collection and correction`_

    * - `correct_dark_current`
      - .. math::

            I_{j,c} = I_j - D_a - D_b \delta t - D_cI_j

      - - :math:`\delta t`: Count time
        - :math:`D_a`: Base dark current
        - :math:`D_b`: Temporal dark current
        - :math:`D_c`: Flux dependant dark current
      - `Everything SAXS: small-angle scattering pattern collection and correction`_

    * - `normalize_frame_time`
      - .. math::

            I_{j,c} = \frac{I_j}{\delta t}

      - - :math:`\delta t`: Count time
      - `Everything SAXS: small-angle scattering pattern collection and correction`_


    * - `normalize_transmitted_flux`
      - .. math::

            I_{j,c} = \frac{I_j}{\unicode{x222F}{I_j}}

      -
      - `The modular small-angle X-ray scattering data correction sequence`_

    * - `correct_self_absorption`
      - .. math::

            I_{j,c} &= I_j\frac{1-T^{\sec{2\theta}-1}}{\ln{T}(1-\sec{2\theta})} \\[8pt]
            T &= e^{-\mu t} \\[8pt]
            {2\theta}_j &= \arctan{\frac{\sqrt{{\delta x}_j^2 + {\delta y}_j^2}}{\Delta z}}

      - - :math:`\mu`: Absorption coefficient
        - :math:`t`: Sample thickness
        - :math:`{\delta x}_j`: Horizontal pixel beam separation
        - :math:`{\delta y}_j`: Vertical pixel beam separation
        - :math:`\Delta z`: Detector sample separation
      - `Everything SAXS: small-angle scattering pattern collection and correction`_

    * - `average_all_frames`
      - .. math::

            I_{j,c} = \sum{I_{n,j}}

      - - :math:`I_{n,j}`: Intensity of pixel :math:`j` at frame :math:`n`
      - `The modular small-angle X-ray scattering data correction sequence`_

    * - `subtract_background`
      - .. math::

            I_{j,c} = I_j - I_{j,b}

      - - :math:`I_{j,b}`: Background frame intensity of pixel :math:`j`
      - `Everything SAXS: small-angle scattering pattern collection and correction`_

    * - `correct_flatfield`
      - .. math::

            I_{j,c} = I_j F_j

      - - :math:`F_j`: Flatfield correction factor of pixel :math:`j`
      - `The modular small-angle X-ray scattering data correction sequence`_

    * - `correct_angular_efficiency`
      - .. math::

            I_{j,c} &= \frac{I_j}{1-e^{\frac{-\mu t}{\cos{2\theta}}}} \\[8pt]
            {2\theta}_j &= \arctan{\frac{\sqrt{{\delta x}_j^2 + {\delta y}_j^2}}{\Delta z}}

      - - :math:`\mu`: Absorption coefficient
        - :math:`t`: Sample thickness
        - :math:`{\delta x}_j`: Horizontal pixel beam separation
        - :math:`{\delta y}_j`: Vertical pixel beam separation
        - :math:`\Delta z`: Detector sample separation
      - `The modular small-angle X-ray scattering data correction sequence`_

    * - `correct_solid_angle`
      - .. math::

            I_{j,c} &= \frac{I_j}{\cos{2\theta}^3} \\[8pt]
            {2\theta}_j &= \arctan{\frac{\sqrt{{\delta x}_j^2 + {\delta y}_j^2}}{\Delta z}}

      - - :math:`{\delta x}_j`: Horizontal pixel beam separation
        - :math:`{\delta y}_j`: Vertical pixel beam separation
        - :math:`\Delta z`: Detector sample separation
      - `Everything SAXS: small-angle scattering pattern collection and correction`_

    * - `correct_polarization`
      - .. math::

            I_{j,c} &= I_j[PK_v+(1-P)K_h] \\[8pt]
            K_v &= 1-(\sin{\psi}\sin{2\theta})^2 \\[8pt]
            K_h &= 1-(\cos{\psi}\sin{2\theta})^2 \\[8pt]
            {2\theta}_j &= \arctan{\frac{\sqrt{{\delta x}_j^2 + {\delta y}_j^2}}{\Delta z}} \\[8pt]
            \psi &= \arctan{\frac{\delta x}{\delta y}}

      - - :math:`P`: Horizontal polarization
        - :math:`{\delta x}_j`: Horizontal pixel beam separation
        - :math:`{\delta y}_j`: Vertical pixel beam separation
        - :math:`\Delta z`: Detector sample separation
      - `Everything SAXS: small-angle scattering pattern collection and correction`_

    * - `normalize_thickness`
      - .. math::

            I_{j,c} = \frac{I_j}{t}

      - - :math:`t`: Sample thickness
      - `Everything SAXS: small-angle scattering pattern collection and correction`_

    * - `correct_displaced_volume`
      - .. math::

            I_{j,c} = I_j (1 - v_d)

      - - :math:`v_d`: Displaced volume
      - `The modular small-angle X-ray scattering data correction sequence`_

.. _Everything SAXS\: small-angle scattering pattern collection and correction: https://doi.org/10.1088/0953-8984/25/38/383201

.. _The modular small-angle X-ray scattering data correction sequence: https://doi.org/10.1107/S1600576717015096