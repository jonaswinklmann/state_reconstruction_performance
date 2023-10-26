"""
Transformation generator.

Generates affine transformations between sites and image coordinates.
"""

import numpy as np
import os

from libics.env.logging import get_logger
from libics.core import io
from libics.tools.trafo.linear import AffineTrafo2d

from state_reconstruction import config


###############################################################################
# Defaults
###############################################################################


class TrafoManager:

    """
    Namespace class managing default transformations and phase references.

    Attributes
    ----------
    _phase_ref_image, _phase_ref_site : `(float, float)`
        Lattice phase transformation reference coordinates/sites.
    _trafos_site_to_image : `dict(str->AffineTrafo2d)`
        Maps transformation IDs to transformation objects.
    """

    LOGGER = get_logger("srec.TrafoManager")

    _phase_ref_image = None
    _phase_ref_site = None
    _trafos_site_to_image = {}
    _default_trafo_site_to_image = AffineTrafo2d()

    @classmethod
    def update_config(cls):
        """
        Updates and reloads the transformation objects.
        """
        cls._phase_ref_image = config.get_config("trafo_gen.phase_ref_image")
        cls._phase_ref_site = config.get_config("trafo_gen.phase_ref_site")
        _trafo_fps = config.get_config("trafo_gen.trafo_site_to_image")
        cls._trafos_site_to_image = {}
        for k, fp in _trafo_fps.items():
            if os.path.exists(fp):
                _trafo = io.load(fp)
                if isinstance(_trafo, AffineTrafo2d):
                    cls._trafos_site_to_image[k] = _trafo
                else:
                    cls.LOGGER.error(
                        f"update_config: Trafo `{k}` file is invalid: {fp}"
                    )
            else:
                cls.LOGGER.error(
                    f"update_config: Trafo `{k}` file does not exist: {fp}"
                )

    @classmethod
    def discover_configs(cls):
        return cls._trafos_site_to_image.keys()

    @classmethod
    def get_phase_ref_image(cls):
        return np.copy(cls._phase_ref_image)

    @classmethod
    def get_phase_ref_site(cls):
        return np.copy(cls._phase_ref_site)

    @classmethod
    def get_phase_refs(cls):
        return {
            "phase_ref_image": cls.get_phase_ref_image(),
            "phase_ref_site": cls.get_phase_ref_site()
        }

    @classmethod
    def get_trafo_site_to_image(cls, key=None):
        if key is None:
            return cls._default_trafo_site_to_image.copy()
        else:
            return cls._trafos_site_to_image[key].copy()


TrafoManager.update_config()


###############################################################################
# Trafos and phases
###############################################################################


def get_trafo_site_to_image(
    trafo_site_to_image=None, magnification=None, angle=None,
    phase_ref_image=None, phase_ref_site=None, phase=np.zeros(2)
):
    """
    Gets the default transformation.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d` or `str` or `None`
        Uses the origin axes from this trafo object.
        Is overwritten by `magnification` and `angle`.
        If `str`, loads a trafo from :py:class:`TrafoManager`.
    magnification, angle : `np.ndarray(1, float)` or `None`
        Origin axes magnification and angle.
    phase_ref_image, phase_ref_site : `np.ndarray(1, float)` or `None`
        Phase reference (phase = (0, 0)) in fluorescence image
        and lattice site coordinates.
    phase : `np.ndarray(float, float)`
        Lattice phase (fluo axes in site space = reference + phase).

    Returns
    -------
    trafo : `AffineTrafo2d`
        Site-to-fluorescence-image transformation.
    """
    # Parse parameters
    if not isinstance(trafo_site_to_image, AffineTrafo2d):
        trafo_site_to_image = TrafoManager.get_trafo_site_to_image(
            trafo_site_to_image
        )
    if trafo_site_to_image is not None:
        _m, _a, _ = trafo_site_to_image.get_origin_axes()
    if magnification is None:
        magnification = _m
    if angle is None:
        angle = _a
    if phase_ref_image is None:
        phase_ref_image = TrafoManager.get_phase_ref_image()
    if phase_ref_site is None:
        phase_ref_site = TrafoManager.get_phase_ref_site()
    # Setup trafo
    trafo = AffineTrafo2d()
    trafo.set_origin_axes(magnification=magnification, angle=angle)
    _a, _b, _ = trafo.get_target_axes()
    _offset = phase_ref_site - trafo.coord_to_origin(phase_ref_image)
    trafo.set_target_axes(magnification=_a, angle=_b, offset=_offset + phase)
    return trafo


def get_phase_from_trafo_site_to_image(
    trafo_site_to_image, phase_ref_image=None, phase_ref_site=None
):
    """
    Gets the lattice phase and ref. integer site in `librbl` convention.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d`
        Transformation between lattice sites and fluorescence image.
    phase_ref_image, phase_ref_site : `np.ndarray(1, float)` or `None`
        Phase reference (phase = (0, 0)) in fluorescence image
        and lattice site coordinates.

    Returns
    -------
    phase : `np.ndarray(1, float)`
        Phase (residual) of lattice w.r.t. to image coordinates (0, 0).
    site : `np.ndarray(1, float)`
        Nearest integer lattice site.
    """
    if phase_ref_image is None:
        phase_ref_image = TrafoManager.get_phase_ref_image()
    if phase_ref_site is None:
        phase_ref_site = TrafoManager.get_phase_ref_site()
    site_float = trafo_site_to_image.coord_to_origin(phase_ref_image)
    phase = (site_float - phase_ref_site + 0.5) % 1 - 0.5
    site = np.round(site_float)
    return phase, site
