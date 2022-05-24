"""
Transformation generator.

Generates affine transformations between sites and image coordinates.
"""

import numpy as np

from libics.tools.trafo.linear import AffineTrafo2d


###############################################################################


def get_trafo_site_to_image(
    magnification=None, angle=None, trafo_site_to_image=None,
    phase_ref_image=(0, 0), phase_ref_site=(169, 84), phase=np.zeros(2)
):
    """
    Gets the default transformation.

    Parameters
    ----------
    magnification, angle : `np.ndarray(1, float)` or `None`
        Origin axes magnification and angle.
    trafo_site_to_image : `AffineTrafo2d` or `None`
        Uses the origin axes from this trafo object.
        Is overwritten by `magnification` and `angle`.
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
    if trafo_site_to_image is not None:
        _m, _a, _ = trafo_site_to_image.get_origin_axes()
    if magnification is None:
        magnification = _m
    if angle is None:
        angle = _a
    # Setup trafo
    trafo = AffineTrafo2d()
    trafo.set_origin_axes(magnification=magnification, angle=angle)
    _a, _b, _ = trafo.get_target_axes()
    _offset = phase_ref_site - trafo.coord_to_origin(phase_ref_image)
    trafo.set_target_axes(magnification=_a, angle=_b, offset=_offset + phase)
    return trafo


def get_phase_from_trafo_site_to_image(
    trafo_site_to_image, phase_ref_image=(0, 0)
):
    """
    Gets the lattice phase and ref. integer site in `librbl` convention.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d`
        Transformation between lattice sites and fluorescence image.
    phase_ref_fluo : `np.ndarray(1, float)`
        Fluorescence image coordinate corresponding to integer lattice site.

    Returns
    -------
    phase : `np.ndarray(1, float)`
        Phase (residual) of lattice w.r.t. to image coordinates (0, 0).
    site : `np.ndarray(1, float)`
        Nearest integer lattice site.
    """
    site_float = trafo_site_to_image.coord_to_origin(phase_ref_image)
    phase = (site_float + 0.5) % 1 - 0.5
    site = np.round(site_float)
    return phase, site
