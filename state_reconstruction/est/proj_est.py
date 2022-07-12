"""
Projection estimator.

Projects an image into site space.
"""


import numba as nb
import numpy as np


###############################################################################
# Projector application
###############################################################################


@nb.njit(
    nb.void(nb.float64[:, :, :], nb.float64[:, :],
            nb.int32[:], nb.int32[:], nb.int32[:], nb.int32[:]),
    parallel=True, cache=True
)
def _slice_local_image(local_images, image, X_min, X_max, Y_min, Y_max):
    for i in nb.prange(len(local_images)):
        local_images[i] = image[X_min[i]:X_max[i], Y_min[i]:Y_max[i]]


def get_local_images(X, Y, image, shape, psf_supersample=5):
    """
    Extracts image subregions and subpixel shifts.

    Parameters
    ----------
    X, Y : `Array[1, float]`
        (Fractional) center positions in image coordinates.
    image : `Array[2, float]`
        Full-sized image.
    shape : `(int, int)`
        Shape of subregion images.
    psf_supersample : `int`
        Supersampling size
        (used to convert fractional positions to subpixel shifts).

    Returns
    -------
    ret : `dict(str->Any)`
        Returns a dictionary containing the following items:
    image : `np.ndarray(3, float)`
        Subregion images. Dimension: `[n_subregions, {shape}]`.
    X_int, Y_int : `np.ndarray(1, int)`
        Rounded PSF center coordinates.
    X_min, X_max, Y_min, Y_max : `np.ndarray(1, int)`
        Rounded PSF rectangle corners.
    dx, dy : `np.ndarray(1, int)`
        Subpixel shifts.
    """
    X_int, Y_int = np.round(X).astype(int), np.round(Y).astype(int)
    dx = np.round((X - X_int) * psf_supersample).astype(int)
    dy = np.round((Y - Y_int) * psf_supersample).astype(int)
    X_min, Y_min = X_int - shape[0] // 2, Y_int - shape[1] // 2
    X_max, Y_max = X_min + shape[0], Y_min + shape[1]
    local_images = np.zeros((len(X),) + tuple(shape), dtype=float)
    image = np.array(image)
    _slice_local_image(local_images, image, X_min, X_max, Y_min, Y_max)
    return {
        "image": local_images, "X_int": X_int, "Y_int": Y_int,
        "X_min": X_min, "X_max": X_max, "Y_min": Y_min, "Y_max": Y_max,
        "dx": dx, "dy": dy
    }


def apply_projectors(local_images, projector_generator):
    """
    Applies subpixel-shifted projectors to subregion images.

    Parameters
    ----------
    local_images : `np.ndarray(3, float)`
        Subregion images. Dimension: `[n_subregions, {shape}]`.
    projector_generator : `ProjectorGenerator`
        Projector generator object.

    Returns
    -------
    emissions : `np.ndarray(1, float)`
        Projected results. Dimensions: `[n_subregions]`.
    """
    images = local_images["image"]
    images = images.reshape((len(images), -1))
    if not projector_generator.proj_cache_built:
        projector_generator.setup_cache()
    projs = projector_generator.proj_cache
    projs = projs.reshape((projs.shape[:2]) + (-1,))
    xidx = local_images["dx"] % projector_generator.psf_supersample
    yidx = local_images["dy"] % projector_generator.psf_supersample
    emissions = [
        _im @ projs[_x, _y]
        for _im, _x, _y in zip(images, xidx, yidx)
    ]
    return np.array(emissions)
