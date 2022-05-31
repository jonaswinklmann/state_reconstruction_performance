"""
Projector generator.

Calculates the projectors to map images to lattice sites.
"""

import copy
import numpy as np

from libics.core.data.arrays import get_coordinate_meshgrid
from libics.core.util import misc

from state_reconstruction.gen.image_gen import get_local_psfs


###############################################################################
# Generate embedded PSFs
###############################################################################


def get_sites_pos_in_roi(
    trafo_site_to_image, rect=None, center=None, size=None
):
    """
    Gets all sites within the ROI of an image rectangle.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d`
        Affine transformation between sites and image coordinates.
    rect : `Iter[(int, int)]`
        Image rectangle specifying the ROI. Dimensions: `[ndim, (min, max)]`.
        Takes precedence over `center/size`.
    center, size : `(int, int)`
        Alternative parametrization of `rect`.

    Returns
    -------
    image_pos : `np.ndarray(2, float)`
        (Fractional) atom positions in image coordinates.
        Dimensions: `[n_atoms, ndim]`.
    """
    # Parse parameters
    if rect is None:
        rect = [
            [_c - _s / 2, _c + _s / 2]
            for _c, _s in zip(center, size)
        ]
    rect = np.array(rect)
    # Transform image corners into site space
    image_corners = misc.get_combinations(rect)
    sites_corners = np.round(
        trafo_site_to_image.coord_to_origin(image_corners)
    ).astype(int)
    # Find all sites within big rectangle and transform back
    sites_pos = get_coordinate_meshgrid(*[
        np.arange(np.min(_sc), np.max(_sc) + 1)
        for _sc in sites_corners.T
    ])
    sites_pos = np.reshape(sites_pos, (-1, 2))
    image_pos = trafo_site_to_image.coord_to_target(sites_pos)
    # Select sites within image ROI
    mask = np.logical_and.reduce([
        (image_pos[:, i] >= rect[i, 0]) & (image_pos[:, i] <= rect[i, 1])
        for i in range(image_pos.shape[1])
    ], axis=0)
    image_pos = image_pos[mask]
    return image_pos


def get_embedded_local_psfs(
    local_psfs, offset=None, size=None, normalize=True
):
    """
    Gets the local PSFs embedded in a full image.

    Parameters
    ----------
    local_psfs : `dict(str->Iter[Any])`
        Data object obtained from :py:func:`gen.image_gen.get_local_psfs`.
    offset : `(int, int)`
        Global offset of coordinates given in `local_psfs`.
    size : `(int, int)`
        Full image shape.
    normalize : `bool`
        Whether to normalize the individual embedded PSFs.
        (Might be necessary if PSFs are located at edges of image.)

    Returns
    -------
    images : `np.ndarray(3, float)`
        Local PSFs embedded in full image. Dimensions: `[n_psfs, x, y]`.
    """
    # Parse parameters
    lpsfs = local_psfs
    integrated_psfs = lpsfs["psf"]
    X_min, X_max = lpsfs["X_min"], lpsfs["X_max"]
    Y_min, Y_max = lpsfs["Y_min"], lpsfs["Y_max"]
    label_num = len(integrated_psfs)
    if offset is None:
        offset = np.min(X_min), np.min(Y_min)
    offset = np.array(offset)
    if size is None:
        size = np.array(np.max(X_max), np.max(Y_max)) - offset
    size = np.array(size)
    # Generate images
    images = []
    for i in range(label_num):
        # Resize to account for borders
        xmin, xmax = X_min[i] - offset[0], X_max[i] - offset[0]
        ymin, ymax = Y_min[i] - offset[1], Y_max[i] - offset[1]
        rois = []
        if xmin < 0:
            rois.append(slice(-xmin, None))
            xmin = 0
        elif xmax > size[0]:
            rois.append(slice(None, size[0] - xmax))
            xmax = size[0]
        else:
            rois.append(slice(None))
        if ymin < 0:
            rois.append(slice(-ymin, None))
            ymin = 0
        elif ymax > size[1]:
            rois.append(slice(None, size[1] - ymax))
            ymax = size[1]
        else:
            rois.append(slice(None))
        rois = tuple(rois)
        ipsf = integrated_psfs[i][rois]
        # Create image
        if normalize:
            ipsf /= np.sum(ipsf)
        im = np.zeros(size, dtype=float)
        im[xmin:xmax, ymin:ymax] = ipsf
        images.append(im)
    return np.array(images)


###############################################################################
# Generate projectors
###############################################################################


def get_embedded_projectors(embedded_psfs):
    """
    Calculates the orthogonal projectors given all embedded PSFs.

    Parameters
    ----------
    embedded_psfs : `np.ndarray(3, float)`
        Local PSFs embedded in full image. Dimensions: `[n_psfs, x, y]`.

    Returns
    -------
    projs : `np.ndarray(3, float)`
        Projectors embedded in full image. Dimensions: `[n_psfs, x, y]`.
    """
    epsfs = np.array(embedded_psfs)
    psf_count = len(epsfs)
    im_shape = epsfs.shape[1:]
    epsfs_vec = epsfs.reshape((psf_count, -1))
    projs_vec = np.linalg.pinv(epsfs_vec)
    projs = projs_vec.reshape(im_shape + (psf_count,))
    projs = np.moveaxis(projs, -1, 0)
    return projs


def get_projector(
    trafo_site_to_image, integrated_psf_generator,
    dx=0, dy=0, rel_embedding_size=4, normalize=True,
    proj_shape=None, proj_fidelity=None, ret_proj_fidelity=False
):
    """
    Gets the projector associated with a subpixel-shifted binned PSF.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d`
        Affine transformation between sites and image coordinates.
    integrated_psf_generator : `IntegratedPsfGenerator`
        Integrated PSF generator object.
    dx, dy : `int`
        PSF center shift in (fully-resolved) subpixels.
    rel_embedding_size : `int`
        Size of the image in which the projector is calculated,
        where the `rel_embedding_size` specifies this size with respect to
        the binned PSF size.
    normalize : `bool`
        Whether to normalize the projector.
    proj_shape : `(int, int)`
        Returned projector shape. Overwrites `proj_fidelity`.
        If `None`, deduces shape from `proj_fidelity`.
    proj_fidelity : `float`
        Adjusts the projector shape such that the projector fidelity
        is above `proj_fidelity`.
        If both `proj_shape` and `proj_fidelity` are `None`,
        the projector shape corresponds to the integrated PSF shape.
    ret_proj_fidelity : `bool`
        Whether to return the projector fidelity.
        The projector fidelity is defined as the absolute value contained
        in the cropped ROI in relative to the full projector.

    Returns
    -------
    center_proj : `np.ndarray(2, float)`
        Projector. Dimensions: `[n_binned_psf, n_binned_psf]`
    proj_fidelity : `float`
        If `ret_proj_fidelity` is set, returns the projector fidelity.
    """
    center_shift = (
        np.array([dx, dy]) / integrated_psf_generator.psf_supersample
    )
    if proj_fidelity is not None:
        if proj_fidelity <= 0 or proj_fidelity > 1:
            raise ValueError("Invalid `proj_fidelity`")
    # Center trafo
    centered_trafo = copy.deepcopy(trafo_site_to_image)
    centered_trafo.offset = np.zeros(2)
    # Build embedding area
    embedding_size = rel_embedding_size * integrated_psf_generator.psf_shape
    image_pos = get_sites_pos_in_roi(
        centered_trafo, center=(0, 0), size=embedding_size
    ) + center_shift
    center_idx = np.argmin(np.linalg.norm((image_pos - center_shift), axis=1))
    # Get local PSFs
    local_psfs = get_local_psfs(
        *image_pos.T, integrated_psf_generator=integrated_psf_generator
    )
    embedded_psfs = get_embedded_local_psfs(
        local_psfs,
        offset=-embedding_size//2,
        size=embedding_size,
        normalize=True
    )
    # Get projectors
    embedded_projs = get_embedded_projectors(embedded_psfs)
    # Crop central projector
    if proj_shape is None and proj_fidelity is None:
        proj_shape = integrated_psf_generator.psf_shape
    if proj_shape is not None:
        roi = misc.cv_index_center_to_slice(
            np.array(embedded_psfs.shape[1:]) // 2, proj_shape
        )
    else:
        _fidelity = 0
        _shape = np.array(integrated_psf_generator.psf_shape)
        _shape = _shape - np.min(_shape) + 1
        _denominator = np.sum(np.abs(embedded_projs[center_idx]))
        while _fidelity < proj_fidelity:
            roi = misc.cv_index_center_to_slice(
                np.array(embedded_psfs.shape[1:]) // 2, _shape
            )
            _fidelity = (
                np.sum(np.abs(embedded_projs[center_idx][roi])) / _denominator
            )
            _shape += 1
    center_proj = embedded_projs[center_idx][roi]
    proj_fidelity = (
        np.sum(np.abs(embedded_projs[center_idx][roi]))
        / np.sum(np.abs(embedded_projs[center_idx]))
    )
    if normalize:
        center_proj /= np.sum(center_proj)
    if ret_proj_fidelity:
        return center_proj, proj_fidelity
    else:
        return center_proj


###############################################################################
# Projector generator class
###############################################################################


class ProjectorGenerator:

    """
    Class for generating projectors from PSFs.

    Uses pre-calculation to allow for high-performance generation
    of subpixel-shifted projectors.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d`
        Affine transformation between site and image coordinates.
    integrated_psf_generator : `IntegratedPsfGenerator`
        Binned PSF generator object.
    rel_embedding_size : `int`
        Relative embedding size used for calculating the projectors.
        See :py:func:`get_projector` for details.
    proj_shape : `(int, int)` or `None`
        Shape of projector array. If `None`, uses PSF shape.

    Attributes
    ----------
    proj_cache_built : `bool`
        Whether the internal cache has been set up.

    Examples
    --------
    Standard use case given a :py:class:`IntegratedPsfGenerator` and
    an `AffineTrafo2d` object:

    >>> type(ipsfgen)
    state_reconstruction.gen.psf_gen.IntegratedPsfGenerator
    >>> ipsfgen.psf_shape
    (21, 21)
    >>> type(trafo)
    libics.tools.trafo.linear.AffineTrafo2d
    >>> prjgen = ProjectorGenerator(
    ...     trafo_site_to_image=trafo, integrated_psf_generator=ipsfgen
    ... )
    >>> prjgen.setup_cache()
    >>> prjgen.proj_cache_built
    True
    >>> prjgen.generate_projector(dx=0, dy=2).shape
    (21, 21)
    """

    def __init__(
        self, trafo_site_to_image=None,
        integrated_psf_generator=None, rel_embedding_size=4,
        proj_shape=None
    ):
        # Protected variables
        self._proj_cache = None
        # Public variables
        self.trafo_site_to_image = trafo_site_to_image
        self.integrated_psf_generator = integrated_psf_generator
        self.rel_embedding_size = rel_embedding_size
        self.proj_cache_built = False
        self.proj_shape = proj_shape

    @property
    def psf_supersample(self):
        return self.integrated_psf_generator.psf_supersample

    @property
    def psf_shape(self):
        return self.integrated_psf_generator.psf_shape

    def setup_cache(self, min_proj_fidelity=0.8, print_progress=False):
        """
        Sets up the cache for generating subpixel-shifted projectors.

        Parameters
        ----------
        min_proj_fidelity : `float`
            If the projector fidelity is smaller than `min_proj_fidelity`,
            an error is printed.
        print_progress : `bool``
            Whether a progress bar is printed.
        """
        # Verify integrated psf generator cache
        if not self.integrated_psf_generator.psf_integrated_cache_built:
            self.integrated_psf_generator.setup_cache(
                print_progress=print_progress
            )
        if self.proj_shape is None:
            self.LOGGER.warn("Using `psf_shape` as `proj_shape`")
            self.proj_shape = self.psf_shape
        # Build projector cache
        _ss = self.psf_supersample
        _hss = _ss // 2
        self._proj_cache = np.full((
            _ss, _ss, self.proj_shape[0], self.proj_shape[1]
        ), np.nan, dtype=float)
        _iter = misc.get_combinations([
            np.arange(-_hss, _ss - _hss),
            np.arange(-_hss, _ss - _hss)
        ])
        if print_progress:
            _iter = misc.iter_progress(_iter)
        for dx, dy in _iter:
            self._proj_cache[dx, dy], _fidelity = get_projector(
                self.trafo_site_to_image, self.integrated_psf_generator,
                dx=dx, dy=dy, rel_embedding_size=self.rel_embedding_size,
                proj_shape=self.proj_shape, ret_proj_fidelity=True
            )
            if _fidelity < min_proj_fidelity:
                self.LOGGER.error(
                    f"Low projector fidelity for index [{dx:d}, {dy:d}]: "
                    f"{_fidelity:.3f} < {min_proj_fidelity:.3f}"
                )
        self.proj_cache_built = True

    def generate_projector(self, dx=0, dy=0):
        """
        Gets the projector with subpixel shift.

        Parameters
        ----------
        dx, dy : `int`
            Projector center shift in units of fully resolved pixels.

        Returns
        -------
        projector : `np.ndarray(2, float)`
            Projector.
        """
        if self.proj_cache_built:
            return self._proj_cache[dx, dy].copy()
        else:
            return get_projector(
                self.trafo_site_to_image, self.integrated_psf_generator,
                dx=dx, dy=dy, rel_embedding_size=self.rel_embedding_size,
                proj_shape=self.proj_shape
            )
