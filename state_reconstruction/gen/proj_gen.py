"""
Projector generator.

Calculates the projectors to map images to lattice sites.
"""

import copy
import numpy as np
import os

from libics.env.logging import get_logger
from libics.core.data.arrays import get_coordinate_meshgrid
from libics.core.util import misc
from libics.core.data.types import AttrHashBase
from libics.core import io

from state_reconstruction import __version__
from state_reconstruction import config
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

    Returns
    -------
    center_proj : `np.ndarray(2, float)`
        (Embedded) projector. Dimensions:
        `[rel_embedding_size*n_binned_psf, rel_embedding_size*n_binned_psf]`.
    """
    center_shift = (
        np.array([dx, dy]) / integrated_psf_generator.psf_supersample
    )
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
        local_psfs, offset=-embedding_size//2,
        size=embedding_size, normalize=True
    )
    # Get projectors
    embedded_projs = get_embedded_projectors(embedded_psfs)
    center_proj = embedded_projs[center_idx]
    if normalize:
        center_proj /= np.sum(center_proj)
    return center_proj


def get_projector_positivity(proj):
    """
    Gets the ratio between signed sum and sum of absolute values of projector.
    """
    return np.sum(proj) / np.sum(np.abs(proj))


def get_projector_fidelity(proj_cropped, proj_full):
    """
    Gets the relative (absolute) weight contained in the cropped projector.
    """
    if not np.isscalar(proj_full):
        proj_full = np.sum(np.abs(proj_full))
    return np.sum(np.abs(proj_cropped)) / proj_full


def crop_projector(
    proj_full, proj_shape=None, proj_fidelity=1, normalize=False
):
    """
    Parameters
    ----------
    proj_full : `Array[2, float]`
        Full (embedded) projector.
    proj_shape : `(int, int)`
        Returned projector shape. Overwrites `proj_fidelity`.
        If `None`, deduces shape from `proj_fidelity`.
    proj_fidelity : `float`
        Adjusts the projector shape such that the projector fidelity
        is above `proj_fidelity`.
    normalize : `bool`
        Whether to normalize the projector.

    Returns
    -------
    proj_cropped : `Array[2, float]`
        Cropped projector.
    """
    # Parse parameters
    if proj_shape is None:
        if proj_fidelity > 1:
            raise ValueError("Invalid `proj_fidelity`")
        elif proj_fidelity == 1:
            return proj_full.copy()
    proj_center = np.array(proj_full.shape) // 2
    # If proj_shape is fixed
    if proj_shape is not None:
        roi = misc.cv_index_center_to_slice(proj_center, proj_shape)
    # Iterate proj_shape until proj_fidelity is reached
    else:
        _fidelity = 0
        _shape = np.array(proj_full.shape)
        _shape = _shape - np.min(_shape)
        _denominator = np.sum(np.abs(proj_full))
        while _fidelity < proj_fidelity:
            _shape += 1
            roi = misc.cv_index_center_to_slice(proj_center, _shape)
            _fidelity = get_projector_fidelity(proj_full[roi], _denominator)
    # Crop projector
    proj_cropped = proj_full[roi]
    if normalize:
        proj_cropped /= np.sum(proj_cropped)
    return proj_cropped


###############################################################################
# Projector generator class
###############################################################################


class ProjectorGenerator(AttrHashBase):

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
    ...     trafo_site_to_image=trafo,
    ...     integrated_psf_generator=ipsfgen,
    ...     proj_shape=(31, 31)
    ... )
    >>> prjgen.setup_cache()
    >>> prjgen.proj_cache_built
    True
    >>> prjgen.generate_projector(dx=0, dy=2).shape
    (31, 31)
    """

    LOGGER = get_logger("srec.ProjectorGenerator")
    HASH_KEYS = AttrHashBase.HASH_KEYS | {
        "trafo_site_to_image", "integrated_psf_generator", "rel_embedding_size"
    }

    def __init__(
        self, trafo_site_to_image=None,
        integrated_psf_generator=None, rel_embedding_size=4,
        proj_shape=None
    ):
        super().__init__()
        # Protected variables
        self._proj_full_cache = None
        self._proj_shape = proj_shape
        # Public input variables
        self.trafo_site_to_image = trafo_site_to_image
        self.integrated_psf_generator = integrated_psf_generator
        self.rel_embedding_size = rel_embedding_size
        # Public output variables
        self.proj_cache = None
        self.proj_cache_built = False
        self.proj_fidelity = None
        self.proj_positivity = None

    def get_attr_str(self):
        keys = ["rel_embedding_size", "proj_cache_built", "proj_positivity"]
        s = [f" → {k}: {str(getattr(self, k))}" for k in keys]
        return "\n".join(s)

    def __str__(self):
        return f"{self.__class__.__name__}:\n{self.get_attr_str()}"

    def __repr__(self):
        s = f"<'{self.__class__.__name__}' at {hex(id(self))}>"
        return f"{s}\n{self.get_attr_str()}"

    @property
    def psf_supersample(self):
        return self.integrated_psf_generator.psf_supersample

    @property
    def psf_shape(self):
        return self.integrated_psf_generator.psf_shape

    @property
    def proj_shape(self):
        _ps = self._proj_shape
        return _ps if _ps is not None else self.psf_shape

    @proj_shape.setter
    def proj_shape(self, val):
        if np.any(self._proj_shape != val):
            self._proj_shape = val
            if self.proj_cache_built is True:
                self.crop_cache()

    def setup_cache(self, from_file=True, to_file=True, print_progress=False):
        """
        Sets up the cache for generating subpixel-shifted projectors.

        Parameters
        ----------
        from_file : `bool`
            Whether to allow loading from a cache file.
        to_file : `bool`
            Whether to allow saving to a cache file.
        print_progress : `bool``
            Whether a progress bar is printed.
        """
        # Verify integrated psf generator cache
        if not self.integrated_psf_generator.psf_integrated_cache_built:
            self.integrated_psf_generator.setup_cache(
                print_progress=print_progress
            )
        # Try to load from file
        if from_file:
            if self.load_cache():
                if print_progress:
                    print("Loaded from cache file")
                return
        # Initialize variables
        _ss = self.psf_supersample
        _hss = _ss // 2
        cache_shape = (_ss, _ss)
        proj_full_shape = tuple(
            np.array(self.psf_shape) * self.rel_embedding_size
        )
        self._proj_full_cache = np.zeros(
            cache_shape + proj_full_shape, dtype=float
        )
        # Calculate projectors
        _iter = misc.get_combinations([
            np.arange(-_hss, _ss - _hss), np.arange(-_hss, _ss - _hss)
        ])
        if print_progress:
            _iter = misc.iter_progress(_iter)
        for dx, dy in _iter:
            _proj = get_projector(
                self.trafo_site_to_image, self.integrated_psf_generator,
                dx=dx, dy=dy, rel_embedding_size=self.rel_embedding_size
            )
            self._proj_full_cache[dx, dy] = _proj
        # Save to file
        if to_file:
            self.save_cache()
        # Assign attributes
        self.proj_cache_built = True
        self._calc_proj_positivity()
        self.crop_cache()

    def _calc_proj_positivity(self):
        """
        Calculates the full projector positivity.
        """
        cache_shape = self._proj_full_cache.shape[:2]
        _positivity = np.zeros(cache_shape, dtype=float)
        for i, j in np.ndindex(*cache_shape):
            _proj = self._proj_full_cache[i, j]
            _positivity[i, j] = get_projector_positivity(_proj)
        self.proj_positivity = np.mean(_positivity)

    def crop_cache(self, proj_shape=False):
        """
        Parameters
        ----------
        proj_shape : `(int, int)` or `None` or `False`
            If `False`, uses its stored attribute value :py:attr:`proj_shape`.
            If `(int, int)`, overwrites the attribute.
            If `None`, uses the PSF shape :py:attr:`psf_shape`.
        """
        # Parse parameters
        if self.proj_cache_built is False:
            raise RuntimeError(
                "Projector cache is not set up, run `setup_cache` first"
            )
        if proj_shape is not False:
            self._proj_shape = proj_shape
        proj_shape = self.proj_shape
        cache_shape = self._proj_full_cache.shape[:2]
        # Crop projectors
        self.proj_cache = np.zeros(cache_shape + proj_shape, dtype=float)
        fidelity = np.zeros(cache_shape, dtype=float)
        for x, y in np.ndindex(*self._proj_full_cache.shape[:2]):
            proj_full = self._proj_full_cache[x, y]
            proj_cropped = crop_projector(proj_full, proj_shape)
            self.proj_cache[x, y] = proj_cropped
            fidelity[x, y] = get_projector_fidelity(proj_cropped, proj_full)
        # Assign attributes
        self.proj_fidelity = np.mean(fidelity)

    def get_cache_fp(self):
        """
        Gets the default cache file path.
        """
        fn = hex(hash(self)) + ".json"
        fp = os.path.join(config.get_config("projector_cache_dir"), fn)
        return fp

    def save_cache(self, fp=None):
        """
        Saves the full projector cache to file.
        """
        if fp is None:
            fp = self.get_cache_fp()
        _d = {
            "_proj_full_cache": self._proj_full_cache,
            "state_estimation_version": __version__
        }
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        return io.save(fp, _d)

    def load_cache(self, fp=None):
        """
        Loads the full projector cache from file.
        """
        if fp is None:
            fp = self.get_cache_fp()
        if not os.path.exists(fp):
            return False
        _d = io.load(fp)
        if _d["state_estimation_version"] != __version__:
            self.LOGGER.warn(
                f"Cached file has wrong version (fp): "
                f"loaded: {_d['state_estimation_version']}, "
                f"installed: {__version__}"
            )
        self._proj_full_cache = _d["_proj_full_cache"]
        self.proj_cache_built = True
        self._calc_proj_positivity()
        self.crop_cache()
        return True

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
            return self.proj_cache[dx, dy].copy()
        else:
            proj_full = get_projector(
                self.trafo_site_to_image, self.integrated_psf_generator,
                dx=dx, dy=dy, rel_embedding_size=self.rel_embedding_size,
                proj_shape=self.proj_shape
            )
            return crop_projector(proj_full, proj_shape=self.proj_shape)
