"""
Image generator.

Simulates fluorescence images of atomic arrays.
"""

import numpy as np

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData, get_coordinate_meshgrid

from .psf_gen import IntegratedPsfGenerator
from .trafo_gen import get_trafo_site_to_image

LOGGER = get_logger("srec.gen.image_gen")


###############################################################################
# Generate sites
###############################################################################


def get_sites_mi(
    center=(85, 85), size=(15, 15), filling=None, shape="round",
    outside_count=None, outside_size=None, seed=None
):
    """
    Gets the site coordinates of an atomic array.

    Parameters
    ----------
    center : `(int, int)`
        Center site.
    size : `(int, int)`
        Full size of the array in units of sites.
    filling : `float` or `None`
        Filling fraction. If `None`, assumes unity filling.
    shape : `str`
        Spatial shape of array. Options: `"round", "square"`.
    outside_count : `int` or `None`
        Number of atoms outside the array `size`.
        If `None`, uses zero.
    outside_size : `int` or `(int, int)`
        Size within which outside atoms are located.
    seed : `int`
        Random number generator seed.
        The RNG is used for finite filling and outside atom location.

    Returns
    -------
    X, Y : `np.ndarray(1, float)`
        Coordinates of occupied sites.
    """
    # Parse parameters
    if np.isscalar(center):
        center = (center, center)
    center = np.array(center)
    if np.isscalar(size):
        size = (size, size)
    size = np.array(size)
    # Generate rectangular sites
    ad = ArrayData()
    ad.add_dim(offset=center[0]-size[0]//2, step=1)
    ad.add_dim(offset=center[1]-size[1]//2, step=1)
    ad.var_shape = size
    X, Y = ad.get_var_meshgrid()
    X, Y = X.ravel(), Y.ravel()
    # Remove sites outside ellipse
    _shape_is_round = (
        shape.lower() == "round" or shape.lower()[:8] == "elliptic"
    )
    if _shape_is_round:
        R = np.sqrt(
            ((X - center[0]) / size[0])**2
            + ((Y - center[1]) / size[1])**2
        )
        mask = R <= 0.5
        X, Y = X[mask], Y[mask]
    # Remove atoms for finite filling
    if filling is not None:
        np.random.seed(seed)
        mask = np.random.random(len(X)) <= filling
        X, Y = X[mask], Y[mask]
    # Add isolated atoms outside MI
    if outside_count is not None:
        if outside_size is None:
            outside_size = 2 * center
        if np.isscalar(outside_size):
            outside_size = (outside_size, outside_size)
        outside_size = np.array(outside_size)
        np.random.seed(seed)
        X_iso, Y_iso = [], []
        if np.any(outside_size <= size):
            LOGGER.warn(
                "get_sites_mi: `outside_size` should be larger than `size`"
            )
        else:
            _outside_half_size = np.round(outside_size / 2).astype(int)
            _coords = np.array(get_coordinate_meshgrid(*[
                np.arange(-_ohs, _ohs + 1)
                for _ohs in _outside_half_size
            ]))
            _coords = np.reshape(_coords, (-1, 2))
            if _shape_is_round:
                _accept = (
                    (np.linalg.norm(_coords / (size / 2), axis=-1) > 1)
                    & (np.linalg.norm(_coords / _outside_half_size, axis=-1)
                       <= 1)
                )
            else:
                _accept = np.all(np.abs(_coords) > (size / 2), axis=-1)
            _coords = _coords[_accept]
            _choice_idx = np.random.choice(
                np.arange(len(_coords)), size=outside_count, replace=False
            )
            X_iso, Y_iso = np.moveaxis(_coords[_choice_idx] + center, -1, 0)
        X, Y = np.concatenate([X, X_iso]), np.concatenate([Y, Y_iso])
    return X, Y


def apply_coords(
    X, Y, vals=0, fill=0,
    rect=None, center=None, size=(170, 170), offset=(0, 0)
):
    """
    Uses given coordinates to fill a 2D array.

    Parameters
    ----------
    X, Y : `Array[1, int]`
        Coordinates to be filled.
    vals : `float` or `Array[1, float]`
        Fill value corresponding to `X, Y`.
        If `float`, fills the given value on all coordinates.
    fill : `float`
        Default unfilled value.
    rect : `Iter[(int, int)]`
        Size of returned 2D array. Dimensions: `[ndim, (min, max)]`.
        Takes precedence for specifying 2D array.
    center, size, offset : `(int, int)`
        A combination of these values are used to construct `rect`
        if `rect` is not given.

    Returns
    -------
    ad : `ArrayData(2, float)`
        Filled 2D array.
    """
    if rect is not None:
        rect = np.array(rect)
        offset = rect[..., 0]
        shape = rect[..., 1] - rect[..., 0]
    else:
        if center is None:
            offset = np.array(offset)
        else:
            offset = np.array(center) - np.array(size) // 2
        shape = np.array(size)
    ad = ArrayData(np.full(shape, fill, dtype=float))
    for i, _o in enumerate(offset):
        ad.set_dim(i, offset=_o, step=1)
    if np.isscalar(vals):
        vals = np.full(len(X), vals)
    XX, YY = np.round([X, Y]).astype(int) - offset[..., np.newaxis]
    for i, (x, y) in enumerate(zip(XX, YY)):
        ad.data[x, y] = vals[i]
    return ad


###############################################################################
# Generate image
###############################################################################


def get_brightness_sample(
    X, Y, std=0.2, additive_distributions=None, seed=None
):
    """
    Gets Gaussian-distributed random brightness values with unity mean.

    Parameters
    ----------
    X, Y : `Array[1, float]`
        Site coordinates.
    std : `float`
        Standard deviation of brightness.
    additive_distributions : `Iter[[float or callable, float, float]]`
        Each item leads to additive brightness. The items are interpreted as:
        `[probability to add this distribution, mean, std]`.
        If a `float` is given for the probability, it is uniformly applied
        to all atoms. If a `callable` is given, it should have the signature:
        `func(X, Y)->probability`.
    seed : `int`
        Random number generator seed.

    Returns
    -------
    brightness : `np.ndarray(1, float)`
        Random brightness values.
    """
    np.random.seed(seed)
    brightness = np.random.normal(loc=1, scale=std, size=len(X))
    if additive_distributions:
        for _prob, _mean, _std in additive_distributions:
            if callable(_prob):
                _prob = _prob(X, Y)
            mask = np.random.random(size=len(X)) <= _prob
            additive = np.random.normal(
                loc=_mean, scale=_std, size=np.count_nonzero(mask)
            )
            brightness[mask] += additive
    return brightness


def get_local_psfs(
    X, Y, integrated_psf_generator=None, psf=None, integration_size=5
):
    """
    Given image coordinates, gets the local binned PSF.

    Parameters
    ----------
    X, Y : `Array[1, float]`
        Fractional image coordinates of PSF centers.
    integrated_psf_generator : `IntegratedPsfGenerator`
        Binned PSF generator object.

    Returns
    -------
    res : `dict(str->Iter[Any])`
        Dictionary containing the following items:
    psf : `list(np.ndarray(2, float))`
        Local binned PSFs.
    X_int, Y_int : `np.ndarray(1, int)`
        Rounded PSF center coordinates.
    X_min, X_max, Y_min, Y_max : `np.ndarray(1, int)`
        Rounded PSF rectangle corners.
    """
    if integrated_psf_generator is None:
        integrated_psf_generator = IntegratedPsfGenerator(
            psf=psf, psf_supersample=integration_size
        )
    psfgen = integrated_psf_generator
    X_int, Y_int = np.round(X).astype(int), np.round(Y).astype(int)
    X_res, Y_res = X - X_int, Y - Y_int
    dx = np.round(X_res * psfgen.psf_supersample).astype(int)
    dy = np.round(Y_res * psfgen.psf_supersample).astype(int)
    psf_shape = np.array(psfgen.psf.shape) // psfgen.psf_supersample
    X_min, Y_min = X_int - psf_shape[0] // 2, Y_int - psf_shape[1] // 2
    X_max, Y_max = X_min + psf_shape[0], Y_min + psf_shape[1]
    integrated_psfs = [
        psfgen.generate_integrated_psf(dx=dx[i], dy=dy[i])
        for i in range(len(X))
    ]
    return {
        "psf": integrated_psfs, "X_int": X_int, "Y_int": Y_int,
        "X_min": X_min, "X_max": X_max, "Y_min": Y_min, "Y_max": Y_max
    }


def get_image_clean(
    X=None, Y=None, psf=None, local_psfs=None,
    brightness=1, integration_size=5, size=(512, 512)
):
    """
    Gets a noise-free image for given atom coordinates and binned PSFs.

    Parameters
    ----------
    local_psfs : `dict(str->Iter[Any])`
        Data object obtained from :py:func:`get_local_psfs`.
        (Alternatively: provide `(X, Y, psf, integration_size)`.)
    brightness : `float` or `Array[1, float]`
        Brightness of each atom.
        If `float`, uses same brightness for all atoms.
    size : `(int, int)`
        Image shape.

    Returns
    -------
    img : `np.ndarray(2, float)`
        Atomic fluorescence image.
    """
    if np.isscalar(size):
        size = (size, size)
    if np.isscalar(brightness):
        brightness = np.full_like(X, brightness)
    if local_psfs is None:
        if X is None or Y is None or psf is None:
            raise ValueError("invalid parameters")
        lpsfs = get_local_psfs(
            X, Y, psf=psf, integration_size=integration_size
        )
    else:
        lpsfs = local_psfs
    integrated_psfs = lpsfs["psf"]
    X_min, X_max = lpsfs["X_min"], lpsfs["X_max"]
    Y_min, Y_max = lpsfs["Y_min"], lpsfs["Y_max"]
    img = np.zeros(size, dtype=float)
    for i, integ_psf in enumerate(integrated_psfs):
        img[X_min[i]:X_max[i], Y_min[i]:Y_max[i]] += integ_psf * brightness[i]
    return img


def get_image_sample(
    image_clean, counts_per_atom, normalize_counts=False,
    white_noise_counts=None, seed=None
):
    """
    Gets an image with Poissonian fluorescence noise and background noise.

    Parameters
    ----------
    image_clean : `Array[2, float]`
        Noise-free image.
        Array sum should equal number of atoms multiplied by brightness.
    counts_per_atom : `int`
        Number of sampled counts per atom.
    normalize_counts : `bool`
        Whether the resulting image should be normalized.
    white_noise_counts : `int` or `None`
        Standard deviation of Gaussian white background noise.
        If `None`, does not apply background noise.
    seed : `int`
        Random number generator seed.

    Returns
    -------
    image_sample : `np.ndarray(2, float)`
        Noisy image.
    """
    np.random.seed(seed)
    image_sample = np.random.poisson(
        lam=image_clean*counts_per_atom, size=image_clean.shape
    ).astype(float)
    if white_noise_counts is not None:
        image_sample += np.random.normal(
            loc=0, scale=white_noise_counts, size=image_clean.shape
        )
    if normalize_counts is not False:
        image_sample /= counts_per_atom
    return image_sample


###############################################################################
# Image generator class
###############################################################################


class ImageGenerator:

    """
    Class for simulating fluorescence images of atomic arrays.

    Parameters
    ----------
    image_size : `(int, int)`
        Image shape.
    image_counts_per_atom : `int`
        Number of counts per atom.
    image_counts_white_noise : `int` or `None`
        Standard deviation of Gaussian white background noise.
        If `None`, does not apply background noise.
    sites_size : `(int, int)`
        Shape of lattice sites.
    atoms_center : `(int, int)`
        Center site of atomic array.
    atoms_size : `(int, int)`
        Full size of atomic array.
    atoms_filling : `float` or `None`
        Array filling fraction. If `None`, assumes unity filling.
    atoms_shape : `str`
        Spatial shape of array. Options: `"round", "square"`.
    atoms_outside_count : `int` or `None`
        Number of atoms outside the array `size`.
        If `None`, uses zero.
    atoms_outside_size : `int` or `(int, int)`
        Size within which outside atoms are located.
    atoms_fluo_std : `float`
        Standard deviation of brightness of atoms.
    atoms_fluo_add_distr : `Iter[[float or callable, float, float]]`
        `[probability to add this distribution, mean, std]`.
        See :py:func:`get_brightness_sample` for details.
    trafo_site_to_image : `AffineTrafo2d`
        Affine transformation object between
        lattice sites and image coordinates.
    psf : `Array[2, float]`
        Fully resolved PSF.
    psf_supersample : `int`
        Binning of PSF.

    Examples
    --------
    Standard use case given a fully resolved PSF and a transformation object:

    >>> psf.shape
    (105, 105)
    >>> type(trafo)
    libics.tools.trafo.linear.AffineTrafo2d
    >>> imggen = ImageGenerator(
    ...     image_size=(512, 512),
    ...     trafo_site_to_image=trafo, psf=psf
    ... )
    >>> imggen.generate_image().shape
    (512, 512)
    """

    def __init__(
        self, image_size=(512, 512),
        image_counts_per_atom=850*14, image_counts_white_noise=70,
        sites_size=(170, 170),
        atoms_center=(85, 85), atoms_size=(30, 30),
        atoms_filling=0.9, atoms_shape="round",
        atoms_outside_count=None, atoms_outside_size=None,
        atoms_fluo_std=0.15, atoms_fluo_add_distr=None,
        trafo_site_to_image=None,
        psf=None, psf_supersample=None
    ):
        # Image configuration
        self.image_size = image_size
        self.image_counts_per_atom = image_counts_per_atom
        self.image_counts_white_noise = image_counts_white_noise
        # Sites configuration
        self.sites_size = sites_size
        # Atoms configuration
        self.atoms_center = atoms_center
        self.atoms_size = atoms_size
        self.atoms_filling = atoms_filling
        self.atoms_shape = atoms_shape
        self.atoms_outside_count = atoms_outside_count
        self.atoms_outside_size = atoms_outside_size
        self.atoms_fluo_std = atoms_fluo_std
        self.atoms_fluo_add_distr = atoms_fluo_add_distr
        self.trafo_site_to_image = trafo_site_to_image
        # Point spread function configuration
        self.psf = psf
        self.psf_supersample = psf_supersample

    def generate_image(
        self, seed=None, sites_phase=None,
        phase_ref_image=(0, 0), phase_ref_site=(169, 84),
        ret_vals=None
    ):
        """
        Gets a simulated fluorescence image.

        Parameters
        ----------
        seed : `int`
            Random number generator seed.
        sites_phase : `(float, float)`
            Lattice phase.
        phase_ref_image, phase_ref_site : `(int, int)`
            Transformation phase reference.
            See :py:func:`gen.trafo_gen.get_trafo_site_to_image` for details.
        ret_vals : `None` or `Iter[str]`
            Selects return values.
            If `None`, returns only the simulated image.
            If `Iter[str]`, returns a dictionary containing selected values.
            See `Returns` for details and options.

        Returns
        -------
        ret : `ArrayData(2, float)` or `dict(str->Any)`
            If `ret_vals` is `None`, returns the simulated image.
            Otherwise returns a dictionary containing the following items
            if selected in `ret_vals`:
        image_sample : `ArrayData(2, float)`
            Simulated image.
        coord_sites : `np.ndarray(2, int)`
            Site coordinates of occupied sites. Dimensions: `[ndim, ...]`.
        coord_image : `np.ndarray(2, float)`
            (Fractional) image coordinates of occupied sites.
            Dimensions: `[ndim, ...]`.
        occ_2d_sites : `ArrayData(2, float)`
            Site occupations as 2D array in site coordinates.
        occ_2d_image : `ArrayData(2, float)`
            Site occupations as 2D array in image coordinates.
        brightness : `np.ndarray(1, float)`
            Brightness of each atom.
        brightness_2d_sites : `ArrayData(2, float)`
            Brightness as 2D array in site coordinates.
        image_clean : `ArrayData(2, float)`
            Noise-free simulated image.
        trafo : `AffineTrafo2d`
            (Phase-shifted) affine transformation between sites and image
            coordinates.
        """
        # Generate filled sites
        _sites = np.array(get_sites_mi(
            center=self.atoms_center, size=self.atoms_size,
            filling=self.atoms_filling, shape=self.atoms_shape,
            outside_count=self.atoms_outside_count,
            outside_size=self.atoms_outside_size,
            seed=seed
        ))
        # Generate random lattice phase
        if sites_phase is None:
            sites_phase = np.random.random(size=2)
        _trafo = get_trafo_site_to_image(
            trafo_site_to_image=self.trafo_site_to_image,
            phase_ref_image=phase_ref_image, phase_ref_site=phase_ref_site,
            phase=sites_phase
        )
        # Transform into camera space
        _positions = np.transpose(_trafo.coord_to_target(np.transpose(_sites)))
        # Randomize fluorescence per atom
        _brightness = get_brightness_sample(
            *_positions, std=self.atoms_fluo_std,
            additive_distributions=self.atoms_fluo_add_distr,
            seed=seed
        )
        # Generate clean image
        _image_clean = get_image_clean(
            *_positions, self.psf, brightness=_brightness,
            integration_size=self.psf_supersample
        )
        # Generate randomized counts
        _image_sample = get_image_sample(
            _image_clean, self.image_counts_per_atom, normalize_counts=False,
            seed=seed, white_noise_counts=self.image_counts_white_noise
        )
        if ret_vals is None:
            return ArrayData(_image_sample)
        else:
            ret = {"image_sample": ArrayData(_image_sample)}
            if "coord_sites" in ret_vals:
                ret["coord_sites"] = _sites
            if "coord_image" in ret_vals:
                ret["coord_image"] = _positions
            if "occ_2d_sites" in ret_vals:
                ret["occ_2d_sites"] = apply_coords(
                    *_sites, vals=1, size=np.array(self.atoms_center)*2
                )
            if "occ_2d_image" in ret_vals:
                ret["occ_2d_image"] = apply_coords(
                    *_positions, vals=1, size=self.image_size
                )
            if "brightness" in ret_vals:
                ret["brightness"] = _brightness
            if "brightness_2d_sites" in ret_vals:
                ret["brightness_2d_sites"] = apply_coords(
                    *_sites, vals=_brightness,
                    size=np.array(self.atoms_center)*2
                )
            if "image_clean" in ret_vals:
                ret["image_clean"] = _image_clean
            if "trafo" in ret_vals:
                ret["trafo"] = _trafo
            return ret
