"""
Affine transformation estimator

Estimates the affine transformation parameters between lattice sites
and fluorescence image coordinates.
"""

import copy
import numpy as np
import scipy.optimize
import time

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData
from libics.core.util import misc
from libics.tools.trafo.linear import AffineTrafo2d
from libics.tools.math.optimize import maximize_discrete_stepwise
from libics.tools.math.signal import find_peaks_1d
from libics.tools.math.models import ModelBase
from libics.tools.math.peaked import FitGaussian1d, gaussian_1d, FitParabolic1d

from state_reconstruction.gen import trafo_gen
from .proj_est import get_local_images, apply_projectors

LOGGER = get_logger("srec.trafo_est")


###############################################################################
# Transformation parameter optimization
###############################################################################


def get_trafo_angle_fidelity(
    x, y, guess_trafo, ax=(0, 1),
    bins_per_site=5, peak_rel_prominence=0.1, peak_base_prominence_ratio=0.1,
    rel_spacing_tol=0.2
):
    """
    Calculates the projected transformation fidelity.

    Parameters
    ----------
    x, y : `Array[1, float]`
        Lattice sites in image coordinates.
    guess_trafo : `AffineTrafo2d`
        Affine transformation from sites to image coordinates to be checked.
    ax : `int` or `Iter[int]`
        Transformation axes along which to perform the fidelity analysis.
        The return values are vectorial if `ax` is vectorial and vice versa.
    bins_per_site : `int`
        Number of histogram bins per unit distance.
    peak_rel_prominence, peak_base_prominence_ratio : `float`
        Peak finding parameters.
        See :py:func:`libics.tools.math.signal.find_peaks_1d` for details.
    rel_spacing_tol : `float`
        Maximum allowed relative spacing deviation between the fitted
        spacing and the guessed spacing (extracted from `guess_trafo`).

    Returns
    -------
    ret : `dict(str->Any)` or `dict(str->Iter[Any])`
        Returns a dictionary containing the following items.
        The return is vectorial if `ax` is vectorial.
    projected_distance : `ArrayData(1, float)`
        Histogram of positional differences.
    spacing : `float`
        Fitted spacing relative to the spacing extracted from `guess_trafo`.
    width : `float`
        Fitted peak width relative to the spacing extracted from `guess_trafo`.
    fidelity : `float`
        Fidelity of transformation, defined as `1 - (width / spacing)²`.

    Notes
    -----
    Summary of the algorithm:

    * First, the image coordinates are transformed to site space.
    * Then the mutual differences are calculated (i.e. N² values).
      Due to translational invariance, the mutual differences should have
      regular peaks spaced by the lattice spacing.
    * Due to the many available values, a well-resolved histogram can be
      generated, revealing this peak structure.
    * Using a peak finder, the position and width of the peaks are determined.
    * If the peaks are sufficiently regular, a periodic Gaussian is fitted.
    * We use `1 - (width / spacing)²` as a fidelity measure.
    * Performing this analysis along the x (y) axis yields the
      fidelity of the transformation angle along the y (x) vectors.
    """
    # Parse parameters
    ax_is_scalar = np.isscalar(ax)
    if ax_is_scalar:
        ax = (ax,)
    ax_mask = np.array([i in ax for i in range(2)])
    # Transform to site space
    sites = np.transpose(guess_trafo.coord_to_origin(np.transpose([x, y])))
    # Calculate difference sites
    # (exploit translation invariance to obtain more points for histogram)
    proj_dists = []
    spacings = []
    widths = []
    fidelities = []
    for _sites in sites[ax_mask]:
        _dsites = _sites - _sites[:, np.newaxis]
        # Filter for positive distances (symmetric) and
        # avoid too few large-distance points
        _dsites = _dsites[_dsites > 0].ravel()
        _dsites = _dsites[_dsites <= np.percentile(_dsites, 90)]
        _max_dist = np.round(np.max(_dsites)).astype(int)
        # Construct histogram of projected distances
        _h, _e = np.histogram(
            _dsites, bins=bins_per_site*_max_dist, range=(0, _max_dist)
        )
        _c = (_e[1:] + _e[:-1]) / 2
        _proj_dist = ArrayData(_h)
        _proj_dist.set_dim(0, points=_c)
        proj_dists.append(_proj_dist)

        # Find peaks and widths of distances
        _peaks = find_peaks_1d(
            _c, _h, ret_vals=["width"],
            rel_prominence=peak_rel_prominence,
            base_prominence_ratio=peak_base_prominence_ratio,
            fit_algorithm="mean"
        )
        # Find spacing from peaks
        _peak_pos = np.array(_peaks["center"])
        _dpeak_pos = (_peak_pos - _peak_pos[:, np.newaxis]).ravel()
        _dpeak_pos = _dpeak_pos[(_dpeak_pos > 0.5) & (_dpeak_pos < 1.5)]
        _h, _e = np.histogram(
            _dpeak_pos, bins=len(_peak_pos), range=(0.5, 1.5)
        )
        _c = (_e[1:] + _e[:-1]) / 2
        try:
            _fit = FitGaussian1d(_c, _h)
            if not _fit.psuccess:
                raise TypeError
            spacings.append(_fit.x0)
        except TypeError:
            _pdf = _h / np.sum(_h)
            spacings.append(np.sum(_c * _pdf))
        # Find peak width
        widths.append(np.mean(_peaks["width"]))
        # Find fidelity (first peak should be around 1)
        _min_peaks = np.sort(_peak_pos)[:2]
        if (
            (len(_min_peaks) < 2)
            or (_min_peaks[0] < 1 - rel_spacing_tol
                or _min_peaks[0] > 1 + rel_spacing_tol)
            or (_min_peaks[1] < 2 - 2 * rel_spacing_tol
                or _min_peaks[1] > 2 + 2 * rel_spacing_tol)
        ):
            fidelities.append(0)
        else:
            fidelities.append(1 - (widths[-1] / spacings[-1])**2)
    # Package result
    spacings, widths = np.array(spacings), np.array(widths)
    fidelities = np.array(fidelities)
    # If high fidelity, perform precision spacing fit
    for i, fidelity in enumerate(fidelities):
        if fidelity > 0.5:
            _fit = FitPeriodicGaussian1d()
            _fit.find_p0(proj_dists[i])
            _fit.set_p0(x0=0, dx=spacings[i])
            _fit.find_popt(proj_dists[i])
            if _fit.psuccess:
                spacings[i] = _fit.dx
                widths[i] = _fit.wx
                fidelities[i] = 1 - (widths[i] / spacings[i])**2
            else:
                fidelities[i] = 0
    if ax_is_scalar:
        proj_dists, fidelities = proj_dists[0], fidelities[0]
        spacings, widths = spacings[0], widths[0]
    return {
        "projected_distance": proj_dists,
        "spacing": spacings, "width": widths,
        "fidelity": fidelities
    }


def get_trafo_from_sites_direct(
    x, y, guess_trafo, angle_range=np.deg2rad(3), angle_num=16,
    bins_per_site=5, peak_rel_prominence=0.1, peak_base_prominence_ratio=0.1,
    min_fidelity=0.8, print_progress=False
):
    """
    Gets the optimum transformation from given lattice sites.

    Parameters
    ----------
    x, y : `Array[1, float]`
        Lattice sites in image coordinates.
    guess_trafo : `AffineTrafo2d`
        Initial guess for the affine transformation between sites and
        image coordinates.
    angle_range : `float`
        Angle range in which to optimize in radians (rad).
    angle_num : `int`
        Number of steps in `angle_range`.
    bins_per_site, peak_rel_prominence, peak_base_prominence_ratio : `float`
        Transformation fidelity estimation parameters.
        See :py:func:`get_trafo_angle_fidelity` for details.
    min_fidelity : `float`
        Minimum transformation fidelity to be considered valid.
    print_progress : `bool`
        Whether to print a progress bar.

    Returns
    -------
    ret : `dict(str->Any)`
        Returns a dictionary containing the following items:
    trafo : `AffineTrafo2d`
        Optimal transformation.
    fidelity : `[ArrayData(1, float), ArrayData(1, float)]`
        Fidelity v. angle for the `[x, y]` axes.
    rel_spacing : `[ArrayData(1, float), ArrayData(1, float)]`
        Spacing relative to `guess_trafo` v. angle for the `[x, y]` axes.
    rel_width : `[ArrayData(1, float), ArrayData(1, float)]`
        Width relative to `guess_trafo` v. angle for the `[x, y]` axes.
    """
    # Parse parameters
    if np.isscalar(angle_range):
        angle_range = (angle_range, angle_range)
    guess_magnification, guess_angle, guess_offset = (
        guess_trafo.get_origin_axes()
    )
    test_angles = [
        np.linspace(
            angle - angle_range[i]/2, angle + angle_range[i]/2, num=angle_num
        ) for i, angle in enumerate(guess_angle)
    ]
    # Iterate axes
    _t0 = time.time()
    test_fidelities = []
    test_rel_spacings = []
    test_rel_widths = []
    for idx_trafo, idx_proj in enumerate(reversed(range(2))):
        # Iterate trafo test angles
        fidelities = []
        rel_spacings = []
        rel_widths = []
        for idx_angle, test_angle in enumerate(test_angles[idx_trafo]):
            if print_progress:
                misc.print_progress(
                    idx_trafo, 2, idx_angle, angle_num, start_time=_t0
                )
            # Set test trafo
            _angle = guess_angle.copy()
            _angle[idx_trafo] = test_angle
            test_trafo = AffineTrafo2d()
            test_trafo.set_origin_axes(
                magnification=guess_magnification, angle=_angle
            )
            # Find trafo fidelity
            _trafo_angle_fidelity = get_trafo_angle_fidelity(
                x, y, test_trafo, ax=idx_proj,
                bins_per_site=bins_per_site,
                peak_rel_prominence=peak_rel_prominence,
                peak_base_prominence_ratio=peak_base_prominence_ratio
            )
            fidelities.append(_trafo_angle_fidelity["fidelity"])
            rel_spacings.append(_trafo_angle_fidelity["spacing"])
            rel_widths.append(_trafo_angle_fidelity["width"])
        # Filter bad fidelities
        mask = np.array(fidelities) > min_fidelity
        fidelities = ArrayData(np.array(fidelities)[mask])
        fidelities.set_data_quantity(name="transformation fidelity")
        fidelities.set_dim(0, points=test_angles[idx_trafo][mask])
        fidelities.set_var_quantity(
            0, name=f"lattice[{idx_trafo:d}] angle", unit="rad"
        )
        test_fidelities.append(fidelities)
        # Get fitted magnifications
        _ad = fidelities.copy()
        _ad.data = np.array(rel_spacings)[mask]
        _ad.set_data_quantity(name="relative spacing")
        test_rel_spacings.append(_ad)
        _ad = fidelities.copy()
        _ad.data = np.array(rel_widths)[mask]
        _ad.set_data_quantity(name="relative width")
        test_rel_widths.append(_ad)
    if print_progress:
        misc.print_progress(2, 2, angle_num, angle_num, start_time=_t0)
    # Fit angles with best fidelities
    angles = []
    for test_fidelity in test_fidelities:
        peaks = find_peaks_1d(
            test_fidelity, npeaks=1, rel_prominence=0, fit_algorithm="gaussian"
        )["center"]
        if len(peaks) < 1:
            peaks = find_peaks_1d(
                test_fidelity, npeaks=1, rel_prominence=0,
                fit_algorithm=FitParabolic1d
            )["center"]
            if len(peaks) < 1:
                peaks = find_peaks_1d(
                    test_fidelity, npeaks=1, rel_prominence=0,
                    fit_algorithm="mean"
                )["center"]
        try:
            angles.append(peaks[0])
        except IndexError:
            raise RuntimeError(
                "get_trafo_from_sites_direct failed. Try higher `angle_num`?"
            )
    # Find spacing at best angles
    optimal_trafo = AffineTrafo2d()
    optimal_trafo.set_origin_axes(
        magnification=guess_magnification, angle=angles, offset=guess_offset
    )
    magnifications = np.flip(get_trafo_angle_fidelity(
        x, y, optimal_trafo,
        bins_per_site=bins_per_site,
        peak_rel_prominence=peak_rel_prominence,
        peak_base_prominence_ratio=peak_base_prominence_ratio
    )["spacing"]) * guess_magnification
    optimal_trafo.set_origin_axes(
        magnification=magnifications, angle=angles, offset=guess_offset
    )
    return {
        "trafo": optimal_trafo,
        "fidelity": test_fidelities,
        "rel_spacing": test_rel_spacings,
        "rel_width": test_rel_widths
    }


def get_trafo_from_sites_fit(x, y, guess_trafo):
    """
    Direct fit of transformation.
    """
    LOGGER.warn("get_trafo_from_sites_fit: Function seems not to work")
    coord = np.array([x, y]).T
    guess_trafo = guess_trafo.invert()

    def _opt_func(var):
        a00, a01, a10, a11, b0, b1 = var
        a = np.array([[a00, a01], [a10, a11]])
        b = np.array([b0, b1])
        f = np.einsum("ij,...j->...i", a, coord) + b
        return np.sum(((f + 0.5) % 1 - 0.5)**2)

    p0_a = np.ravel(guess_trafo.matrix)
    p0_b = np.ravel(guess_trafo.offset)
    res = scipy.optimize.minimize(_opt_func, x0=np.concatenate([p0_a, p0_b]))
    if not res.success:
        raise RuntimeError("get_trafo_from_sites_fit failed")
    popt_a = np.reshape(res.x[:4], (2, 2))
    popt_b = res.x[4:]
    trafo = AffineTrafo2d(matrix=popt_a, offset=popt_b)
    return trafo.invert()


###############################################################################
# Transformation estimator
###############################################################################


class TrafoEstimator:

    """
    Class for estimating lattice angles and spacings.

    Uses low-density atomic fluorescence images to optimize the
    affine transformation between sites and image coordinates.

    Parameters
    ----------
    isolated_locator : `IsolatedLocator`
        Object for locating isolated atoms.
    min_isolated_num : `int`
        Minimum detected isolated atoms.
    guess_trafo : `AffineTrafo2d`
        Initially guessed transformation used as a starting point
        for optimization.
    angle_num : `int`
        Number of angular steps used for optimization.
    angle_range : `float`
        Angular range in radians (rad) used for optimization.
    bins_per_site, peak_rel_prominence, peak_base_prominence_ratio : `float`
        Transformation fidelity estimation parameters.
        See :py:func:`get_trafo_angle_fidelity` for details.
    min_fidelity : `float`
        Angular optimization parameter.
        See :py:func:`get_trafo_from_sites_direct` for details.

    Examples
    --------
    Standard use case given a guessed transformation, an isolated
    atoms locator object and low-density images:

    >>> type(guess_trafo)
    libics.tools.trafo.linear.AffineTrafo2d
    >>> type(isoloc)
    srec.est.iso_est.IsolatedLocator
    >>> images.shape
    (10, 512, 512)
    >>> trfest = TrafoEstimator(
    ...     isolated_locator=isoloc,
    ...     guess_trafo=guess_trafo
    ... )
    >>> type(trfest.find_trafo(*images))
    libics.tools.trafo.linear.AffineTrafo2d
    """

    LOGGER = get_logger("srec.TrafoEstimator")

    def __init__(
        self, isolated_locator=None, min_isolated_num=24,
        guess_trafo=None, angle_num=16, angle_range=np.deg2rad(3),
        bins_per_site=5, min_fidelity=0.8,
        peak_rel_prominence=0.1, peak_base_prominence_ratio=0.1,
    ):
        # Isolated atom location estimator
        self.isolated_locator = isolated_locator
        self.min_isolated_num = min_isolated_num
        # Trafo optimization: main configuration
        self.guess_trafo = guess_trafo
        self.angle_num = angle_num
        self.angle_range = angle_range
        # Trafo optimization: detailed configuration
        self.bins_per_site = bins_per_site
        self.min_fidelity = min_fidelity
        self.peak_rel_prominence = peak_rel_prominence
        self.peak_base_prominence_ratio = peak_base_prominence_ratio

    def check_setup(self):
        """
        Checks whether all attributes are set up.
        """
        if self.isolated_locator is None:
            raise RuntimeError("invalid `isolated_locator`")
        if self.guess_trafo is None:
            raise RuntimeError("invalid `guess_trafo`")
        return True

    def get_optimized_trafo(self, image, print_progress=False):
        """
        Gets the optimal transformation object from a single image.

        Parameters
        ----------
        image : `Array[2, float]`
            Low-density image.
        print_progress : `bool`
            Whether to print a progress bar.

        Returns
        -------
        opt_trafo : `AffineTrafo2d`
            Optimized transformation between sites and image coordinates.
        """
        self.check_setup()
        # Find atom centers
        label_centers = self.isolated_locator.get_label_centers(image)
        if len(label_centers[..., 0]) < self.min_isolated_num:
            return False
        # Optimize trafo
        trafo_params = get_trafo_from_sites_direct(
            *np.transpose(label_centers), self.guess_trafo,
            angle_range=self.angle_range, angle_num=self.angle_num,
            bins_per_site=self.bins_per_site,
            peak_rel_prominence=self.peak_rel_prominence,
            peak_base_prominence_ratio=self.peak_base_prominence_ratio,
            min_fidelity=self.min_fidelity, print_progress=print_progress
        )
        opt_trafo = trafo_params["trafo"]
        return opt_trafo

    def find_trafo(self, *images, print_progress=False):
        """
        Gets the optimal transformation object from a multiple images.

        Averages the transformation parameters obtained from each image.

        Parameters
        ----------
        *images : `Array[2, float]`
            Low-density image.
        print_progress : `bool`
            Whether to print a progress bar.

        Returns
        -------
        trafo : `AffineTrafo2d`
            Optimized transformation between sites and image coordinates.
        """
        _iter = misc.iter_progress(images) if print_progress else images
        opt_trafos = []
        for i, im in enumerate(_iter):
            _trafo = self.get_optimized_trafo(im)
            if _trafo is False:
                self.LOGGER.warn(f"Too few isolated atoms in image[{i:d}]")
            else:
                opt_trafos.append(_trafo)
        if len(opt_trafos) == 0:
            raise RuntimeError("No suitable image found")
        _matrix = np.mean([_trafo.matrix for _trafo in opt_trafos], axis=0)
        _offset = np.mean([_trafo.offset for _trafo in opt_trafos], axis=0)
        trafo = AffineTrafo2d(matrix=_matrix, offset=_offset)
        return trafo


###############################################################################
# Transformation phase estimation
###############################################################################

# ++++++++++++++++++++++++++++++++++++++++++++++++++
# From fitted sites
# ++++++++++++++++++++++++++++++++++++++++++++++++++

def get_trafo_phase_from_points(
    x, y, ref_trafo_site_to_image,
    phase_offset_trial_thr=0.2, phase_std_trial_thr=0.2
):
    """
    Gets the transformation phase from image coordinates.

    Parameters
    ----------
    x, y : `Array[1, float]`
        Atom centers in image coordinates.
    ref_trafo_site_to_image : `AffineTrafo2d`
        Zero-phase reference transformation between sites and image.

    Returns
    -------
    phase : `np.ndarray(1, float)`
        Mean phase along [x, y] sites in interval `[-0.5, 0.5]`.
    phase_err : `np.ndarray(1, float)`
        Standard error of the mean of phase.
    """
    image_coords = ref_trafo_site_to_image.coord_to_origin(
        np.moveaxis([np.ravel(x), np.ravel(y)], 0, -1)
    )
    # Try centering phase = 0
    phase_offset = 0.5
    phases = (image_coords + phase_offset) % 1 - phase_offset
    phase = (np.mean(phases, axis=0) + 0.5) % 1 - 0.5
    phase_std = np.std(phases, axis=0)
    # Try centering phase = 0.5
    try_phase_shift = (
        (np.abs(phase) > np.abs(phase_offset_trial_thr))
        | (phase_std > phase_std_trial_thr)
    )
    if np.any(try_phase_shift):
        _phase_offset = (0.5 * (try_phase_shift + 1)) % 1
        _phases = (
            (image_coords + _phase_offset) % 1 - _phase_offset
        )
        _phase = (np.mean(_phases, axis=0) + 0.5) % 1 - 0.5
        _phase_std = np.std(_phases, axis=0)
        if np.all(_phase_std <= phase_std):
            phase, phase_std = _phase, _phase_std
    # Use optimized phase
    phase_offset = 0.5 - phase
    phases = (image_coords + phase_offset) % 1 - phase_offset
    phase = (np.mean(phases, axis=0) + 0.5) % 1 - 0.5
    phase_std = np.std(phases, axis=0)
    phase_err = phase_std / np.sqrt(phases.shape[1])
    return -phase, phase_err


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# From maximizing projection contrast
# ++++++++++++++++++++++++++++++++++++++++++++++++++

def get_shifted_subimage_trafo(trafo, shift, subimage_center, site=(0, 0)):
    """
    Gets the lattice transformation from subimage center and shift.
    """
    shifted_trafo = copy.deepcopy(trafo)
    shifted_trafo.set_offset_by_point_pair(
        site, np.array(subimage_center) + np.array(shift)
    )
    return shifted_trafo


def get_subsite_shape(prjgen, subimage_shape, min_shape=(3, 3)):
    """
    Gets the default subimage sites shape.
    """
    magnification = prjgen.trafo_site_to_image.get_origin_axes()[0]
    subsite_size = np.round(
        np.mean(subimage_shape) / np.mean(magnification)
    ).astype(int)
    return (
        max(subsite_size, min_shape[0]),
        max(subsite_size, min_shape[1])
    )


def get_subimage_emission_std(
    shift, subimage_center, full_image, prjgen, subsite_shape=(5, 5)
):
    """
    Performs projection with given transformation shift and subimage.
    """
    # Parse parameters
    shift = np.array(shift)
    tmp_prjgen = copy.deepcopy(prjgen)
    if np.isscalar(subsite_shape):
        subsite_shape = (subsite_shape, subsite_shape)
    # Set sites
    subsite_1d = [
        np.arange(-subsite_size // 2, (subsite_size + 1) // 2 + 1)
        for subsite_size in subsite_shape
    ]
    subsite_coords = np.moveaxis(
        np.meshgrid(*subsite_1d, indexing="ij"), 0, -1
    )
    subsite_coords = np.reshape(subsite_coords, (-1, 2))
    # Find coordinates
    _trafo = get_shifted_subimage_trafo(
        tmp_prjgen.trafo_site_to_image, shift, subimage_center, site=(0, 0)
    )
    subimage_coords = _trafo.coord_to_target(subsite_coords).T
    # Keep only sites within image
    _half_proj_shape = np.array(prjgen.proj_shape) // 2
    mask = np.logical_and.reduce([
        (subimage_coords[i] > _half_proj_shape[i] + 1)
        & (subimage_coords[i] < full_image.shape[i] - _half_proj_shape[i] - 1)
        for i in range(2)
    ])
    subimage_coords = np.array([_c[mask] for _c in subimage_coords])
    # Find local images
    tmp_prjgen.trafo_site_to_image = _trafo
    local_images = get_local_images(
        *subimage_coords, full_image, tmp_prjgen.proj_shape,
        psf_supersample=tmp_prjgen.psf_supersample
    )
    # Perform projection
    emissions = apply_projectors(local_images, tmp_prjgen)
    return np.std(emissions)


def get_trafo_phase_from_projections(
    im, prjgen, phase_ref_image=(0, 0),
    subimage_shape=None, subsite_shape=None, search_range=1
):
    """
    Gets the lattice phase by maximizing the emission standard deviation.

    Parameters
    ----------
    im : `Array[2, float]`
        Fluorescence image.
    prjgen : `srec.ProjectionGenerator`
        Projection generator object.
    phase_ref_image : `(int, int)`
        Lattice phase reference in fluorescence image coordinates.
    subimage_shape : `(int, int)` or `None`
        Shape of subimages (subdivisions of full image) used for
        standard deviation evaluation.
    subsite_shape : `(int, int)` or `None`
        Shape of sites used for projection.
    search_range : `int`
        Discrete optimization search range.
        See :py:func:`libics.tools.math.optimize.minimize_discrete_stepwise`.

    Returns
    -------
    phase : `np.ndarray(1, float)`
        Phase (residual) of lattice w.r.t. to image coordinates (0, 0).
    """
    # Parse parameters
    if subimage_shape is None:
        subimage_shape = np.copy(prjgen.psf_shape)
    # Cropped image to avoid checking image borders
    proj_shape = np.array(prjgen.proj_shape)
    im_roi = im[proj_shape[0]:-proj_shape[0], proj_shape[1]:-proj_shape[1]]
    crop_shape = (np.array(im_roi.shape) // subimage_shape) * subimage_shape
    im_crop = im_roi[tuple(slice(None, _s) for _s in crop_shape)]
    grid_shape = crop_shape // subimage_shape
    # Get subimage with maximum signal variance as proxy for mixed filling
    subimages = np.reshape(
        im_crop,
        (grid_shape[0], subimage_shape[0], grid_shape[1], subimage_shape[1])
    )
    subimages_std = np.std(subimages, axis=(1, 3))
    idx = np.unravel_index(np.argmax(subimages_std), subimages_std.shape)
    subimage_center = ((np.array(idx) + 0.5) * subimage_shape).astype(int)
    subimage_center += proj_shape

    # Get phase by maximizing projected emissions variance
    if subsite_shape is None:
        subsite_shape = get_subsite_shape(prjgen, subimage_shape)
    init_shift = [0, 0]
    # Maximize on integer pixels
    opt_shift_int, results_cache = maximize_discrete_stepwise(
        get_subimage_emission_std, init_shift,
        args=(subimage_center, im, prjgen),
        kwargs=dict(subsite_shape=subsite_shape),
        dx=1, search_range=search_range, ret_cache=True
    )
    # Maximize on subpixels
    opt_shift_float = maximize_discrete_stepwise(
        get_subimage_emission_std, opt_shift_int,
        args=(subimage_center, im, prjgen),
        kwargs=dict(subsite_shape=subsite_shape),
        dx=1/prjgen.psf_supersample/2, search_range=search_range,
        results_cache=results_cache
    )
    # Calculate phase
    opt_trafo = get_shifted_subimage_trafo(
        prjgen.trafo_site_to_image, opt_shift_float, subimage_center
    )
    phase, _ = trafo_gen.get_phase_from_trafo_site_to_image(
        opt_trafo, phase_ref_image=phase_ref_image
    )
    return phase


###############################################################################
# Helper
###############################################################################


class FitPeriodicGaussian1d(ModelBase):

    P_ALL = ["a", "x0", "wx", "dx", "c"]
    P_DEFAULT = [1, 0, 1, 1, 0]

    def _func(self, var, *p):
        a, x0, wx, dx, c = p
        x0 = x0 % dx
        num_left = (np.min(var) - x0 - 1) // dx
        num_gaussian = (4 * wx + np.max(var) - np.min(var)) / dx
        num_gaussian = max(1, np.round(num_gaussian).astype(int))
        xcs = np.arange(num_left, num_gaussian - num_left) * dx + x0
        return np.sum(
            [gaussian_1d(var, a, xc, wx, 0) for xc in xcs], axis=0
        ) + c

    def find_p0(self, *data):
        var_data, func_data, _ = self._split_fit_data(*data)
        peaks = find_peaks_1d(
            var_data.ravel(), func_data, rel_prominence=0,
            base_prominence_ratio=0.1, edge_peaks=False,
            fit_algorithm="mean",
            ret_vals=["width"]
        )
        _centers = np.sort(peaks["center"])
        x0 = _centers[0]
        dx = np.mean(_centers[1:] - _centers[:-1])
        wx = np.nanmean(peaks["width"])
        c = np.min(func_data)
        a = (np.max(func_data) - c) / 2
        self.p0 = [a, x0, wx, dx, c]
