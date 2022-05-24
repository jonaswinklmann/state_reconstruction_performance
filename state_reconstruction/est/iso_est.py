"""
Isolated atom estimator.

Filters an image to find isolated atoms.
"""

import numpy as np
import scipy.ndimage

from libics.core.util import misc
from libics.core.data.arrays import ArrayData
from libics.tools.math.signal import find_peaks_1d
from libics.tools import plot
from libics.tools.math.peaked import FitGaussian2dTilt


###############################################################################
# Isolated atom estimator
###############################################################################


def get_onsite_filter(psf_integrated, inner_thr=1/np.e):
    _ar = np.array(psf_integrated) - inner_thr * np.max(psf_integrated)
    _ar[_ar < 0] = 0
    _ar /= np.sum(_ar)
    return _ar


def get_neighbor_filter(psf_integrated, inner_thr=1/np.e, outer_thr=1/np.e**4):
    _ar = np.array(psf_integrated) / np.max(psf_integrated)
    _inner_mask = _ar > inner_thr
    _outer_mask = _ar < outer_thr
    _ar = inner_thr - _ar
    _ar[_inner_mask | _outer_mask] = 0
    _ar /= np.sum(_ar)
    return _ar


def apply_filter(image, filter):
    """
    Performs convolutional filter.

    Parameters
    ----------
    image : `Array[2, float]`
        Raw image.
    filter : `Array[2, float]`
        Filter kernel.

    Returns
    -------
    image : `Array[2, float]`
        Filtered image.
    """
    return scipy.ndimage.convolve(image, filter)


def interpret_filter_hist(
    image_filtered, bins=None,
    split_cond_width=None, split_cond_center=None,
    _DEBUG=False
):
    """
    Gets a boolean mask by analyzing filtered data.

    Generates a histogram and fits two peaks.
    The masking threshold is determined from either the distance
    between the peaks or the width of the first peak.

    Parameters
    ----------
    image_filtered : `Array[2, float]`
        Filtered image.
    bins : `int`
        Number of histogram bins
    split_cond_width : `float`
        Thresholding on the width of the first peak.
        Values above `split_cond_width` relative to the width
        of the first peak are set to `True`.
    split_cond_center : `float`
        Thresholding on the position between the two peak centers.
        Values above `split_cond_center` relative to the distance
        between the peaks are set to `True`.

    Returns
    -------
    mask : `np.ndarray(2, bool)`
        Mask array.
    """
    # Parse parameters
    _im = np.ravel(image_filtered)
    if bins is None:
        bins = min(128, max(32, _im.size // 2048))
    if split_cond_width is None and split_cond_center is None:
        raise ValueError("No split condition given")
    elif split_cond_width is not None and split_cond_center is not None:
        raise ValueError("Overspecified split condition")
    elif split_cond_width is True:
        split_cond_width = 5
    elif split_cond_center is True:
        split_cond_center = 0.5
    # Construct histogram
    _h, _e = np.histogram(_im, bins=bins)
    _c = (_e[1:] + _e[:-1]) / 2
    peaks = find_peaks_1d(
        _c, _h, npeaks=2, rel_prominence=0, check_npeaks="error",
        base_prominence_ratio=0.1, ret_vals=["width"]
    )
    # Apply split condition
    if split_cond_width:
        split_val = peaks["center"][0] + peaks["width"][0] * split_cond_width
    elif split_cond_center:
        split_val = np.sum(
            np.array([split_cond_center, 1 - split_cond_center])
            * peaks["center"]
        )
    if _DEBUG:
        plot.scatter(_c, _h)
        for _x in peaks["center"]:
            plot.axvline(_x, color="grey")
        plot.axvline(split_val, color="red")
    return np.array(image_filtered) >= split_val


def find_labels(mask, image):
    """
    Labels and evaluates a masked image.

    Parameters
    ----------
    mask : `Array[2, bool]`
        Mask array. `True` represents labelled space.
    image : `Array[2, float]`
        Original image.

    Returns
    -------
    ret : `dict(str->Any)`
        Returns a dictionary containing the following items:
    labels : `np.ndarray(2, int)`
        Label array. Unlabelled areas have value `0`.
        Labelled unconnected areas have incrementing integers
        at their respective positions.
    label_size : `np.ndarray(1, int)`
        Number of points of each label.
    label_com : `np.ndarray(1, float)`
        Center of mass of each label.
    """
    labels, label_num = scipy.ndimage.label(mask)
    _iter = np.arange(1, label_num + 1)
    label_sizes = np.array([
        np.count_nonzero(labels == i) for i in _iter
    ])
    label_coms = np.array([
        np.round(scipy.ndimage.center_of_mass(image, labels=labels, index=i))
        for i in _iter
    ], dtype=int)
    return {
        "labels": labels, "label_size": label_sizes, "label_com": label_coms
    }


def find_label_centers_gaussian(
    image, label_coms, im_size,
    max_ellipticity=0.2, rel_center_tol=1, rel_width_tol=2
):
    """
    Fits a Gaussian to each labelled area.

    Parameters
    ----------
    image : `Array[2, float]`
        Image.
    label_coms : `Array[1, float]`
        Center of mass of labelled areas.
    im_size : `(int, int)`
        Image size used for alignment.
    max_ellipticity : `float`
        Maximally allowed ellipticity.
    rel_center_tol : `float`
        Acceptable positional difference between fitted center and
        center of mass (relative to `im_size`).
    rel_width_tol : `float`
        Acceptable fitted width relative to `im_size`.

    Returns
    -------
    label_centers : `np.ndarray(2, float)`
        Fitted Gaussian center of labelled areas. Labels for which
        the fit failed have a label center of `(np.nan, np.nan)`.
        Dimensions: `[n_labels, ndim]`.
    """
    image = ArrayData(image)
    if np.isscalar(im_size):
        im_size = (im_size, im_size)
    im_size = np.array(im_size)
    center_tol, width_tol = im_size * rel_center_tol, im_size * rel_width_tol
    label_centers = []
    for label_com in label_coms:
        # Image resizing for Gaussian fit
        _roi = misc.cv_index_center_to_slice(label_com, im_size)
        # Gaussian fit for centering
        try:
            _fit = FitGaussian2dTilt(image[_roi])
            _center = np.array([_fit.x0, _fit.y0])
            _width = np.array([_fit.wu, _fit.wv])
            if (
                _fit.psuccess
                and np.all(np.abs(_center - label_com) <= center_tol)
                and np.all(np.abs(_width) <= width_tol)
                and _fit.ellipticity <= max_ellipticity
            ):
                label_centers.append(_center)
            else:
                label_centers.append(np.full(2, np.nan))
        except TypeError:
            label_centers.append(np.full(2, np.nan))
    label_centers = np.array(label_centers)
    return label_centers


def find_label_regions_subpixel(
    image, label_centers, region_size, supersample=5, normalize=False
):
    """
    Get subpixel-aligned supersampled subregions.

    Parameters
    ----------
    image : `Array[2, float]`
        Original image.
    label_centers : `Array[2, float]`
        (Fractional) label centers. Dimensions: `[n_labels, ndim]`.
    region_size : `(int, int)`
        Subregion size in original pixels.
    supersample : `int`
        Supersampling.
    normalize : `bool`
        Whether to normalize each subregion image.

    Returns
    -------
    label_regions : `np.ndarray(3, float)`
        Subpixel-aligned supersampled image subregions.
        Dimensions: `[n_labels, x, y]`.
    """
    if np.isscalar(region_size):
        region_size = (region_size, region_size)
    region_size = np.array(region_size)
    label_regions = []
    for label_center in label_centers:
        if not np.any(np.isnan(label_center)):
            # Crop image
            center_int = np.round(label_center).astype(int)
            roi_int = misc.cv_index_center_to_slice(
                center_int, region_size + 2
            )
            im_int = image[roi_int]
            # Supersample image
            if supersample is None:
                im_float = im_int
            else:
                dcenter = np.round(
                    supersample * (label_center - center_int)
                ).astype(int)
                roi_float = (
                    slice(supersample + dcenter[0], -supersample + dcenter[0]),
                    slice(supersample + dcenter[1], -supersample + dcenter[1])
                )
                im_float = im_int.supersample(supersample)[roi_float]
            if normalize:
                im_float /= np.sum(im_float)
            label_regions.append(im_float)
    return np.array(label_regions)


###############################################################################
# Point spread function estimator
###############################################################################


class IsolatedLocator:

    """
    Class for locating isolated atoms.

    Due to overlapping PSFs of neighboring atoms, the transformation phase
    might not be detected directly. This class allows to find the image
    coordinates of isolated atoms, whose positions may be used to, e.g.,
    extract the lattice phase.

    Parameters
    ----------
    psf_integrated : `Array[2, float]`
        Binned PSF array.
    filter_inner_thr, filter_outer_thr : `float`
        Filter kernel parameters.
    onsite_split_cond_width, neighbor_split_cond_width : `float`
        Histogram width condition for determining mask.
        Parameters given for the onsite-present filter
        and the neighbor-present filter.
        See :py:func:`interpret_filter_hist` for details.
    label_center_rel_im_size : `float`
        Label center subregion size relative to the
        Gaussian `1/e` radius.
    label_center_max_ellipticity : `float`
        Maximum ellipticity for subregion to be designated as isolated atom.
    label_center_tol : `(float, float)`
        Maximum `(center, width)` fit tolerance relative to the subregion size
        to be designated as isolated atom.

    Examples
    --------
    Standard use case given a binned PSF and an image:

    >>> psf_integr.shape
    (21, 21)
    >>> image.shape
    (512, 512)
    >>> isoloc = IsolatedLocator(
    ...     psf_integrated=psf_integr
    ... )
    >>> isoloc.setup()
    >>> isoloc.get_label_centers(image).shape
    (7, 2)
    """

    def __init__(
        self, psf_integrated=None,
        filter_inner_thr=1/np.e, filter_outer_thr=1/np.e**4,
        onsite_split_cond_width=5, neighbor_split_cond_width=3,
        label_center_rel_im_size=1,
        label_center_max_ellipticity=0.2, label_center_tol=(1, 2)
    ):
        # Point spread function configuration
        self.psf_integrated = psf_integrated
        # Filtering configuration
        self.filter_inner_thr = filter_inner_thr
        self.filter_outer_thr = filter_outer_thr
        self.onsite_split_cond_width = onsite_split_cond_width
        self.neighbor_split_cond_width = neighbor_split_cond_width
        # Labelling configuration
        self.label_center_rel_im_size = label_center_rel_im_size
        self.label_center_max_ellipticity = label_center_max_ellipticity
        self.label_center_tol = label_center_tol
        # Derived quantities
        self._is_set_up = False
        self._label_fit_im_size = None

    def setup(self):
        """
        Fits a Gaussian to the PSF to determine the label subregion size.
        """
        _fit = FitGaussian2dTilt(
            self.psf_integrated, const_p0=dict(tilt=0)
        )
        self._label_fit_im_size = np.round(
            np.array([_fit.wu, _fit.wv]) * 2 * self.label_center_rel_im_size
        ).astype(int)
        self._is_set_up = True

    def get_label_centers(self, image, remove_nan=True):
        """
        Gets the centers of isolated atoms from an image.

        Parameters
        ----------
        image : `Array[2, float]`
            Image to be analyzed.
        remove_nan : `bool`
            Whether to remove invalid center values.

        Returns
        -------
        label_centers : `np.ndarray(2, float)`
            Fitted Gaussian center of labelled areas. Labels for which
            the fit failed have a label center of `(np.nan, np.nan)`.
            Dimensions: `[n_labels, ndim]`.
        """
        if not self._is_set_up:
            self.setup()
        # Filter images
        onsite_filter = get_onsite_filter(
            self.psf_integrated, inner_thr=self.filter_inner_thr
        )
        neighbor_filter = get_neighbor_filter(
            self.psf_integrated, inner_thr=self.filter_inner_thr,
            outer_thr=self.filter_outer_thr
        )
        onsite_present = apply_filter(image, onsite_filter)
        neighbor_present = apply_filter(image, neighbor_filter)
        onsite_mask = interpret_filter_hist(
            onsite_present, split_cond_width=self.onsite_split_cond_width
        )
        neighbor_mask = interpret_filter_hist(
            neighbor_present, split_cond_width=self.neighbor_split_cond_width
        )
        isolated_mask = onsite_mask & ~neighbor_mask
        # Label images
        labeld = find_labels(isolated_mask, image)
        label_centers = find_label_centers_gaussian(
            image, labeld["label_com"], self._label_fit_im_size,
            max_ellipticity=self.label_center_max_ellipticity,
            rel_center_tol=self.label_center_tol[0],
            rel_width_tol=self.label_center_tol[1]
        )
        if remove_nan and len(label_centers) > 0:
            label_centers = label_centers[~np.isnan(label_centers[..., 0])]
        return label_centers


class SupersamplePsfEstimator(IsolatedLocator):

    """
    Class for estimating a subpixel PSF.

    Using multiple sparse images, the fluorescence image of isolated atoms
    is used to estimate the PSF by supersampling and subpixel alignment.

    Parameters
    ----------
    psf_supersample : `int`
        Supersampling size.

    Examples
    --------
    Standard use case given a binned initial-guess PSF and multiple images:

    >>> guess_psf_integr.shape
    (21, 21)
    >>> images.shape
    (10, 512, 512)
    >>> psfest = SupersamplePsfEstimator(
    ...     psf_integrated=guess_psf_integr,
    ...     psf_supersample=5
    ... )
    >>> psfest.setup()
    >>> psfest.get_psf(*images).shape
    (105, 105)
    """

    def __init__(self, psf_supersample=5, **kwargs):
        super().__init__(**kwargs)
        self.psf_supersample = psf_supersample

    def get_label_regions(self, image, normalize=False, remove_nan=True):
        """
        Gets a list of supersampled and subpixel-aligned image subregions.

        Parameters
        ----------
        image : `Array[2, float]`
            Original image.
        normalize : `bool`
            Whether to normalize each returned image subregion.
        remove_nan : `bool`
            Whether to remove all invalid subregions.

        Returns
        -------
        label_regions : `np.ndarray(3, float)`
            Subpixel-aligned supersampled image subregions.
            Dimensions: `[n_labels, x, y]`.
        """
        label_centers = self.get_label_centers(image, remove_nan=remove_nan)
        label_regions = find_label_regions_subpixel(
            image, label_centers,
            region_size=self.psf_integrated.shape,
            supersample=self.psf_supersample,
            normalize=normalize
        )
        return label_regions

    def get_psf(self, *images, print_progress=False):
        """
        Gets the estimated subpixel PSF from sparse images.

        Parameters
        ----------
        *images : `Array[2, float]`
            Fluorescence images of well-separated atoms.
        print_progress : `bool`
            Whether to print a progress bar.

        Returns
        -------
        label_region : `np.ndarray(2, float)`
            Estimated subpixel PSF.
        """
        _iter = images
        if print_progress:
            _iter = misc.iter_progress(_iter)
        label_regions = np.concatenate([
            self.get_label_regions(image, normalize=True)
            for image in _iter
        ])
        label_region = np.mean(label_regions, axis=0)
        return label_region
