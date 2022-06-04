"""
Subpixel PSF estimator.

Uses images of isolated atoms to find a subpixel point spread function.
"""

import numpy as np

from libics.core.data.arrays import ArrayData
from libics.core.util import misc

from .iso_est import IsolatedLocator


###############################################################################
# Subpixel alignment
###############################################################################


def find_label_regions_subpixel(
    image, label_centers, region_size, supersample=5,
    interpolation="linear", normalize=False, rmv_invalid_shape=True
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
    interpolation : `str`
        Supersampling interpolation. Options: `"nearest", "linear"`.
    normalize : `bool`
        Whether to normalize each subregion image.
    rmv_invalid_shape : `bool`
        Whether to remove invalid shapes.

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
            im_int = ArrayData(image[roi_int])
            if rmv_invalid_shape:
                if not np.allclose(np.array(im_int.shape), region_size + 2):
                    continue
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
                if interpolation == "linear":
                    im_float = im_int.interpolate(
                        im_float.get_var_meshgrid(),
                        mode="linear", extrapolation=0
                    )
            if normalize:
                im_float /= np.sum(im_float)
            label_regions.append(im_float)
    return np.array(label_regions)


###############################################################################
# Point spread function estimator
###############################################################################


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

    def __init__(
        self, psf_supersample=5, psf_interpolation="linear", **kwargs
    ):
        super().__init__(**kwargs)
        self.psf_supersample = psf_supersample
        self.psf_interpolation = psf_interpolation

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
            interpolation=self.psf_interpolation,
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
