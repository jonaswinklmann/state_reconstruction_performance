"""
Image preprocessing

Scales raw images to account for sensitivity inhomogeneities and
removes outliers.
"""

import numpy as np

from libics.env.logging import get_logger
from libics.core.util import misc
from libics.core.data.types import AttrHashBase
from libics.core import io


###############################################################################
# Image scaling
###############################################################################


def prescale_image(im, scale):
    """
    Scales an image by a sensitivity array.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    scale : `Array[2, float]`
        Sensitivity scale. Must have same shape as `im`.

    Returns
    -------
    im_new : `Array[2, float]`
        Scaled image.
    """
    return im / scale


###############################################################################
# Outlier removal
###############################################################################


def analyze_image_outlier(im, outlier_size=(5, 5), min_ref_val=5):
    """
    Analyzes an image for negative and positive outliers.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    outlier_size : `(int, int)`
        Expected size of potential outliers.
    min_ref_val : `float`
        Minimum reference value to be considered valid outlier.

    Returns
    -------
    outlier_ratios : `np.ndarray([float, float])`
        Ratio between (background-subtracted) image minimum/maximum
        and reference, which is the `product(outlier_size)`-th
        smallest/largest image value.
        The background is the image median.
    outlier_idxs : `[(int, int), (int, int)]`
        Image coordinates of minimum/maximum pixel:
        `[(xmin, ymin), (xmax, ymax)]`.
    ar_bg : `float`
        Background (i.e. median) of image.
    """
    im = np.array(im)
    ar = np.ravel(im)
    order = np.argsort(ar)
    ar_min, ar_ref_min = ar[order[0]], ar[order[np.prod(outlier_size)]]
    ar_max, ar_ref_max = ar[order[-1]], ar[order[-1 - np.prod(outlier_size)]]
    ar_bg = ar[order[ar.size // 2]]
    outlier_ratios = np.array([
        (ar_bg - ar_min) / max(min_ref_val, ar_bg - ar_ref_min),
        (ar_max - ar_bg) / max(min_ref_val, ar_ref_max - ar_bg)
    ])
    outlier_idxs = [
        np.unravel_index(order[0], im.shape),
        np.unravel_index(order[-1], im.shape)
    ]
    return outlier_ratios, outlier_idxs, ar_bg


def remove_image_outlier(im, outlier_idx, outlier_size=(3, 3), val=None):
    """
    Removes an outlier from an image.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    outlier_idx : `(int, int)`
        Image coordinates of central outlier pixel.
    outlier_size : `(int, int)`
        Removal area around outlier.
    val : `float`
        Replacement value.

    Returns
    -------
    im_new : `Array[2, float]`
        Copy of image with outlier area set to `val`.
    """
    im = im.copy()
    if val is None:
        val = np.median(np.ravel(im))
    im[misc.cv_index_center_to_slice(outlier_idx, outlier_size)] = val
    return im


def process_image_outliers_recursive(
    im, max_outlier_ratio=None, min_ref_val=5, outlier_size=(5, 5), max_its=2,
    _it=0
):
    """
    Recursively checks for outliers and enlarges the outlier size if necessary.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    max_outlier_ratio : `float` or `None`
        Maximum allowed outlier ratio.
        If `None`, does not apply removal.
    min_ref_val : `float`
        Minimum reference value to be considered valid outlier.
    outlier_size : `(int, int)`
        Removal area around outlier.
    max_its : `int`
        Maximum number of iterations.
    _it : `int`
        Current iteration.

    Returns
    -------
    res : `dict(str->Any)`
        Analysis results containing the items:
    image : `Array[2, float]`
        Image with outliers removed.
    iterations : `int`
        Number of iterations applied.
        (`iterations == 1` means no recursive call.)
    outlier_ratios : `np.ndarray([float, float])`
        Minimum/maximum outlier ratios for last iteration.
    outlier_size : `(int, int)`
        Removal area for last iteration.
    """
    # Analyze outlier
    outlier_ratios, outlier_idxs, bg = analyze_image_outlier(
        im, outlier_size=outlier_size,
        min_ref_val=min_ref_val
    )
    if (
        max_outlier_ratio is None or
        np.all(outlier_ratios <= max_outlier_ratio) or _it >= max_its
    ):
        return dict(
            image=im, iterations=_it,
            outlier_ratios=outlier_ratios,
            outlier_size=np.array(outlier_size)
        )
    else:
        # Remove outliers
        for i, outlier_ratio in enumerate(outlier_ratios):
            if outlier_ratio > max_outlier_ratio:
                im = remove_image_outlier(
                    im, outlier_idxs[i],
                    outlier_size=np.array(outlier_size),
                    val=bg
                )
        # Call function to get analysis
        return process_image_outliers_recursive(
            im, max_outlier_ratio, min_ref_val=min_ref_val,
            outlier_size=outlier_size, max_its=max_its,
            _it=_it+1
        )


###############################################################################
# Image preprocessing
###############################################################################


class ImagePreprocessor(AttrHashBase):

    """
    Class for preprocessing raw images.

    Checks for and removes image outliers.

    Parameters
    ----------
    scale : `Array[2, float]` or `str`
        Image amplitude prescaling.
        Must have same shape as images to be processed.
        If `str`, `scale` is interpreted as file path from which the image
        is loaded.
    outlier_size : `(int, int)`
        Area around outlier over which the outlier is analyzed and removed.
    max_outlier_ratio : `float`
        Maximum accepted ratio between outlier and non-outlier maximum.
    outlier_min_ref_val : `float`
        Minimum reference value to be considered valid outlier.
    outlier_iterations : `int`
        Maximum number of outlier removal iterations.
    """

    LOGGER = get_logger("srec.ImagePreprocessor")
    HASH_KEYS = AttrHashBase.HASH_KEYS | {
        "scale", "outlier_size", "max_outlier_ratio",
        "outlier_min_ref_val", "outlier_iterations"
    }

    def __init__(
        self, scale=None,
        outlier_size=(5, 5), max_outlier_ratio=5, outlier_min_ref_val=5,
        outlier_iterations=2
    ):
        if isinstance(scale, str):
            scale = io.load(scale)
        self.scale = scale
        self.outlier_size = outlier_size
        self.max_outlier_ratio = max_outlier_ratio
        self.outlier_min_ref_val = outlier_min_ref_val
        self.outlier_iterations = outlier_iterations

    def get_attr_str(self):
        s = []
        if self.scale is not None:
            _scale_mean = f"{np.mean(self.scale):.2f}"
            _scale_shape = ("scalar" if np.isscalar(self.scale)
                            else str(self.scale.shape))
            s.append(f" → scale: mean: {_scale_mean}, shape: {_scale_shape}")
        for k in ["outlier_size", "max_outlier_ratio", "outlier_iterations"]:
            s.append(f" → {k}: {str(getattr(self, k))}")
        return "\n".join(s)

    def __str__(self):
        return f"{self.__class__.__name__}:\n{self.get_attr_str()}"

    def __repr__(self):
        s = f"<'{self.__class__.__name__}' at {hex(id(self))}>"
        return f"{s}\n{self.get_attr_str()}"

    def process_image(self, im):
        """
        Performs image preprocessing (scaling and outlier detection).

        Parameters
        ----------
        im : `Array[2, float]`
            Raw image.
        ret_success : `bool`
            Whether to return a success flag.

        Returns
        -------
        im : `Array[2, float]`
            Processed image.
        outlier_ratio : `bool`
            Outlier ratio.
        """
        # Prescale image
        if self.scale is not None:
            im = prescale_image(im, self.scale)
        # Process outliers
        outliers = process_image_outliers_recursive(
            im, self.max_outlier_ratio, min_ref_val=self.outlier_min_ref_val,
            outlier_size=self.outlier_size, max_its=self.outlier_iterations
        )
        im = outliers["image"]
        outlier_ratios = outliers["outlier_ratios"]
        outlier_size = outliers["outlier_size"]
        outlier_iterations = outliers["iterations"]
        s_outl_ratios = misc.cv_iter_to_str(outlier_ratios, fmt="{:.1f}")
        s_outl_size = misc.cv_iter_to_str(outlier_size, fmt="{:.0f}")
        _msg = (
            f"(ratios: {s_outl_ratios:s}, size: {s_outl_size}, "
            f"iterations: {outlier_iterations:d})"
        )
        if outlier_iterations == 1:
            self.LOGGER.debug(f"process_image: No outlier detected {_msg}")
        else:
            if np.any(outlier_ratios > self.max_outlier_ratio):
                self.LOGGER.warning(
                    f"process_image: Outlier removal failed {_msg}"
                )
            else:
                self.LOGGER.info(f"process_image: Outlier detected {_msg}")
        return im, outlier_ratios
