"""
Image preprocessing

Scales raw images to account for sensitivity inhomogeneities and
removes outliers.
"""

import numpy as np

from libics.env.logging import get_logger
from libics.core.util import misc
from libics.core.data.types import AttrHashBase


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


def analyze_image_outlier(im, outlier_size=(5, 5)):
    """
    Analyzes an image for outliers.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    outlier_size : `(int, int)`
        Expected size of potential outliers.

    Returns
    -------
    outlier_ratio : `float`
        Ratio between (background-subtracted) image maximum and reference.
        The reference is the `product(outlier_size)`-th largest image value.
        The background is the image median.
    outlier_idx : `(int, int)`
        Image coordinates of maximum pixel.
    ar_bg : `float`
        Background (i.e. median) of image.
    """
    im = np.array(im)
    ar = np.ravel(im)
    order = np.flip(np.argsort(ar))
    ar_max, ar_ref = ar[order[0]], ar[order[np.prod(outlier_size)]]
    ar_bg = ar[order[ar.size // 2]]
    outlier_ratio = (ar_max - ar_bg) / (ar_ref - ar_bg)
    outlier_idx = np.unravel_index(order[0], im.shape)
    return outlier_ratio, outlier_idx, ar_bg


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


###############################################################################
# Image preprocessing
###############################################################################


class ImagePreprocessor(AttrHashBase):

    """
    Class for preprocessing raw images.

    Checks for and removes image outliers.

    Parameters
    ----------
    scale : `Array[2, float]`
        Image amplitude prescaling.
        Must have same shape as images to be processed.
    outlier_size : `(int, int)`
        Area around outlier over which the outlier is analyzed and removed.
    max_outlier_ratio : `float`
        Maximum accepted ratio between outlier and non-outlier maximum.
    """

    LOGGER = get_logger("srec.ImagePreprocessor")
    HASH_KEYS = AttrHashBase.HASH_KEYS | {
        "scale", "outlier_size", "max_outlier_ratio"
    }

    def __init__(
        self, scale=None,
        outlier_size=(5, 5), max_outlier_ratio=5
    ):
        self.scale = scale
        self.outlier_size = outlier_size
        self.max_outlier_ratio = max_outlier_ratio

    def get_attr_str(self):
        s = []
        if self.scale is not None:
            _scale_mean = f"{np.mean(self.scale):.2f}"
            _scale_shape = ("scalar" if np.isscalar(self.scale)
                            else str(self.scale.shape))
            s.append(f" → scale: mean: {_scale_mean}, shape: {_scale_shape}")
        for k in ["outlier_size", "max_outlier_ratio"]:
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
        if self.scale is not None:
            im = prescale_image(im, self.scale)
        outlier_ratio, outlier_idx, bg = analyze_image_outlier(
            im, outlier_size=self.outlier_size
        )
        if self.max_outlier_ratio is not None:
            if outlier_ratio > self.max_outlier_ratio:
                self.LOGGER.info(f"process_image: Outlier detected with "
                                 f"ratio {outlier_ratio:.1f}")
                im = remove_image_outlier(
                    im, outlier_idx, outlier_size=self.outlier_size, val=bg
                )
                outlier_ratio, _, _ = analyze_image_outlier(
                    im, outlier_size=self.outlier_size
                )
                if outlier_ratio > self.max_outlier_ratio:
                    self.LOGGER.warning(
                        "process_image: Outlier removal failed"
                    )
            else:
                self.LOGGER.debug(f"process_image: No outlier detected "
                                  f"(ratio: {outlier_ratio:.1f})")
        return im, outlier_ratio
