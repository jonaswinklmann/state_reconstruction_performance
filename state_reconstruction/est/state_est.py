"""
State estimator.

Assigns emission states to projections.
"""

import sys
import os
os.add_dll_directory(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__) + "/../")
import state_reconstruction_cpp
from datetime import datetime

import copy
import json
import PIL
import matplotlib as mpl
import numpy as np
import os
import scipy.optimize
import scipy.ndimage
from uuid import uuid4 as uuid

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData
from libics.core import io
from libics.core.util import misc, path
from libics.tools.math.peaked import FitGaussian1d
from libics.tools.math.signal import (
    find_histogram, find_peaks_1d_prominence, analyze_single_peak, PeakInfo
)
from libics.tools.trafo.linear import AffineTrafo2d
from libics.tools import plot

from state_reconstruction.config import get_config
from state_reconstruction.gen import trafo_gen, image_gen, proj_gen
from .image_est import ImagePreprocessor
from .iso_est import IsolatedLocator
from .proj_est import get_local_images, apply_projectors
from .trafo_est import (
    get_trafo_phase_from_points, get_trafo_phase_from_projections
)

LOGGER = get_logger("srec.est.state_est")


###############################################################################
# Emission histogram analysis
###############################################################################


def get_emission_histogram(
    emissions, bin_range=None
):
    """
    Generates a histogram from the projected images.

    Parameters
    ----------
    emissions : `Array[1, float]`
        Projected image subregions.
    bin_range : `None` or `(float, float)`
        Bin range used to obtain the histogram.

    Returns
    -------
    hist : `ArrayData(1, float)`
        Projected image histogram.
    """
    emissions = np.ravel(np.array(emissions))
    # Create histogram
    size = emissions.size
    if size < 64:
        raise RuntimeError("Size of `emissions` too small")
    bins = min(size / 6, np.sqrt(size) / 2)
    bins = np.round(bins).astype(int)
    hist = find_histogram(emissions, bins=bins, range=bin_range)
    hist.set_data_quantity(name="histogram")
    hist.set_var_quantity(0, name="projected emission")
    return hist


def analyze_emission_histogram(
    hist, strat=None, strat_size=20, strat_prominence=0.08,
    sfit_idx_center_ratio=0.1, sfit_filter=1.,
    bgth_signal_err_num=0.1, n12_err_num=None
):
    """
    Gets state discrimination thresholds from projected images.

    Uses the histogram to find two or three peaks.
    There are two algorithms to determine the thresholds:

    * For the `signal fit` strategy, the background and signal peaks are
      fitted. The threshold is then set to have equal false negativer/positive
      rates.
    * For the `background threshold` strategy, only the background peak is
      fitted. The threshold is set to match a given false positive rate.
    * If no strategy is selected, a low-resolution histogram is used to
      estimate whether a signal peak can potentially be fitted.


    Parameters
    ----------
    hist : `ArrayData(1, float)`
        Projected image histogram.
    strat : `str` or `None`
        Threshold determination strategy. Options:
        `"sfit"` for signal fit. "bgth" for background threshold.
    strat_size : `int`
        Number of histogram bins, used for strategy selection.
    strat_prominence : `float`
        Relative peak prominence required to select `"sfit"`.
    sfit_idx_center_ratio : `float`
        The final background fit uses a cropped histogram.
        This parameter determines the starting point of the cropping,
        where `0` corresponds to the right base and `1` to the center.
    sfit_filter : `float`
        If the raw signal could not be fitted, a second fit is applied on
        Gaussian filtered data. This parameter selects the filtering width.
    bgth_signal_err_num : `float`
        Absolute false positive rate for background thresholding.
        In the typical use case, there are only few signal counts, so
        selecting a small value is advisable.
        If this strategy is used for a larger signal count, consider
        using a larger rate.
    n12_err_num : `float` or `None`
        Absolute false positive rate for signal thresholding (between
        states `1` and `2`). Works analogously to `bgth_signal_err_num`.
        If `None`, uses the error on the other boundary (`n01_err_num`).

    Returns
    -------
    ret : `dict(str->Any)`
        Returns a dictionary containing the following items:
    strat : `str`
        Which peak analysis strategy was chosen (see: `strat`).
    center : `[float, float]`
        Peak centers for states `0` and `1`.
    threshold : `[float, float]`
        State thresholds between states `0/1` and `1/2`.
    error_prob : `[float, float]`
        False positive/negative rates for states `0` and `1`
    error_num : `[float, float]`
        False positive/negative counts for states `0` and `1`.
        In contrast to the normalized `error_prob` parameter,
        this number takes `emission_num` into account.
    emission_num : `[float, float, float]`
        Number of sites mapped to the states `0`, `1` and `2`.
    peak_info : `[PeakInfo, PeakInfo]`
        Detailed peak information for states `0` and `1`.
    """
    if len(hist) < 16:
        raise RuntimeError(
            "`analyze_emission_histogram`: Invalid histogram resolution"
        )
    # Analyze background peak
    # 1. Find single most prominent peak
    # 2. Fit (multiple) skew Gaussians to it
    bg_peaks = find_peaks_1d_prominence(hist, npeaks=1, rel_prominence=0)
    try:
        bg_ad = bg_peaks["data"][0]
    except IndexError:
        raise RuntimeError("Could not find background peak")
    bg_pi = analyze_single_peak(bg_ad, max_subpeaks=0)
    if bg_pi is None:
        LOGGER.info(
            "`analyze_emission_histogram`: "
            "Increase function evaluations to fit background peak"
        )
        bg_pi = analyze_single_peak(bg_ad, max_subpeaks=0, maxfev=100000)
        if bg_pi is None:
            raise RuntimeError("Could not fit background peak")
    # Determine analysis strategy: fit signal peak vs threshold background
    # 1. Filter histogram to get a smooth function
    # 2. Check whether a second peak is visible with sufficient prominence
    # 3. Check that the most prominent peak is the background edge peak
    # 4. If both are true, a fit of the signal peak can be used
    if strat is None:
        strat_hist = hist.copy()
        strat_hist.data = scipy.ndimage.uniform_filter(
            strat_hist.data, size=len(strat_hist)//strat_size, mode="nearest"
        )
        strat_idx = hist.cv_quantity_to_index(bg_pi.base[1], 0)
        strat_hist = strat_hist[strat_idx:]
        strat_peaks = find_peaks_1d_prominence(
            strat_hist, npeaks=None, rel_prominence=strat_prominence,
            base_prominence_ratio=0, edge_peaks=True
        )
        if (
            len(strat_peaks["data"]) <= 1 or
            strat_peaks["data"][0].cv_quantity_to_index(
                strat_peaks["position"][0], 0
            ) <= 2
        ):
            strat = "bgth"
        else:
            strat = "sfit"
    LOGGER.debug(f"`analyze_emission_histogram`: using strategy {strat}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # Strategy: signal fit
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    if strat == "sfit":
        # Crop histogram to start from right base of background peak
        sfit_idx = hist.cv_quantity_to_index(
            sfit_idx_center_ratio * bg_pi.center
            + (1 - sfit_idx_center_ratio) * bg_pi.base[1], 0
        )
        sfit_hist = hist[max(sfit_idx - 2, 0):]
        # If extremely narrow peak, broaden for more consistent performance
        bg_idx = [bg_pi.data.cv_quantity_to_index(_b, 0) for _b in bg_pi.base]
        if bg_idx[1] - bg_idx[0] < 7:
            sfit_hist.data = scipy.ndimage.gaussian_filter(
                sfit_hist.data, sfit_filter * 0.7
            )
        sfit_peaks = find_peaks_1d_prominence(
            sfit_hist, npeaks=2, rel_prominence=0,
            base_prominence_ratio=0, edge_peaks=True
        )
        # Order by peak position
        n0_ad, n1_ad = [
            sfit_peaks["data"][i]
            for i in np.argsort(sfit_peaks["position"])
        ]
        n1_ad = sfit_hist[len(n0_ad):]
        _pi = [p for p in bg_pi.iter_peaks()][-1]
        n0_pi, n1_pi = None, None
        # Background: Try skew Gaussian fit
        if len(n0_ad) >= 5:
            n0_pi = analyze_single_peak(
                n0_ad, x0=_pi.center, p0=_pi.fit.get_popt()
            )
        # Background: Try non-skewed Gaussian fit
        if n0_pi is None:
            n0_pi = analyze_single_peak(
                n0_ad, x0=_pi.center, alpha=0, p0=_pi.fit.get_popt()
            )
        # Signal: Try skew Gaussian fit
        if len(n1_ad) >= 5:
            n1_pi = analyze_single_peak(n1_ad)
            # Signal: Try non-skewed Gaussian fit if weird width
            if (
                n1_pi is None or
                n1_pi.width > (n1_pi.data[-1] - n1_pi.data[0]) * 2
            ):
                n1_pi = analyze_single_peak(n1_ad, alpha=0)

        # Handling if fits did not succeed
        if n0_pi is None or n1_pi is None:
            LOGGER.info(
                "`analyze_emission_histogram`: Initial signal fit failed"
            )
            # Provide more background peak data for fit
            if len(n0_ad) <= 4:
                sfit_idx = max(sfit_idx - (5 - len(n0_ad)), 0)
            # Filter data for more reliable fitting
            sfit_hist = hist[sfit_idx:]
            sfit_hist.data = scipy.ndimage.gaussian_filter(
                sfit_hist.data, sfit_filter
            )
            # Perform same analysis again
            sfit_peaks = find_peaks_1d_prominence(
                sfit_hist, npeaks=2, rel_prominence=0,
                base_prominence_ratio=0, edge_peaks=True
            )
            n0_ad, n1_ad = [
                sfit_peaks["data"][i]
                for i in np.argsort(sfit_peaks["position"])
            ]
            n1_ad = sfit_hist[len(n0_ad):]
            # Check if each peak data has sufficient data points
            if len(n0_ad) < 5:
                _idx = sfit_hist.cv_quantity_to_index(n0_ad[-1], 0)
                n0_ad = sfit_hist[0:max(_idx, 5)]
            if len(n1_ad) < 5:
                _idxs = [
                    sfit_hist.cv_quantity_to_index(n1_ad[0], 0),
                    sfit_hist.cv_quantity_to_index(n1_ad[-1], 0)
                ]
                _idxs[0] -= (1 + 5 - len(n1_ad)) // 2
                _idxs[1] = max(_idxs[1], _idxs[0] + 5)
                n1_ad = sfit_hist[_idxs[0]:_idxs[1]]
            # Gaussian fits
            n0_pi = analyze_single_peak(
                n0_ad, x0=_pi.center, alpha=0, p0=_pi.fit.get_popt()
            )
            n1_pi = analyze_single_peak(n1_ad, alpha=0)
            if n0_pi is None or n1_pi is None:
                LOGGER.warning(
                    "`analyze_emission_histogram`: Signal fit failed. "
                    "Trying background thresholding."
                )
                return analyze_emission_histogram(
                    hist, strat="bgth",
                    bgth_signal_err_num=bgth_signal_err_num
                )
        # Perform state separation
        n01_thr, n0_err, n1_err = n0_pi.separation_loc(n0_pi, n1_pi)
        n0_center, n1_center = n0_pi.center, n1_pi.center

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # Strategy: background threshold
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    elif strat == "bgth":
        # Minimize false positive rate
        n0_pi = bg_pi
        n0_center = n0_pi.center
        n01_thr = n0_pi.distribution.isf(
            bgth_signal_err_num / n0_pi.distribution_amplitude
        )
        # Check for zero atoms
        _hist = hist[hist.cv_quantity_to_index(n01_thr, 0):]
        _pmf = np.array(_hist, dtype=float)
        _sum = np.sum(_pmf)
        if np.isclose(_sum, 0) or len(_hist) < 2:
            LOGGER.debug("`analyze_emission_histogram`: no atoms found.")
            n1_center = 2 * n01_thr - n0_center
            n12_thr = 3 * n01_thr - 2 * n0_center
            _mean = n1_center
            _std = hist.get_step(0)
            _amp = 0.1
        else:
            # Find n1_center by median
            _pmf = _pmf / _sum
            _cmf = np.array([np.sum(_pmf[:i + 1]) for i in range(len(_pmf))])
            _idx_median = np.argmin(abs(_cmf - 0.51))  # >0.5 to not round to 0
            n1_center = _hist.get_points(0)[_idx_median]
            # Estimate signal distribution from statistics
            n12_thr = 2 * n1_center - n01_thr
            _idxs = [
                hist.cv_quantity_to_index(n01_thr, 0),
                hist.cv_quantity_to_index(n12_thr, 0) + 1
            ]
            _hist = hist[_idxs[0]:_idxs[1]]
            _pmf = np.array(_hist, dtype=float)
            _pmf /= np.sum(_pmf)
            _mean = np.sum(_pmf * _hist.get_points(0))
            _std = np.sqrt(np.sum(_pmf * (_hist.get_points(0) - _mean)**2))
            if np.isclose(_std, 0):
                _std = _hist.get_step(0)
            _amp = np.sum(_hist)
        # Create signal peak information object
        model = FitGaussian1d()
        model.find_p0(_hist)
        model.set_p0(a=_amp/np.sqrt(2*np.pi), x0=_mean, wx=_std, c=0)
        n1_pi = PeakInfo(
            data=_hist, fit=model, center=_mean, width=_std,
            base=(n01_thr, n12_thr), subpeak=None
        )
        # Calculate errors
        n0_err = n0_pi.distribution.sf(n01_thr)
        n1_err = n1_pi.distribution.cdf(n01_thr)
        n1_center = n1_pi.center

    else:
        raise RuntimeError(f"Invalid `strat` {str(strat)}")

    if n12_err_num is not None:
        n12_err = n12_err_num / n1_pi.distribution_amplitude
    else:
        n12_err = n1_err
    # Cutoff due to machine precision
    if n12_err < 1e-50:
        n12_thr = 2 * n1_center - n01_thr
    else:
        n12_thr = n1_pi.distribution.isf(n12_err)

    # Package results
    n0_num = np.sum(hist.data[hist.get_points(0) < n01_thr])
    n1_num = np.sum(hist.data[
        (hist.get_points(0) >= n01_thr)
        & (hist.get_points(0) < n12_thr)
    ])
    n2_num = np.sum(hist.data[hist.get_points(0) >= n12_thr])
    return {
        "strat": strat,
        "center": [n0_center, n1_center],
        "threshold": [n01_thr, n12_thr],
        "error_prob": [n0_err, n1_err],
        "error_num": [n0_err * n0_num, n1_err * n1_num],
        "emission_num": [n0_num, n1_num, n2_num],
        "peak_info": [n0_pi, n1_pi]
    }


def get_state_estimate(emissions, thresholds):
    """
    Applies state thresholds to a projected image.

    Parameters
    ----------
    emissions : `Array[float]`
        Projected image.
    thresholds : `Iter[float]`
        State thresholds.
        States will be labelled according to the `thresholds` index.

    Returns
    -------
    state : `Array[int]`
        Projected image mapped to states.
    """
    if np.isscalar(thresholds):
        thresholds = [thresholds]
    thresholds = np.sort(thresholds)
    state = ArrayData(emissions).copy()
    state.set_data_quantity(name="state")
    state.data = np.zeros_like(state.data, dtype=int)
    emissions = np.array(emissions)
    for thr in thresholds:
        state.data += (emissions >= thr)
    return state


class EmissionHistogramAnalysis:

    """
    Container class for emission histogram analysis functions.

    See the respective functions for parameter details:

    * :py:func:`get_emission_histogram`
    * :py:func:`analyze_emission_histogram`
    """

    def __init__(
        self, bin_range=None, strat_size=20, strat_prominence=0.08,
        sfit_idx_center_ratio=0.1, sfit_filter=1., n12_err_num=None,
        bgth_signal_err_num=0.001
    ):
        # Histogram parameters
        self.bin_range = bin_range
        # Analysis strategy
        self.strat_size = strat_size
        self.strat_prominence = strat_prominence
        # Signal fit
        self.sfit_idx_center_ratio = sfit_idx_center_ratio
        self.sfit_filter = sfit_filter
        self.n12_err_num = n12_err_num
        # Background threshold
        self.bgth_signal_err_num = bgth_signal_err_num

    def get_attr_str(self):
        keys = [
            "bin_range", "strat_size", "strat_prominence",
            "sfit_idx_center_ratio", "sfit_filter", "n12_err_num",
            "bgth_signal_err_num"
        ]
        s = []
        for k in keys:
            v = getattr(self, k)
            if v is not None:
                s.append(f" → {k}: {str(v)}")
        return "\n".join(s)

    def __str__(self):
        return f"{self.__class__.__name__}:\n{self.get_attr_str()}"

    def __repr__(self):
        s = f"<'{self.__class__.__name__}' at {hex(id(self))}>"
        return f"{s}\n{self.get_attr_str()}"

    def get_emission_histogram(self, emissions):
        return get_emission_histogram(emissions, self.bin_range)

    def analyze_emission_histogram(self, hist):
        return analyze_emission_histogram(
            hist,
            strat_size=self.strat_size, strat_prominence=self.strat_prominence,
            sfit_idx_center_ratio=self.sfit_idx_center_ratio,
            sfit_filter=self.sfit_filter,
            n12_err_num=self.n12_err_num,
            bgth_signal_err_num=self.bgth_signal_err_num
        )

    def get_state_estimate(self, emissions, thresholds):
        return get_state_estimate(emissions, thresholds)


###############################################################################
# State estimator
###############################################################################


class ReconstructionResult(io.FileBase):

    """
    State reconstruction results container.

    Attributes
    ----------
    state_estimator_id : `str`
        Identifier of :py:class:`StateEstimator` object that
        generated this reconstruction result.
    outlier_ratios : `float`
        Image preprocessing outlier ratios (see `:py:class:ImagePreprocessor`).
    trafo : `AffineTrafo2d`
        Affine transformation between sites and image coordinates.
    trafo_phase, trafo_phase_ref_image, trafo_phase_ref_site : `(float, float)`
        Phase and phase references of transformation.
    emissions : `ArrayData(2, float)`
        Image projected onto sites.
    state : `ArrayData(2, int)`
        Estimated state of each site.
    label_center : `np.ndarray(2, float)`
        Positions of isolated atoms in image coordinates.
        Dimensions: `[n_atoms, ndim]`.
    histogram : `ArrayData(1, float)`
        Histogram of projected values.
    hist_strat : `str`
        Histogram peak analysis strategy. Options: `"sfit", "bgth"`.
        See :py:func:`analyze_emission_histogram` for details.
    hist_center : `[float, float, float]`
        Projected center value for the states.
    hist_threshold : `[float, float]`
        State discrimination threshold of projected values.
    hist_error_prob : `[float, float]`
        False negative/positive probability for first threshold
    hist_error_num : `[float, float]`
        False negative/positive number for first threshold.
    hist_emission_num : `[int, int, int]`
        Number of sites associated to each state.
    hist_peak_info : `[PeakInfo, PeakInfo]`
        Detailed histogram peak information.

    Notes
    -----
    This object can be saved with all information using :py:meth:`save`
    in the formats `"json", "bson"`.
    The (integer) state array itself can be saved using :py:meth:`save_state`
    in the formats `"csv", "txt", "json", "png"`.
    """

    _attributes = {
        "state_estimator_id", "outlier_ratios", "trafo",
        "trafo_phase", "trafo_phase_ref_image", "trafo_phase_ref_site",
        "emissions", "state",
        "histogram", "hist_strat", "hist_center", "hist_threshold",
        "hist_error_prob", "hist_error_num", "hist_emission_num",
        "hist_peak_info", "success"
    }

    LOGGER = get_logger("srec.ReconstructionResult")

    def __init__(self, **kwargs):
        for k in self._attributes:
            setattr(self, k, None)
        for k, v in kwargs.items():
            if k not in self._attributes:
                self.LOGGER.warn(f"Invalid attribute: {str(k)}")
            setattr(self, k, v)

    def attributes(self):
        return {k: getattr(self, k) for k in self._attributes}

    def copy(self):
        attrs = {}
        for k, v in self.attributes().items():
            try:
                attrs[k] = v.copy()
            except AttributeError:
                attrs[k] = copy.deepcopy(v)
        return ReconstructionResult(**attrs)

    def get_attr_str(self):
        attrs = self.attributes().copy()
        if self.success:
            keys = [
                "outlier_ratios", "trafo_phase",
                "hist_strat", "hist_center", "hist_threshold",
                "hist_error_prob", "hist_error_num", "hist_emission_num"
            ]
        else:
            keys = ["outlier_ratios", "trafo_phase", "success"]
        s = [f" → {k}: {str(attrs[k])}" for k in keys]
        return "\n".join(s)

    def __str__(self):
        return f"{self.__class__.__name__}:\n{self.get_attr_str()}"

    def __repr__(self):
        s = f"<'{self.__class__.__name__}' at {hex(id(self))}>"
        return f"{s}\n{self.get_attr_str()}"

    def save_state(
        self, file_path, fmt=None, flip_orientation=False, **kwargs
    ):
        """
        Saves the reconstructed state array.

        Parameters
        ----------
        file_path : `str`
            File path to save to.
        fmt : `str` or `None`
            File format. Options: `"csv", "txt", "json", "png"`.
            If `None`, is deduced from the `file_path` extension.
            If deduction is unsuccessful, uses `"csv"` and appends the
            extension to the `file_path`.
        flip_orientation : `bool`
            The array orientation in this library uses the convention to
            have the coordinate axes `+x -> right`, `+y -> up`.
            The default orientation in the saved files have the coordinate
            axes `+x -> down`, `+y -> right`, thus saved files opened in
            third-party programs might appear flipped in orientation.
            Setting `flip_orientation` to `True` accounts for this flip.
        **kwargs
            Keyword arguments to the functions writing to file. These are:
            For `"csv", "txt"`: `np.savetxt`.
            For `"json"`: `json.dump`.
            For `"png"`: `PIL.Image.save`.

        Returns
        -------
        file_path : `str`
            Saved file path.
        """
        if fmt is None:
            try:
                fmt = io.get_file_format(file_path, fmt=fmt)
            except KeyError:
                fmt = "csv"
                file_path = file_path + "." + fmt
        ar = np.array(self.state, dtype=np.uint8)
        if flip_orientation:
            ar = np.transpose(ar)
            ar = np.flip(ar, axis=1)
        if "csv" in fmt or "txt" in fmt:
            if "delimiter" not in kwargs:
                kwargs["delimiter"] = ","
            np.savetxt(file_path, ar, **kwargs)
        elif "json" in fmt:
            with open(file_path, "w") as _f:
                json.dump(list(ar), _f, **kwargs)
        elif "png" in fmt:
            PIL.Image.fromarray(ar).save(file_path, format=fmt, **kwargs)
        else:
            raise RuntimeError(f"Invalid file format ({str(fmt)})")
        return file_path


class StateEstimator:

    """
    Class for state reconstruction from a fluorescence image.

    Parameters
    ----------
    id : `str`
        Object identifier.
    image_preprocessor : `ImagePreprocessor`
        Image preprocessor object.
    phase_ref_site, phase_ref_image : `(float, float)`
        Transformation phase reference points in sites and image space.
    trafo_site_to_image : `AffineTrafo2d`
        Transformation between sites and image coordinates.
        Its phase is optimized for each image individually.
    projector_generator : `ProjectorGenerator`
        Projector generator object.
    isolated_locator : `IsolatedLocator`
        Isolated atoms locator object,
        used to obtain the transformation phase.
    sites_shape : `(int, int)`
        Shape of 2D array representing lattice sites.
    emission_histogram_analysis : `EmissionHistogramAnalysis`
        Emission histogram analysis object.

    Examples
    --------
    Standard use case given a projector generator, isolated atoms locator
    and an image to be reconstructed:

    >>> type(prjgen)
    state_reconstruction.gen.proj_gen.ProjectorGenerator
    >>> type(isoloc)
    state_reconstruction.est.iso_est.IsolatedLocator
    >>> image.shape
    (512, 512)
    >>> stest = StateEstimator(
    ...     projector_generator=prjgen,
    ...     isolated_locator=isoloc,
    ...     sites_shape=(170, 170)
    ... )
    >>> stest.setup()
    >>> recres = stest.reconstruct(image)
    >>> type(recres)
    state_reconstruction.est.state_est.ReconstructionResult
    >>> recres.state.shape
    (170, 170)
    """

    LOGGER = get_logger("srec.StateEstimator")

    def __init__(
        self, id=None,
        image_preprocessor=None,
        phase_ref_image=None, phase_ref_site=None,
        trafo_site_to_image=None,
        projector_generator=None,
        isolated_locator=None,
        sites_shape=(170, 170),
        emission_histogram_analysis=None
    ):
        # Parse parameters
        if id is None:
            id = str(uuid()).split("-")[-1]
        if phase_ref_image is None:
            phase_ref_image = trafo_gen.TrafoManager.get_phase_ref_image()
        if phase_ref_site is None:
            phase_ref_site = trafo_gen.TrafoManager.get_phase_ref_site()
        # Assign attributes
        self.id = id
        self.image_preprocessor = image_preprocessor
        self.phase_ref_site = phase_ref_site
        self.phase_ref_image = phase_ref_image
        self.trafo_site_to_image = trafo_site_to_image
        self.projector_generator = projector_generator
        self.isolated_locator = isolated_locator
        self.sites_shape = sites_shape
        self.emission_histogram_analysis = emission_histogram_analysis
        self.sest_cpp = state_reconstruction_cpp.StateEstimator()
        self.sest_cpp.init()
        self.sest_cpp.loadProj(self.projector_generator)
        if self.image_preprocessor.scale:
            self.sest_cpp.setImagePreProcScale(self.image_preprocessor.scale, 
                self.image_preprocessor.outlier_size, self.image_preprocessor.max_outlier_ratio, 
                self.image_preprocessor.outlier_min_ref_val, self.image_preprocessor.outlier_iterations)
        else:
            self.sest_cpp.setImagePreProc(
                self.image_preprocessor.outlier_size, self.image_preprocessor.max_outlier_ratio, 
                self.image_preprocessor.outlier_min_ref_val, self.image_preprocessor.outlier_iterations)

    @classmethod
    def discover_configs(cls, config_dir=None):
        """Gets a list of configuration file paths."""
        if config_dir is None:
            config_dir = get_config("state_estimator_config_dir")
        return [
            os.path.join(config_dir, fn) for fn in path.get_folder_contents(
                config_dir, regex=r".json$"
            ).files_matched
        ]

    @classmethod
    def from_config(
        cls, *args, config=None, **kwargs
    ):
        # Parse parameters
        if len(args) == 1:
            if isinstance(args[0], str):
                return cls.from_config(config=args[0])
            elif isinstance(args[0], dict):
                return cls.from_config(**args[0])
            else:
                raise ValueError("Invalid parameters")
        elif len(args) > 1:
            raise ValueError("Invalid parameters")
        # From config file
        if config is not None:
            config = misc.assume_endswith(config, ".json")
            if not os.path.exists(config):
                config = os.path.join(
                    get_config("state_estimator_config_dir"), config
                )
            config = dict(io.load(config))
            config.update(kwargs)
            return cls.from_config(**config)
        # Construct object
        if "image_preprocessor" in kwargs:
            kwargs["image_preprocessor"] = misc.assume_construct_obj(
                kwargs["image_preprocessor"], ImagePreprocessor
            )
        if "trafo_site_to_image" in kwargs:
            _trafo = kwargs["trafo_site_to_image"]
            if isinstance(_trafo, str):
                _trafo = io.load(_trafo)
            kwargs["trafo_site_to_image"] = misc.assume_construct_obj(
                _trafo, AffineTrafo2d
            )
        else:
            raise ValueError("`trafo_site_to_image` must be specified")
        if "projector_generator" in kwargs:
            _prjgen = kwargs["projector_generator"]
            if not isinstance(_prjgen, proj_gen.ProjectorGenerator):
                if "trafo_site_to_image" not in _prjgen:
                    _prjgen["trafo_site_to_image"] = (
                        kwargs["trafo_site_to_image"]
                    )
            kwargs["projector_generator"] = misc.assume_construct_obj(
                _prjgen, proj_gen.ProjectorGenerator
            )
        if "isolated_locator" in kwargs:
            kwargs["isolated_locator"] = misc.assume_construct_obj(
                kwargs["isolated_locator"], IsolatedLocator
            )
        if "emission_histogram_analysis" in kwargs:
            kwargs["emission_histogram_analysis"] = misc.assume_construct_obj(
                kwargs["emission_histogram_analysis"],
                EmissionHistogramAnalysis
            )
        return cls(**kwargs)

    def get_attr_str(self):
        keys = [
            "id", "trafo_site_to_image", "phase_ref_site",
            "phase_ref_image", "sites_shape"
        ]
        s = [f" → {k}: {str(getattr(self, k))}" for k in keys]
        if self.image_preprocessor:
            s.append(str(self.image_preprocessor))
        s.append(str(self.projector_generator.integrated_psf_generator))
        s.append(str(self.projector_generator))
        if self.isolated_locator:
            s.append(str(self.isolated_locator))
        if self.emission_histogram_analysis:
            s.append(str(self.emission_histogram_analysis))
        return "\n".join(s)

    def __str__(self):
        return f"{self.__class__.__name__}:\n{self.get_attr_str()}"

    def __repr__(self):
        s = f"<'{self.__class__.__name__}' at {hex(id(self))}>"
        return f"{s}\n{self.get_attr_str()}"

    def setup(self, print_progress=True):
        """
        Checks that all attributes are set and initializes them.
        """
        if self.trafo_site_to_image is None:
            if self.projector_generator is not None:
                self.trafo_site_to_image = (
                    self.projector_generator.trafo_site_to_image
                )
            else:
                raise RuntimeError("`trafo_site_to_image` is not set")
        if self.projector_generator is None:
            raise RuntimeError("`projector_generator` is not set")
        if not self.projector_generator.proj_cache_built:
            self.LOGGER.info("Building `projector_generator` cache")
            self.projector_generator.setup_cache(print_progress=print_progress)
        if self.emission_histogram_analysis is None:
            self.emission_histogram_analysis = EmissionHistogramAnalysis()
        return True

    @property
    def trafo_site_to_image(self):
        return self._trafo_site_to_image

    @trafo_site_to_image.setter
    def trafo_site_to_image(self, val):
        self._trafo_site_to_image = trafo_gen.get_trafo_site_to_image(
            trafo_site_to_image=val, phase=np.zeros(2),
            phase_ref_site=self.phase_ref_site,
            phase_ref_image=self.phase_ref_image
        )

    @property
    def proj_shape(self):
        return self.projector_generator.proj_shape

    @property
    def psf_shape(self):
        return self.projector_generator.psf_shape

    @property
    def psf_supersample(self):
        return self.projector_generator.psf_supersample

    def get_phase_shifted_trafo(
        self, phase=None, image=None, method="projection_optimization",
        preprocess_image=True
    ):
        """
        Gets a phase-shifted lattice transformation from an image.

        Parameters
        ----------
        phase : `(float, float)`
            Directly specifies phase. Takes precedence over `image`.
        image : `Array[2, float]`
            Image from which to extract lattice transformation.
        method : `str`
            Method for determining lattice transformation:
            `"projection_optimization", "isolated_atoms"`.
        preprocess_image : `bool`
            Whether to preprocess image.

        Returns
        -------
        new_trafo : `AffineTrafo2d`
            Phase-shifted lattice transformation.
        """
        if phase is None:
            # Image preprocessing
            if image is None:
                raise ValueError("No `image` or `phase` given")
            image = np.array(image)
            if np.isfortran(image):
                image = np.ascontiguousarray(image)
            if preprocess_image and self.image_preprocessor:
                image, _ = self.image_preprocessor.process_image(image)
            # From isolated atoms
            if (
                method == "isolated_atoms"
                and self.isolated_locator is not None
            ):
                label_centers = self.isolated_locator.get_label_centers(image)
                if len(label_centers) > 0:
                    phase, _ = get_trafo_phase_from_points(
                        *np.moveaxis(label_centers, -1, 0),
                        self.trafo_site_to_image
                    )
                else:
                    phase = np.zeros(2)
                    self.LOGGER.error(
                        f"get_phase_shifted_trafo: "
                        f"Could not locate isolated atoms. "
                        f"Using default phase: {str(phase)}"
                    )
            # From emission variance maximization
            else:
                if method != "projection_optimization":
                    self.LOGGER.warning(
                        f"get_phase_shifted_trafo: "
                        f"Method `{str(method)}` unavailable, "
                        f"using `projection_optimization`"
                    )
                srec_cpp = state_reconstruction_cpp.TrafoEstimator()
                phase = srec_cpp.get_trafo_phase_from_projections(image, self.projector_generator, self.phase_ref_image, self.phase_ref_site)

        # Construct phase-shifted trafo
        new_trafo = trafo_gen.get_trafo_site_to_image(
            trafo_site_to_image=self.trafo_site_to_image, phase=phase,
            phase_ref_site=self.phase_ref_site,
            phase_ref_image=self.phase_ref_image
        )
        return new_trafo

    def reconstruct(self, image, new_trafo=None):
        """
        Reconstructs the state of each lattice site from an image.

        Parameters
        ----------
        image : `Array[2, float]`
            Fluorescence image to be reconstructed.
        new_trafo : `None` or `AffineTrafo2d`
            If `AffineTrafo2d`, uses `new_trafo` to project the state.
            If `None`, a `new_trafo` is created by optimizing the
            transformation phase.

        Returns
        -------
        res : `ReconstructionResult`
            Reconstruction result.
        """
        # Preprocess image
        start_time = datetime.now()
        image = np.array(image, dtype=float)
        if np.isfortran(image):
            image = np.ascontiguousarray(image)
        emissions = ArrayData(np.zeros(self.sites_shape, dtype=float))
        
        local_emissions, trafo_phase, outlier_ratios, trafo_matrix, trafo_offset = self.sest_cpp.reconstruct(image, None, self.sites_shape, self.proj_shape, 
            self.psf_supersample, self.projector_generator, self.phase_ref_image, self.phase_ref_site, emissions.data, self.trafo_site_to_image)
        if new_trafo is None:
            new_trafo = AffineTrafo2d(np.array(trafo_matrix), np.array(trafo_offset))
        t = datetime.now() - start_time
        print("After applying projectors: " + t.seconds.__str__() + "s " + t.microseconds.__str__() + "us")
        # Perform histogram analysis for state discrimination
        eha = self.emission_histogram_analysis
        histogram = eha.get_emission_histogram(local_emissions)
        try:
            histogram_data = eha.analyze_emission_histogram(histogram)
            state = eha.get_state_estimate(
                emissions, histogram_data["threshold"]
            )
            state_estimation_success = True
        except (RuntimeError, IndexError, ValueError) as e:
            self.LOGGER.error(
                f"Emission histogram analysis failed: {str(e)}"
            )
            histogram_data = None
            state_estimation_success = False
        t = datetime.now() - start_time
        print("After performing histogram analysis: " + t.seconds.__str__() + "s " + t.microseconds.__str__() + "us")
        # Package result
        res = dict(
            state_estimator_id=self.id,
            outlier_ratios=outlier_ratios,
            trafo=new_trafo,
            trafo_phase=trafo_phase,
            trafo_phase_ref_image=self.phase_ref_image,
            trafo_phase_ref_site=self.phase_ref_site,
            emissions=emissions,
            histogram=histogram,
            success=state_estimation_success
        )
        if state_estimation_success:
            res.update(dict(
                hist_strat=histogram_data["strat"],
                hist_center=histogram_data["center"],
                hist_threshold=histogram_data["threshold"],
                hist_error_prob=histogram_data["error_prob"],
                hist_error_num=histogram_data["error_num"],
                hist_emission_num=histogram_data["emission_num"],
                hist_peak_info=histogram_data["peak_info"],
                state=state
            ))
        else:
            res.update(dict(
                state=np.zeros(self.sites_shape)
            ))
        res = ReconstructionResult(**res)
        t = datetime.now() - start_time
        print("After packaging results: " + t.seconds.__str__() + "s " + t.microseconds.__str__() + "us")
        return res

    def get_reconstructed_image(self, res, image_shape=(512, 512)):
        """
        Gets a reconstructed fluorescence image.

        Parameters
        ----------
        res : `ReconstructionResult`
            Successful reconstruction result object.
        image_shape : `(int, int)`
            Shape of image.

        Returns
        -------
        image_clean : `ArrayData[2, float]`
            Reconstructed fluorescence image.
        """
        # Parse parameters
        if res.success is not True:
            self.LOGGER.warning(
                "get_reconstructed_image: ReconstructionResult invalid"
            )
        if np.isscalar(image_shape):
            image_shape = (image_shape, image_shape)
        # Get site positions inside image
        _half_proj_shape = np.array(self.proj_shape) // 2
        site_coords = res.emissions.get_var_meshgrid()
        image_coords = res.trafo.coord_to_target(site_coords.T).T
        mask = np.logical_and.reduce([
            (image_coords[i] > _half_proj_shape[i])
            & (image_coords[i] < image_shape[i] - _half_proj_shape[i])
            for i in range(2)
        ])
        image_coords = np.array([_c[mask] for _c in image_coords])
        emissions_masked = np.array(res.emissions)[mask]
        # Generate image
        ipsfgen = self.projector_generator.integrated_psf_generator
        local_psfs = image_gen.get_local_psfs(
            *image_coords, integrated_psf_generator=ipsfgen
        )
        image_clean = image_gen.get_image_clean(
            local_psfs=local_psfs, brightness=emissions_masked,
            size=image_shape
        )
        return ArrayData(image_clean)

    def get_reconstructed_emissions(self, res, image_shape=(512, 512)):
        """
        Gets the reconstructed image coordinates and emissions.

        Parameters
        ----------
        res : `ReconstructionResult`
            Successful reconstruction result object.
        image_shape : `(int, int)`
            Shape of image.

        Returns
        -------
        _d : `dict(str->Any)`
            Result dictionary containing the following items:
        image_coords_occupied, image_coords_empty : `np.ndarray(2, float)`
            Image coordinates of occupied/empty sites.
        emissions_occupied, emissions_empty : `np.ndarray(1, float)`
            Corresponding emissions.
        """
        # Parse parameters
        if res.success is not True:
            self.LOGGER.warning(
                "get_reconstructed_emissions: ReconstructionResult invalid"
            )
        if np.isscalar(image_shape):
            image_shape = (image_shape, image_shape)
        # Get all coordinates inside image
        site_coords = res.emissions.get_var_meshgrid()
        image_coords = res.trafo.coord_to_target(np.transpose(site_coords)).T
        mask = np.logical_and.reduce([
            (image_coords[i] > 0) & (image_coords[i] < image_shape[i])
            for i in range(2)
        ])
        image_coords = np.array([_c[mask] for _c in image_coords])
        emissions_masked = np.array(res.emissions)[mask]
        # Separate empty and occupied sites
        if res.success:
            mask_occupied = emissions_masked > res.hist_threshold[0]
        else:
            mask_occupied = np.full_like(emissions_masked, True, dtype=bool)
        image_coords_occupied = np.array([
            _c[mask_occupied] for _c in image_coords
        ])
        image_coords_empty = np.array([
            _c[~mask_occupied] for _c in image_coords
        ])
        emissions_occupied = emissions_masked[mask_occupied]
        emissions_empty = emissions_masked[~mask_occupied]
        return {
            "image_coords_occupied": image_coords_occupied,
            "emissions_occupied": emissions_occupied,
            "image_coords_empty": image_coords_empty,
            "emissions_empty": emissions_empty
        }


###############################################################################
# Plotting reconstruction results
###############################################################################


def plot_reconstructed_emissions(
    res=None, image_coords_occupied=None, emissions_occupied=None,
    image_coords_empty=None, plot_range=True, colorbar=True, clabel=None,
    cmap="plasma", size_occupied=12, color_empty="white", size_empty=1,
    ax=None, color_ax=np.full(3, 0.75), **_
):
    """
    Plots reconstructed emissions as circles in their image coordinates.

    Parameters
    ----------
    res : `ReconstructionResult`
        Reconstruction result for determining the color map range.
    image_coords_occupied, image_coords_empty : `Array[2, float]`
        Image coordinates of occupied and empty sites with dimensions:
        `[ndim, nsites]`.
    emissions_occupied : `Array[1, float]`
        Projected emissions in the order corresponding to
        `image_coords_occupied`.
    plot_range : `bool``
        Whether `xlim, ylim` of the `ax` are automatically set.
    colorbar : `bool` or `mpl.axes.Axes`
        Whether a color bar is shown.
        If `mpl.axes.Axes`, uses `colorbar` as color bar axes.
    clabel : `str`
        Color bar label.
    cmap : `str`
        Color map for projected emissions.
    size_occupied, size_empty : `float`
        Marker size for occupied and empty sites.
    color_empty : `str`
        Color of empty sites.
    ax : `mpl.axes.Axes` or `None`
        Matplotlib axes to plot into.
    color_ax : `str`
        Background color of axes.
    """
    # Parse parameters
    if emissions_occupied is None:
        raise ValueError("No `emissions_occupied` available")
    if ax is None:
        ax = plot.gca()
    if res is not None and res.success is True:
        vmin = res.hist_threshold[0]
        vcenter = res.hist_center[1]
        vdif = vcenter - vmin
    else:
        vmin, vmax = np.min(emissions_occupied), np.max(emissions_occupied)
        vdif = (vmax - vmin) / 2
    if vdif == 0:
        vdif = 1
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(name=cmap)
    # Set plot range
    plot_rect = []
    for dim in range(2):
        _coords = np.concatenate([
            np.ravel(_x) for _x in [
                image_coords_empty[dim], image_coords_occupied[dim]
            ]
        ])
        plot_rect.append([
            np.floor(np.min(_coords)), np.ceil(np.max(_coords)) + 1
        ])
    plot_rect = np.array(plot_rect, dtype=int)
    if plot_range is True:
        ((xmin, xmax), (ymin, ymax)) = plot_rect
        plot.style_axes(ax=ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # Generate colorbar
    if colorbar:
        background = ArrayData(np.full((2, 2), np.nan))
        for dim in range(2):
            background.set_dim(dim, center=np.mean(plot_rect[dim]), step=1)
        plot.pcolorim(
            background, ax=ax, cmap=cmap,
            vmin=vmin, vmax=vmin+2*vdif, colorbar=colorbar, clabel=clabel,
            rasterized=True
        )
    # Plot empty sites
    if image_coords_empty is not None:
        ax.scatter(*image_coords_empty, s=size_empty, color=color_empty,
                   rasterized=True)
    # Plot occupied sites
    colors_occupied = cmap((emissions_occupied - vmin) / 2 / vdif)
    ax.scatter(*image_coords_occupied, c=colors_occupied, s=size_occupied)
    ax.set_facecolor(color_ax)


def plot_reconstructed_histogram(
    res, rel_margin=0.05,
    size_histogram=5, color_histogram="grey", alpha_histogram=0.3,
    colors_fit=["C1", "C2"], linewidth_fit=2, color_thr="black",
    min_ymax=15,
    ax=None
):
    """
    Plots reconstructed emissions histogram.

    Parameters
    ----------
    res : `ReconstructionResult`
        Reconstruction result containing histogram.
    rel_margin : `float`
        Margin of the plot relative to n0 and n12 lines.
    size_histogram : `float`
        Marker size of histogram points.
    color_histogram : `str`
        Color of histogram plot.
    alpha_histogram : `float`
        Opacity of histogram fill.
    colors_fit : `(str, str)`
        Colors of (n0, n1) histogram fits.
    linewidth_fit : `float`
        Line width of histogram fits.
    color_thr : `str`
        Color of n01 and n12 threshold lines.
    min_ymax : `float`
        Minimum value of the y-axis maximum.
    ax : `mpl.axes.Axes` or `None`
        Matplotlib axes to plot into.
    """
    # Parse parameters
    if ax is None:
        ax = plot.gca()
    # Plot raw histogram
    plot.plot(res.histogram, ax=ax, zorder=-5, color=color_histogram)
    plot.scatter(
        res.histogram, ax=ax, zorder=0,
        markersize=size_histogram, color=color_histogram
    )
    ax.fill_between(
        res.histogram.get_points(0), res.histogram, zorder=-10,
        color=color_histogram, ec=color_histogram, alpha=alpha_histogram
    )
    if not res.success:
        return
    # Parse results
    dx = res.hist_threshold[1] - res.hist_center[0]
    xmin = res.hist_center[0] - rel_margin * dx
    ymax = max(min_ymax, np.max(res.histogram[
        res.histogram.cv_quantity_to_index(res.hist_threshold[0], 0):
    ]))
    ymin, ymax = -rel_margin * ymax, (1 + rel_margin) * ymax
    # Plot fits
    color_n1 = colors_fit[0]
    color_n2 = colors_fit[1]
    plot.plot(
        res.hist_peak_info[0].get_model_data(), ax=ax, zorder=-2,
        linewidth=linewidth_fit, color=color_n1
    )
    if res.hist_strat == "sfit":
        plot.plot(
            res.hist_peak_info[1].get_model_data(), ax=ax, zorder=-2,
            linewidth=linewidth_fit, color=color_n2
        )
    ax.axvline(res.hist_center[0], color=color_n1)
    ax.axvline(res.hist_center[1], color=color_n2)
    ax.axvline(res.hist_threshold[0], color=color_thr)
    ax.axvline(res.hist_threshold[1], color=color_thr)
    # Axes properties
    plot.style_axes(ax=ax, ymin=ymin, ymax=ymax, xmin=xmin)


def plot_reconstruction_results(
    state_estimator, rec_res, raw_image, image_id=None,
    fig=None, figsize=(14, 10),
    kwargs_plot_image={}, kwargs_plot_histogram={}, kwargs_plot_emissions={}
):
    """
    Performs a default plot of reconstruction results.

    Parameters
    ----------
    state_estimator : `StateEstimator`
        State estimator object used for reconstruction.
    rec_res : `ReconstructionResult`
        Reconstruction result to be plotted.
    raw_image : `Array[2, float]`
        Raw image.
    image_id : `str`
        Image identifier.
    fig : `mpl.figure.Figure`
        Matplotlib figure to be plotted into. Overwrites `figsize`.
    figsize : `(float, float)`
        Matplotlib figure size.
    kwargs_plot_... : `dict(str->Any)`
        Keyword arguments for various underlying plot functions.

    Returns
    -------
    fig : `mpl.figure.Figure`
        Generated matplotlib figure.
    """
    # Set plot parameters
    if fig is None:
        fig = plot.figure(figsize=figsize)
    axs = fig.subplots(ncols=2, nrows=2)
    plot_shape = np.array(raw_image.shape)
    plot_rect = np.array(misc.cv_index_center_to_rect(
        plot_shape//2, size=plot_shape-2*np.array(state_estimator.proj_shape)
    ))
    plot_lim = dict(
        xmin=plot_rect[0, 0], xmax=plot_rect[0, 1],
        ymin=plot_rect[1, 0], ymax=plot_rect[1, 1]
    )
    # Images
    rec_image = state_estimator.get_reconstructed_image(
        rec_res, image_shape=raw_image.shape
    )
    title_appendix = "" if image_id is None else f" ({str(image_id)})"
    if rec_res.outlier_ratios is None:
        raw_appendix = ""
    else:
        raw_appendix = (
            f", Outlier: "
            f"{misc.cv_iter_to_str(rec_res.outlier_ratios, fmt='{:.1f}'):s}"
        )
    plt_param = dict(
        cmap="hot", vmin=0, vmax=np.max(rec_image)*1.25,
        colorbar=True, clabel="Camera counts"
    )
    plt_param.update(kwargs_plot_image)
    plot.pcolorim(
        np.array(raw_image), ax=axs[0, 0],
        title="Raw image" + title_appendix + raw_appendix, **plt_param
    )
    plot.pcolorim(
        rec_image, ax=axs[1, 0],
        title="Reconstructed image" + title_appendix, **plt_param
    )
    [plot.style_axes(ax=ax, **plot_lim) for ax in axs[:, 0]]
    # Emissions
    ax = axs[0, 1]
    emissions = state_estimator.get_reconstructed_emissions(
        rec_res, image_shape=raw_image.shape
    )
    emissions.update(dict(clabel="Projected emissions"))
    emissions.update(kwargs_plot_emissions)
    plot_reconstructed_emissions(res=rec_res, ax=ax, **emissions)
    plot.style_axes(ax=ax, title=(
        f"Sites (Rec-ID: {state_estimator.id}), Phase: "
        f"[{', '.join([f'{_e:.2f}' for _e in rec_res.trafo_phase])}]"
    ), **plot_lim)
    # Histogram
    ax = axs[1, 1]
    plt_param = dict(min_ymax=50)
    plt_param.update(kwargs_plot_histogram)
    plot_reconstructed_histogram(rec_res, ax=ax, **plt_param)
    if rec_res.success:
        plot.style_axes(ax=ax, title=(
            f"States: {str(rec_res.hist_emission_num)}, Error (abs.): "
            f"[{', '.join([f'{_e:.2e}' for _e in rec_res.hist_error_num])}]"
        ))
    else:
        plot.style_axes(ax=ax, title="Reconstruction result invalid")
    # Return figure
    plot.style_figure(fig=fig, tight_layout=True)
    return fig
