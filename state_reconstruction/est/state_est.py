"""
State estimator.

Assigns emission states to projections.
"""

import json
import PIL
import numpy as np
import scipy.optimize
import scipy.ndimage

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData
from libics.core import io
from libics.tools.math.peaked import FitGaussian1d
from libics.tools.math.signal import (
    find_histogram, find_peaks_1d_prominence, analyze_single_peak, PeakInfo
)

from state_reconstruction.gen import trafo_gen
from .proj_est import get_local_images, apply_projectors
from .trafo_est import get_trafo_phase_from_points

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
    bgth_signal_err_num=0.1, n12_err_num=0.1
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
    n12_err_num : `float`
        Absolute false positive rate for signal thresholding (between
        states `1` and `2`). Works analogously to `bgth_signal_err_num`.

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
            strat = "sfit"
        else:
            strat = "bgth"
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
        # Estimate signal distribution from statistics
        n12_thr = 3 * n01_thr - 2 * n0_center
        _idxs = [
            hist.cv_quantity_to_index(n01_thr, 0),
            hist.cv_quantity_to_index(n12_thr, 0)
        ]
        _hist = hist[_idxs[0]:_idxs[1]]
        _pmf = np.array(_hist)
        _pmf /= np.sum(_pmf)
        _mean = np.sum(_pmf * _hist.get_points(0))
        _std = np.sqrt(np.sum(_pmf * _hist.get_points(0)**2) - _mean**2)
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
        n0_err = n0_pi.sf(n01_thr)
        n1_err = n1_pi.cdf(n01_thr)
        n1_center = n1_pi.center

    else:
        raise RuntimeError(f"Invalid `strat` {str(strat)}")

    n12_thr = n1_pi.distribution.isf(
        n12_err_num / n1_pi.distribution_amplitude
    )

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
        sfit_idx_center_ratio=0.1, sfit_filter=1., n12_err_num=0.1,
        bgth_signal_err_num=0.1
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
    trafo : `AffineTrafo2d`
        Affine transformation between sites and image coordinates.
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
        "trafo", "emissions", "state", "label_center",
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
    projector_generator : `ProjectorGenerator`
        Projector generator object.
    isolated_locator : `IsolatedLocator`
        Isolated atoms locator object,
        used to obtain the transformation phase.
    trafo_site_to_image : `AffineTrafo2d`
        Transformation between sites and image coordinates.
        Its phase is optimized for each image individually.
    phase_ref_site, phase_ref_image : `(float, float)`
        Transformation phase reference points in sites and image space.
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
        self,
        projector_generator=None,
        isolated_locator=None,
        trafo_site_to_image=None,
        phase_ref_site=(0, 0), phase_ref_image=(169, 84),
        sites_shape=(170, 170),
        emission_histogram_analysis=None
    ):
        self.projector_generator = projector_generator
        self.isolated_locator = isolated_locator
        self.phase_ref_site = phase_ref_site
        self.phase_ref_image = phase_ref_image
        self.trafo_site_to_image = trafo_site_to_image
        self.sites_shape = sites_shape
        self.emission_histogram_analysis = emission_histogram_analysis

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
        # Find trafo phase
        label_centers = np.array([])
        if new_trafo is None:
            label_centers = self.isolated_locator.get_label_centers(image)
            if len(label_centers) > 0:
                phase, _ = get_trafo_phase_from_points(
                    *np.moveaxis(label_centers, -1, 0),
                    self.trafo_site_to_image
                )
            else:
                phase = np.zeros(2)
                self.LOGGER.error(
                    f"Could not locate isolated atoms. "
                    f"Using default phase: {str(phase)}"
                )
            new_trafo = trafo_gen.get_trafo_site_to_image(
                trafo_site_to_image=self.trafo_site_to_image, phase=phase,
                phase_ref_site=self.phase_ref_site,
                phase_ref_image=self.phase_ref_image
            )
        # Construct local images
        emissions = ArrayData(np.zeros(self.sites_shape, dtype=float))
        emissions_coord = emissions.get_var_meshgrid()
        image_rect = np.array([
            [0+self.proj_shape[i], image.shape[i]-self.proj_shape[i]-1]
            for i in range(len(self.proj_shape))
        ])
        emissions_mask = new_trafo.get_mask_origin_coords_within_target_rect(
            np.moveaxis(emissions_coord, 0, -1), rect=image_rect
        )
        rec_coord_sites = emissions_coord[:, emissions_mask]
        rec_coord_image = new_trafo.coord_to_target(rec_coord_sites.T).T
        local_images = get_local_images(
            *rec_coord_image, image, self.proj_shape,
            psf_supersample=self.psf_supersample
        )
        # Apply projectors and embed local images
        local_emissions = apply_projectors(
            local_images, self.projector_generator
        )
        emissions.data[emissions_mask] = local_emissions
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
        # Package result
        res = dict(
            trafo=new_trafo,
            emissions=emissions,
            label_center=label_centers,
            histogram=histogram,
            success=state_estimation_success
        )
        if state_estimation_success:
            res.update(dict(
                hist_center=histogram_data["center"],
                hist_threshold=histogram_data["threshold"],
                hist_error_prob=histogram_data["error_prob"],
                hist_error_num=histogram_data["error_num"],
                hist_emission_num=histogram_data["emission_num"],
                hist_peak_info=histogram_data["peak_info"],
                state=state
            ))
        res = ReconstructionResult(**res)
        return res
