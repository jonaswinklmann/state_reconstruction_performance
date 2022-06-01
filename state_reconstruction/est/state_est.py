"""
State estimator.

Projects an image into site space and assigns a emission state.
"""

import json
import PIL
import numba as nb
import numpy as np
import scipy.optimize
import scipy.special

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData
from libics.core import io
from libics.tools.math.signal import find_peaks_1d

from state_reconstruction.gen import trafo_gen
from .trafo_est import get_trafo_phase_from_points


###############################################################################
# Projector application
###############################################################################


@nb.njit(
    nb.void(nb.float64[:, :, :], nb.float64[:, :],
            nb.int32[:], nb.int32[:], nb.int32[:], nb.int32[:]),
    parallel=True, cache=True
)
def _slice_local_image(local_images, image, X_min, X_max, Y_min, Y_max):
    for i in nb.prange(len(local_images)):
        local_images[i] = image[X_min[i]:X_max[i], Y_min[i]:Y_max[i]]


def get_local_images(X, Y, image, shape, psf_supersample=5):
    """
    Extracts image subregions and subpixel shifts.

    Parameters
    ----------
    X, Y : `Array[1, float]`
        (Fractional) center positions in image coordinates.
    image : `Array[2, float]`
        Full-sized image.
    shape : `(int, int)`
        Shape of subregion images.
    psf_supersample : `int`
        Supersampling size
        (used to convert fractional positions to subpixel shifts).

    Returns
    -------
    ret : `dict(str->Any)`
        Returns a dictionary containing the following items:
    image : `np.ndarray(3, float)`
        Subregion images. Dimension: `[n_subregions, {shape}]`.
    X_int, Y_int : `np.ndarray(1, int)`
        Rounded PSF center coordinates.
    X_min, X_max, Y_min, Y_max : `np.ndarray(1, int)`
        Rounded PSF rectangle corners.
    dx, dy : `np.ndarray(1, int)`
        Subpixel shifts.
    """
    X_int, Y_int = np.round(X).astype(int), np.round(Y).astype(int)
    dx = np.round((X - X_int) * psf_supersample).astype(int)
    dy = np.round((Y - Y_int) * psf_supersample).astype(int)
    X_min, Y_min = X_int - shape[0] // 2, Y_int - shape[1] // 2
    X_max, Y_max = X_min + shape[0], Y_min + shape[1]
    local_images = np.zeros((len(X),) + tuple(shape), dtype=float)
    image = np.array(image)
    _slice_local_image(local_images, image, X_min, X_max, Y_min, Y_max)
    return {
        "image": local_images, "X_int": X_int, "Y_int": Y_int,
        "X_min": X_min, "X_max": X_max, "Y_min": Y_min, "Y_max": Y_max,
        "dx": dx, "dy": dy
    }


def apply_projectors(local_images, projector_generator):
    """
    Applies subpixel-shifted projectors to subregion images.

    Parameters
    ----------
    local_images : `np.ndarray(3, float)`
        Subregion images. Dimension: `[n_subregions, {shape}]`.
    projector_generator : `ProjectorGenerator`
        Projector generator object.

    Returns
    -------
    emissions : `np.ndarray(1, float)`
        Projected results. Dimensions: `[n_subregions]`.
    """
    images = local_images["image"]
    images = images.reshape((len(images), -1))
    if not projector_generator.proj_cache_built:
        projector_generator.setup_cache()
    projs = projector_generator.proj_cache
    projs = projs.reshape((projs.shape[:2]) + (-1,))
    xidx = local_images["dx"] % projector_generator.psf_supersample
    yidx = local_images["dy"] % projector_generator.psf_supersample
    emissions = [
        _im @ projs[_x, _y]
        for _im, _x, _y in zip(images, xidx, yidx)
    ]
    return np.array(emissions)


###############################################################################
# Emission histogram analysis
###############################################################################


def _double_erf_overlap(x, a0, a1, x0, x1, w0, w1):
    return (
        a0 * scipy.special.erf((x - x0) / w0 / np.sqrt(2))
        - a1 * scipy.special.erf((x - x1) / w1 / np.sqrt(2))
        - a0 - a1
    )**2


def _erf_probability(x, x0, wx):
    return 0.5 * (scipy.special.erf((x - x0) / wx / np.sqrt(2)) + 1)


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
    bins = min(size / 4, np.sqrt(size))
    if bins > size / 4:
        bins = size / 4
    bins = np.round(bins).astype(int)
    _h, _e = np.histogram(emissions, bins=bins, range=bin_range)
    _c = (_e[1:] + _e[:-1]) / 2
    hist = ArrayData(_h)
    hist.set_data_quantity(name="histogram")
    hist.set_dim(0, points=_c)
    hist.set_var_quantity(0, name="projected emission")
    return hist


def analyze_emission_histogram(hist, min_peak_rel_dist=2):
    """
    Gets state discrimination thresholds from projected images.

    Uses the histogram to find two or three peaks. The thresholds are set
    by setting equal false negativer/positive rates.

    Parameters
    ----------
    hist : `ArrayData(1, float)`
        Projected image histogram.
    min_peak_rel_dist : `int`
        If the spacing between the first two peaks is smaller than
        the product of background peak width and `min_peak_rel_dist`,
        an error is raised.

    Returns
    -------
    ret : `dict(str->Any)`
        Returns a dictionary containing the following items:
    center : `[float, float, float]`
        Peak centers.
    threshold : `[float, float]`
        State thresholds.
    error_num : `[float, float]`
        False positive/negative counts at first/second threshold.
    emission_num : `[float, float, float]`
        Number of sites mapped to the different states.
    """
    # Find histogram peaks
    peaks = find_peaks_1d(
        hist, npeaks=3, rel_prominence=0, base_prominence_ratio=0.1,
        check_npeaks=False,
        edge_peaks=True, fit_algorithm="gaussian", ret_vals=["width", "fit"]
    )
    if (
        peaks["center"][1]
        < peaks["center"][0] + peaks["width"][0] * min_peak_rel_dist
    ):
        raise RuntimeError("background peak stronger than signal peak")
    # Separate peaks by minimizing errors
    a0, a1 = [_fit.a for _fit in peaks["fit"][:2]]
    x0, x1 = [_x for _x in peaks["center"][:2]]
    w0, w1 = [_w for _w in peaks["width"][:2]]
    x01_res = scipy.optimize.minimize(
        _double_erf_overlap, np.mean([x0, x1]),
        args=(a0, a1, x0, x1, w0, w1)
    )
    x01 = x01_res.x[0]
    # Handle second peak
    has_second_peak = (len(peaks["center"]) == 3)
    if has_second_peak:
        if peaks["center"][2] - x1 < x1 - x01:
            has_second_peak = False
    if has_second_peak:
        a2, x2, w2 = peaks["fit"][2].a, peaks["center"][2], peaks["width"][2]
        x12_res = scipy.optimize.minimize(
            _double_erf_overlap, np.mean([x1, x2]),
            args=(a1, a2, x1, x2, w1, w2)
        )
        x12 = x12_res.x[0]
    else:
        x2 = 2 * x1 - x0
        x12 = 2 * x1 - x01
    # Package result
    err01 = a1 * _erf_probability(x01, x1, w1)
    if has_second_peak:
        err12 = a2 * _erf_probability(x12, x2, w2)
    else:
        err12 = a1 * (1 - _erf_probability(x12, x1, w1))
    _h = np.array(hist)
    emissions_num0 = np.sum(_h[hist.get_points(0) < x01])
    emissions_num1 = np.sum(_h[
        (hist.get_points(0) >= x01) & (hist.get_points(0) <= x12)
    ])
    emissions_num2 = np.sum(_h[hist.get_points(0) > x12])
    return {
        "center": [x0, x1, x2],
        "threshold": [x01, x12],
        "error_num": [err01, err12],
        "emission_num": [emissions_num0, emissions_num1, emissions_num2]
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
    hist_center : `[float, float, float]`
        Projected center value for the states.
    hist_threshold : `[float, float]`
        State discrimination threshold of projected values.
    hist_error_num : `[float, float]`
        False negative/positive number at thresholds.
    hist_emission_num : `[int, int, int]`
        Number of sites associated to each state.

    Notes
    -----
    This object can be saved with all information using :py:meth:`save`
    in the formats `"json", "bson"`.
    The (integer) state array itself can be saved using :py:meth:`save_state`
    in the formats `"csv", "txt", "json", "png"`.
    """

    _attributes = {
        "trafo", "emissions", "state", "label_center",
        "histogram", "hist_center", "hist_threshold",
        "hist_error_num", "hist_emission_num", "success"
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
        sites_shape=(170, 170)
    ):
        self.projector_generator = projector_generator
        self.isolated_locator = isolated_locator
        self.phase_ref_site = phase_ref_site
        self.phase_ref_image = phase_ref_image
        self.trafo_site_to_image = trafo_site_to_image
        self.sites_shape = sites_shape

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
        histogram = get_emission_histogram(local_emissions)
        try:
            histogram_data = analyze_emission_histogram(histogram)
            state = get_state_estimate(emissions, histogram_data["threshold"])
            state_estimation_success = True
        except (RuntimeError, IndexError):
            self.LOGGER.error("emission histogram analysis failed")
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
                hist_error_num=histogram_data["error_num"],
                hist_emission_num=histogram_data["emission_num"],
                state=state
            ))
        res = ReconstructionResult(**res)
        return res
