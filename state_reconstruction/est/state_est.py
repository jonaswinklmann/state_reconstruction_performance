import numba as nb
import numpy as np
import scipy.optimize
import scipy.special

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData
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
    images = local_images["image"]
    images = images.reshape((len(images), -1))
    if not projector_generator.proj_cache_built:
        projector_generator.setup_cache()
    projs = projector_generator._proj_cache
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
    emissions, min_peak_rel_dist=2, bin_range=None
):
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
    has_second_peak = (len(peaks["center"]) == 3)
    if has_second_peak:
        if peaks["center"][2] < peaks["center"][1] + peaks["width"][1]:
            has_second_peak = False
    # Separate peaks by minimizing errors
    a0, a1 = [_fit.a for _fit in peaks["fit"][:2]]
    x0, x1 = [_x for _x in peaks["center"][:2]]
    w0, w1 = [_w for _w in peaks["width"][:2]]
    x01_res = scipy.optimize.minimize(
        _double_erf_overlap, np.mean([x0, x1]),
        args=(a0, a1, x0, x1, w0, w1)
    )
    x01 = x01_res.x[0]
    if has_second_peak:
        a2, x2, w2 = peaks["fit"][2].a, peaks["center"][2], peaks["width"][2]
        x12_res = scipy.optimize.minimize(
            _double_erf_overlap, np.mean([x0, x1]),
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
    emissions_num0 = np.count_nonzero(emissions < x01)
    emissions_num1 = np.count_nonzero(
        (emissions >= x01) & (emissions <= x12)
    )
    emissions_num2 = np.count_nonzero(emissions > x12)
    return {
        "histogram": hist,
        "center": [x0, x1, x2],
        "threshold": [x01, x12],
        "error_num": [err01, err12],
        "emission_num": [emissions_num0, emissions_num1, emissions_num2]
    }


def get_state_estimate(emissions, thresholds):
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


class ReconstructionResult:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StateEstimator:

    LOGGER = get_logger("srec.StateEstimator")

    def __init__(
        self,
        projector_generator=None,
        isolated_locator=None,
        trafo_site_to_image=None,
        sites_shape=(170, 170)
    ):
        self.projector_generator = projector_generator
        self.isolated_locator = isolated_locator
        self.trafo_site_to_image = trafo_site_to_image
        self.sites_shape = sites_shape

    def setup(self, print_progress=True):
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
            trafo_site_to_image=val, phase=np.zeros(2)
        )

    @property
    def psf_shape(self):
        return self.projector_generator.psf_shape

    @property
    def psf_supersample(self):
        return self.projector_generator.psf_supersample

    def reconstruct(self, image, new_trafo=None):
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
                trafo_site_to_image=self.trafo_site_to_image, phase=phase
            )
        # Construct local images
        emissions = ArrayData(np.zeros(self.sites_shape, dtype=float))
        emissions_coord = emissions.get_var_meshgrid()
        image_rect = np.array([
            [0+self.psf_shape[i], image.shape[i]-self.psf_shape[i]-1]
            for i in range(len(self.psf_shape))
        ])
        emissions_mask = new_trafo.get_mask_origin_coords_within_target_rect(
            np.moveaxis(emissions_coord, 0, -1), rect=image_rect
        )
        rec_coord_sites = emissions_coord[:, emissions_mask]
        rec_coord_image = new_trafo.coord_to_target(rec_coord_sites.T).T
        local_images = get_local_images(
            *rec_coord_image, image, self.psf_shape,
            psf_supersample=self.psf_supersample
        )
        # Apply projectors and embed local images
        local_emissions = apply_projectors(
            local_images, self.projector_generator
        )
        emissions.data[emissions_mask] = local_emissions
        # Perform histogram analysis for state discrimination
        histogram_data = get_emission_histogram(local_emissions)
        # Package result
        res = ReconstructionResult(
            trafo=new_trafo,
            emissions=emissions,
            label_center=label_centers,
            histogram=histogram_data["histogram"],
            hist_center=histogram_data["center"],
            hist_threshold=histogram_data["threshold"],
            hist_error_num=histogram_data["error_num"],
            hist_emission_num=histogram_data["emission_num"],
            state=get_state_estimate(emissions, histogram_data["threshold"])
        )
        return res
