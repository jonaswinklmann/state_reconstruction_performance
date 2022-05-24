import numba as nb
import numpy as np
import scipy.optimize
import scipy.special

from libics.core.data.arrays import ArrayData
from libics.tools.math.signal import find_peaks_1d


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
    local_images = np.zeros((len(X),) + shape, dtype=float)
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
    emissions = np.ravel(emissions)
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
