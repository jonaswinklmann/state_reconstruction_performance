"""
Affine transformation estimator

Estimates the affine transformation parameters between lattice sites
and fluorescence image coordinates.
"""

import numpy as np
import scipy.optimize
import time

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData
from libics.core.util import misc
from libics.tools.trafo.linear import AffineTrafo2d
from libics.tools.math.signal import find_peaks_1d
from libics.tools.math.models import ModelBase
from libics.tools.math.peaked import FitGaussian1d, gaussian_1d, FitParabolic1d


###############################################################################
# Transformation parameter optimization
###############################################################################


def get_trafo_angle_fidelity(
    x, y, guess_trafo, ax=(0, 1),
    bins_per_site=5, peak_rel_prominence=0.1, peak_base_prominence_ratio=0.1,
    rel_spacing_tol=0.2
):
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
        if self.isolated_locator is None:
            raise RuntimeError("invalid `isolated_locator`")
        if self.guess_trafo is None:
            raise RuntimeError("invalid `guess_trafo`")
        return True

    def get_optimized_trafo(self, image, print_progress=False):
        self.check_setup()
        # Find atom centers
        label_centers = self.isolated_locator.get_label_centers(image)
        if len(label_centers[..., 0]) < self.min_isolated_num:
            return False
        # Optimize trafo
        trafo_params = get_trafo_from_sites_direct(
            *np.transpose(label_centers), self.guess_trafo,
            print_progress=print_progress
        )
        opt_trafo = trafo_params["trafo"]
        return opt_trafo

    def find_trafo(self, *images, print_progress=False):
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
        wx = np.mean(peaks["width"])
        c = np.min(func_data)
        a = (np.max(func_data) - c) / 2
        self.p0 = [a, x0, wx, dx, c]
