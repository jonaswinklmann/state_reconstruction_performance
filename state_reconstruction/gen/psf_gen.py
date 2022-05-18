import numpy as np
import scipy.special

from libics.core.data.arrays import ArrayData
from libics.core.util import misc
from libics.tools.math import peaked


###############################################################################
# Incoherent point spread functions
###############################################################################


def get_psf_gaussian_width(
    wavelength, px_size=1,
    focal_length=None, aperture_radius=None, numerical_aperture=0.68
):
    """
    https://en.wikipedia.org/wiki/Airy_disk#Approximation_using_a_Gaussian_profile
    """
    if focal_length is not None and aperture_radius is not None:
        numerical_aperture = np.sin(np.arctan(aperture_radius / focal_length))
    w = 1 / 2 * 0.84 * wavelength / 2 / numerical_aperture
    return w / px_size


def get_psf_gaussian(wx=5, wy=None, dx=0, dy=0, tilt=0, size=25):
    if wy is None:
        wy = wx
    if np.isscalar(size):
        size = np.full(2, size)
    size = np.array(size, dtype=int)
    psf = ArrayData()
    psf.add_dim(2)
    psf.var_shape = size
    psf.data = peaked.gaussian_2d_tilt(
        psf.get_var_meshgrid(), 1,
        psf.get_center(0)+dx, psf.get_center(1)+dy,
        wx, wy, tilt
    )
    psf.data /= np.sum(psf.data)
    return psf


def get_psf_airy_width(
    wavelength, px_size=1,
    focal_length=None, aperture_radius=None, numerical_aperture=0.68
):
    """
    https://www1.univap.br/irapuan/Exame/difracao/difracao.html
    """
    if focal_length is not None and aperture_radius is not None:
        numerical_aperture = np.sin(np.arctan(aperture_radius / focal_length))
    k = 2 * np.pi / wavelength
    w = 1 / numerical_aperture / k
    return w / px_size


def get_psf_airy(wx=5, wy=None, dx=0, dy=0, size=25):
    if wy is None:
        wy = wx
    if np.isscalar(size):
        size = np.full(2, size)
    size = np.array(size, dtype=int)
    psf = ArrayData()
    psf.add_dim(2)
    psf.var_shape = size
    x, y = psf.get_var_meshgrid()
    x, y = x - psf.get_center(0) - dx, y - psf.get_center(1) - dy
    u = np.sqrt((x / wx)**2 + (y / wy)**2)
    psf.data = np.ones(size, dtype=float)
    np.true_divide(2 * scipy.special.j1(u), u, where=(u != 0), out=psf.data)
    psf.data = psf.data**2
    psf.data /= np.sum(psf.data)
    return psf


###############################################################################
# Integrated point spread functions
###############################################################################


def get_integrated_psf(psf, dx=0, dy=0, integration_size=5):
    if np.any(np.array(psf.shape) % integration_size != 0):
        raise ValueError("`psf.shape` must be multiple of `integration_size`")
    new_shape = np.array(psf.shape) // integration_size
    if dx == 0 and dy == 0:
        new_psf = psf
    else:
        new_psf = np.zeros_like(psf, dtype=float)
        old_slices, new_slices = [], []
        if dx == 0:
            old_slices.append(slice(None))
            new_slices.append(slice(None))
        elif dx > 0:
            old_slices.append(slice(None, -dx))
            new_slices.append(slice(dx, None))
        elif dx < 0:
            old_slices.append(slice(-dx, None))
            new_slices.append(slice(None, dx))
        if dy == 0:
            old_slices.append(slice(None))
            new_slices.append(slice(None))
        elif dy > 0:
            old_slices.append(slice(None, -dy))
            new_slices.append(slice(dy, None))
        elif dy < 0:
            old_slices.append(slice(-dy, None))
            new_slices.append(slice(None, dy))
        new_psf[tuple(new_slices)] = psf[tuple(old_slices)]
    new_psf = np.mean(
        np.reshape(
            new_psf,
            (new_shape[0], integration_size, new_shape[1], integration_size)
        ), axis=(1, 3)
    )
    new_psf /= np.sum(new_psf)
    return new_psf


class IntegratedPsfGenerator:

    def __init__(self, psf=None, psf_supersample=5):
        # Protected variables
        self._psf = None
        self._psf_supersample = None
        self._psf_integrated = None
        # Public variables
        self.psf = psf
        self.psf_supersample = psf_supersample
        self.psf_integrated_cache_built = False

    @property
    def psf(self):
        return self._psf

    @psf.setter
    def psf(self, val):
        self._psf = val
        self.psf_integrated_cache_built = False

    @property
    def psf_supersample(self):
        return self._psf_supersample

    @psf_supersample.setter
    def psf_supersample(self, val):
        self._psf_supersample = val
        self.psf_integrated_cache_built = False

    def setup_cache(self, print_progress=False):
        _ss = self.psf_supersample
        _hss = _ss // 2
        self._psf_integrated = np.full((
            _ss, _ss, self.psf.shape[0] // _ss, self.psf.shape[1] // _ss
        ), np.nan, dtype=float)
        _iter = misc.get_combinations([
            np.arange(_ss - _hss, 2 * _ss - _hss),
            np.arange(_ss - _hss, 2 * _ss - _hss)
        ])
        if print_progress:
            _iter = misc.iter_progress(_iter)
        for dx, dy in _iter:
            self._psf_integrated[dx, dy] = get_integrated_psf(
                self.psf, dx=dx, dy=dy, integration_size=_ss
            )
        self.psf_integrated_cache_built = True

    def generate_integrated_psf(self, dx=0, dy=0):
        if self.psf_integrated_cache_built:
            return self._psf_integrated[dx, dy]
        else:
            get_integrated_psf(
                self.psf, dx=dx, dy=dy, integration_size=self.psf_supersample
            )
