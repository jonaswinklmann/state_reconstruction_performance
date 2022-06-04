"""
PSF generator.

Generates point spread functions.
"""

import numpy as np
import scipy.special

from libics.core.data.arrays import ArrayData
from libics.core.data.types import AttrHashBase
from libics.core.util import misc
from libics.core import io
from libics.tools.math import peaked


###############################################################################
# Incoherent point spread functions
###############################################################################


def get_psf_gaussian_width(
    wavelength, px_size=1,
    focal_length=None, aperture_radius=None, numerical_aperture=0.68
):
    """
    Gets the width of a Gaussian PSF of a diffraction-limited optical system.

    https://en.wikipedia.org/wiki/Airy_disk#Approximation_using_a_Gaussian_profile

    Parameters
    ----------
    wavelength : `float`
        Imaging wavelength in meter (m).
    px_size : `float`
        Pixel size of imaging system in meter (m). Serves as spatial unit.
    focal_length, aperture_radius : `float`
        Focal length and aperture radius of imaging system in meter (m).
        Alternativ to directly specifying the `numerical_aperture`.
    numerical_aperture : `float`
        Numerical aperture of imaging system.
        Takes precedence over `focal_length`/`aperture_radius`.

    Returns
    -------
    width : `float`
        Gaussian PSF width in units of the `px_size`.
    """
    if focal_length is not None and aperture_radius is not None:
        numerical_aperture = np.sin(np.arctan(aperture_radius / focal_length))
    w = 1 / 2 * 0.84 * wavelength / 2 / numerical_aperture
    return w / px_size


def get_psf_gaussian(wx=5, wy=None, dx=0, dy=0, tilt=0, size=25):
    """
    Gets an array representing a Gaussian PSF.

    Parameters
    ----------
    wx, wy : `float`
        Gaussian width of PSF along the respective axes in pixels.
    dx, dy : `float`
        Center offset along the respective axes in pixels.
    tilt : `float`
        Orientation of symmetry axes in radians (rad).
    size : `float` or `(float, float)`
        Array size of PSF in pixels.

    Returns
    -------
    psf : `ArrayData[2, float]`
        Gaussian PSF array.
    """
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
    Gets the width of an Airy PSF of a diffraction-limited optical system.

    https://www1.univap.br/irapuan/Exame/difracao/difracao.html

    Parameters
    ----------
    wavelength : `float`
        Imaging wavelength in meter (m).
    px_size : `float`
        Pixel size of imaging system in meter (m). Serves as spatial unit.
    focal_length, aperture_radius : `float`
        Focal length and aperture radius of imaging system in meter (m).
        Alternativ to directly specifying the `numerical_aperture`.
    numerical_aperture : `float`
        Numerical aperture of imaging system.
        Takes precedence over `focal_length`/`aperture_radius`.

    Returns
    -------
    width : `float`
        Airy PSF width in units of the `px_size`.
    """
    if focal_length is not None and aperture_radius is not None:
        numerical_aperture = np.sin(np.arctan(aperture_radius / focal_length))
    k = 2 * np.pi / wavelength
    w = 1 / numerical_aperture / k
    return w / px_size


def get_psf_airy(wx=5, wy=None, dx=0, dy=0, size=25):
    """
    Gets an array representing an Airy PSF.

    Parameters
    ----------
    wx, wy : `float`
        Airy width of PSF along the respective axes in pixels.
    dx, dy : `float`
        Center offset along the respective axes in pixels.
    tilt : `float`
        Orientation of symmetry axes in radians (rad).
    size : `float` or `(float, float)`
        Array size of PSF in pixels.

    Returns
    -------
    psf : `ArrayData[2, float]`
        Airy PSF array.
    """
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
    """
    Gets a binned PSF with subpixel shift.

    Parameters
    ----------
    psf : `Array[2, float]`
        Fully resolved PSF array.
    dx, dy : `int`
        Center shift of PSF in (fully resolved) pixels.
        Used to generate subpixel-precise binned PSFs (binned pixels).
    integration_size : `int`
        Binning size in (fully resolved) pixels.

    Returns
    -------
    new_psf : `np.ndarray(2, float)`
        Binned PSF with shape `psf.shape // integration_size`.
    """
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


class IntegratedPsfGenerator(AttrHashBase):

    """
    Class for generating binned PSFs.

    Uses pre-calculation to allow for high-performance generation
    of subpixel-shifted binned PSFs.

    Parameters
    ----------
    psf : `Array[2, float]`
        Fully resolved PSF.
    psf_supersample : `int`
        Binning size.

    Attributes
    ----------
    psf_integrated_cache_built : `bool`
        Whether the internal cache has been set up.

    Examples
    --------
    Standard use case given a fully resolved PSF:

    >>> psf.shape
    (105, 105)
    >>> ipsfgen = IntegratedPsfGenerator(
    ...     psf=psf, psf_supersample=5
    ... )
    >>> ipsfgen.setup_cache()
    >>> ipsfgen.psf_integrated_cache_built
    True
    >>> ipsfgen.generate_integrated_psf(dx=1, dy=-2).shape
    (21, 21)
    """

    HASH_KEYS = AttrHashBase.HASH_KEYS | {"psf", "psf_supersample"}

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
        if np.any(np.array(val.shape) % 2 == 0):
            raise ValueError("Invalid `psf` (shape must be odd)")
        self._psf = val
        self.psf_integrated_cache_built = False

    @property
    def psf_supersample(self):
        return self._psf_supersample

    @psf_supersample.setter
    def psf_supersample(self, val):
        if val % 2 == 0:
            raise ValueError("Invalid `psf_supersample` (must be odd)")
        self._psf_supersample = val
        self.psf_integrated_cache_built = False

    @property
    def psf_shape(self):
        return np.array(self.psf.shape) // self.psf_supersample

    def save(self, fp):
        """
        Saves PSF and supersampling to a *.json file.

        Parameters
        ----------
        fp : `str`
            Save file path.
        """
        _d = {
            "psf": self.psf,
            "psf_supersample": self.psf_supersample
        }
        io.save(misc.assume_endswith(fp, ".json"), _d)

    @staticmethod
    def load(fp):
        """
        Loads a PSF file, returns an :py:class:`IntegratedPsfGenerator` object.
        """
        _d = io.load(misc.assume_endswith(fp, ".json"))
        if "psf" not in _d or "psf_supersample" not in _d:
            raise FileNotFoundError("Invalid file")
        return IntegratedPsfGenerator(
            psf=_d["psf"], psf_supersample=_d["psf_supersample"]
        )

    def setup_cache(self, print_progress=False):
        """
        Sets up the cache for generating subpixel-shifted binned PSFs.
        """
        _ss = self.psf_supersample
        _hss = _ss // 2
        self._psf_integrated = np.full((
            _ss, _ss, self.psf.shape[0] // _ss, self.psf.shape[1] // _ss
        ), np.nan, dtype=float)
        _iter = misc.get_combinations([
            np.arange(-_hss, _ss - _hss),
            np.arange(-_hss, _ss - _hss)
        ])
        if print_progress:
            _iter = misc.iter_progress(_iter)
        for dx, dy in _iter:
            self._psf_integrated[dx, dy] = get_integrated_psf(
                self.psf, dx=dx, dy=dy, integration_size=_ss
            )
        self.psf_integrated_cache_built = True

    def generate_integrated_psf(self, dx=0, dy=0):
        """
        Gets the binned PSF with subpixel shift.

        Parameters
        ----------
        dx, dy : `int`
            PSF center shift in units of fully resolved pixels.

        Returns
        -------
        integrated_psf : `np.ndarray(2, float)`
            Normalized binned PSF.
        """
        if self.psf_integrated_cache_built:
            return self._psf_integrated[dx, dy].copy()
        else:
            return get_integrated_psf(
                self.psf, dx=dx, dy=dy, integration_size=self.psf_supersample
            )
