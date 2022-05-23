import numpy as np

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData, get_coordinate_meshgrid

from .psf_gen import IntegratedPsfGenerator

LOGGER = get_logger("srec.gen.image_gen")


###############################################################################
# Generate sites
###############################################################################


def get_sites_mi(
    center=(85, 85), size=(15, 15), filling=None, shape="round",
    outside_count=None, outside_size=None, seed=None
):
    # Parse parameters
    if np.isscalar(center):
        center = (center, center)
    center = np.array(center)
    if np.isscalar(size):
        size = (size, size)
    size = np.array(size)
    # Generate rectangular sites
    ad = ArrayData()
    ad.add_dim(offset=center[0]-size[0]//2, step=1)
    ad.add_dim(offset=center[1]-size[1]//2, step=1)
    ad.var_shape = size
    X, Y = ad.get_var_meshgrid()
    X, Y = X.ravel(), Y.ravel()
    # Remove sites outside ellipse
    _shape_is_round = (
        shape.lower() == "round" or shape.lower()[:8] == "elliptic"
    )
    if _shape_is_round:
        R = np.sqrt(
            ((X - center[0]) / size[0])**2
            + ((Y - center[1]) / size[1])**2
        )
        mask = R <= 0.5
        X, Y = X[mask], Y[mask]
    # Remove atoms for finite filling
    if filling is not None:
        np.random.seed(seed)
        mask = np.random.random(len(X)) <= filling
        X, Y = X[mask], Y[mask]
    # Add isolated atoms outside MI
    if outside_count is not None:
        if outside_size is None:
            outside_size = 2 * center
        if np.isscalar(outside_size):
            outside_size = (outside_size, outside_size)
        outside_size = np.array(outside_size)
        np.random.seed(seed)
        X_iso, Y_iso = [], []
        if np.any(outside_size <= size):
            LOGGER.warn(
                "get_sites_mi: `outside_size` should be larger than `size`"
            )
        else:
            _outside_half_size = np.round(outside_size / 2).astype(int)
            _coords = np.array(get_coordinate_meshgrid(*[
                np.arange(-_ohs, _ohs + 1)
                for _ohs in _outside_half_size
            ]))
            _coords = np.moveaxis(_coords, 0, -1).reshape((-1, 2))
            if _shape_is_round:
                _accept = np.linalg.norm(_coords / (size / 2), axis=-1) > 1
            else:
                _accept = np.all(np.abs(_coords) > (size / 2), axis=-1)
            _coords = _coords[_accept]
            _choice_idx = np.random.choice(
                np.arange(len(_coords)), size=outside_count, replace=False
            )
            X_iso, Y_iso = np.moveaxis(_coords[_choice_idx] + center, -1, 0)
        X, Y = np.concatenate([X, X_iso]), np.concatenate([Y, Y_iso])
    return X, Y


###############################################################################
# Generate image
###############################################################################


def get_brightness_sample(
    X, Y, std=0.2, additive_distributions=None, seed=None
):
    """
    additive_distributions : `Iter[[float or callable, float, float]]`
        Each item leads to additive brightness. The items are interpreted as:
        `[probability to add this distribution, mean, std]`.
        If a `float` is given for the probability, it is uniformly applied
        to all atoms. If a `callable` is given, it should have the signature:
        `func(X, Y)->probability`.
    """
    np.random.seed(seed)
    brightness = np.random.normal(loc=1, scale=std, size=len(X))
    if additive_distributions:
        for _prob, _mean, _std in additive_distributions:
            if callable(_prob):
                _prob = _prob(X, Y)
            mask = np.random.random(size=len(X)) <= _prob
            additive = np.random.normal(
                loc=_mean, scale=_std, size=np.count_nonzero(mask)
            )
            brightness[mask] += additive
    return brightness


def get_local_psfs(
    X, Y, integrated_psf_generator=None, psf=None, integration_size=5
):
    if integrated_psf_generator is None:
        integrated_psf_generator = IntegratedPsfGenerator(
            psf=psf, psf_supersample=integration_size
        )
    psfgen = integrated_psf_generator
    X_int, Y_int = np.round(X).astype(int), np.round(Y).astype(int)
    X_res, Y_res = X - X_int, Y - Y_int
    dx = np.round(X_res * psfgen.psf_supersample).astype(int)
    dy = np.round(Y_res * psfgen.psf_supersample).astype(int)
    psf_shape = np.array(psfgen.psf.shape) // psfgen.psf_supersample
    X_min, Y_min = X_int - psf_shape[0] // 2, Y_int - psf_shape[1] // 2
    X_max, Y_max = X_min + psf_shape[0], Y_min + psf_shape[1]
    integrated_psfs = [
        psfgen.generate_integrated_psf(dx=dx[i], dy=dy[i])
        for i in range(len(X))
    ]
    return {
        "psf": integrated_psfs, "X_int": X_int, "Y_int": Y_int,
        "X_min": X_min, "X_max": X_max, "Y_min": Y_min, "Y_max": Y_max
    }


def get_image_clean(
    X=None, Y=None, psf=None, local_psfs=None,
    brightness=1, integration_size=5, size=(512, 512)
):
    """
    Either provide (X, Y, psf) or (local_psfs)
    """
    if np.isscalar(size):
        size = (size, size)
    if np.isscalar(brightness):
        brightness = np.full_like(X, brightness)
    if local_psfs is None:
        if X is None or Y is None or psf is None:
            raise ValueError("invalid parameters")
        lpsfs = get_local_psfs(
            X, Y, psf=psf, integration_size=integration_size
        )
    else:
        lpsfs = local_psfs
    integrated_psfs = lpsfs["psf"]
    X_min, X_max = lpsfs["X_min"], lpsfs["X_max"]
    Y_min, Y_max = lpsfs["Y_min"], lpsfs["Y_max"]
    img = np.zeros(size, dtype=float)
    for i, integ_psf in enumerate(integrated_psfs):
        img[X_min[i]:X_max[i], Y_min[i]:Y_max[i]] += integ_psf * brightness[i]
    return img


def get_image_sample(
    image_clean, counts_per_atom, normalize_counts=False,
    white_noise_counts=None, seed=None
):
    np.random.seed(seed)
    image_sample = np.random.poisson(
        lam=image_clean*counts_per_atom, size=image_clean.shape
    ).astype(float)
    if white_noise_counts is not None:
        image_sample += np.random.normal(
            loc=0, scale=white_noise_counts, size=image_clean.shape
        )
    if normalize_counts is not False:
        image_sample /= counts_per_atom
    return image_sample


###############################################################################
# Image generator class
###############################################################################


class ImageGenerator:

    def __init__(
        self, image_size=(512, 512),
        image_counts_per_atom=850*14, image_counts_white_noise=70,
        sites_size=(170, 170),
        atoms_center=(85, 85), atoms_size=(30, 30),
        atoms_filling=0.9, atoms_shape="round",
        atoms_outside_count=None, atoms_outside_size=None,
        atoms_fluo_std=0.15, atoms_fluo_add_distr=None,
        trafo_site_to_image=None,
        psf=None, psf_supersample=None
    ):
        # Image configuration
        self.image_size = image_size
        self.image_counts_per_atom = image_counts_per_atom
        self.image_counts_white_noise = image_counts_white_noise
        # Sites configuration
        self.sites_size = sites_size
        # Atoms configuration
        self.atoms_center = atoms_center
        self.atoms_size = atoms_size
        self.atoms_filling = atoms_filling
        self.atoms_shape = atoms_shape
        self.atoms_outside_count = atoms_outside_count
        self.atoms_outside_size = atoms_outside_size
        self.atoms_fluo_std = atoms_fluo_std
        self.atoms_fluo_add_distr = atoms_fluo_add_distr
        self.trafo_site_to_image = trafo_site_to_image
        # Point spread function configuration
        self.psf = psf
        self.psf_supersample = psf_supersample

    def generate_image(self, seed=None, sites_phase=None):
        # Generate filled sites
        _sites = np.array(get_sites_mi(
            center=self.atoms_center, size=self.atoms_size,
            filling=self.atoms_filling, shape=self.atoms_shape,
            outside_count=self.atoms_outside_count,
            outside_size=self.atoms_outside_size,
            seed=seed
        ))
        # Generate random lattice phase
        if sites_phase is None:
            sites_phase = np.random.random(size=2)
        if np.allclose(sites_phase, 0):
            _trafo = self.trafo_site_to_image
        else:
            _trafo = self.trafo_site_to_image.shift_target_axes(sites_phase)
        # Transform into camera space
        _positions = np.transpose(_trafo.coord_to_target(np.transpose(_sites)))
        # Randomize fluorescence per atom
        _brightness = get_brightness_sample(
            *_positions, std=self.atoms_fluo_std,
            additive_distributions=self.atoms_fluo_add_distr,
            seed=seed
        )
        # Generate clean image
        _image_clean = get_image_clean(
            *_positions, self.psf, brightness=_brightness,
            integration_size=self.psf_supersample
        )
        # Generate randomized counts
        _image_sample = get_image_sample(
            _image_clean, self.image_counts_per_atom, normalize_counts=False,
            seed=seed, white_noise_counts=self.image_counts_white_noise
        )
        return ArrayData(_image_sample)
