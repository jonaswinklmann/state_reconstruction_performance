import copy
import numpy as np

from libics.core.data.arrays import get_coordinate_meshgrid
from libics.core.util import misc

from state_reconstruction.gen.image_gen import get_local_psfs


###############################################################################
# Generate embedded PSFs
###############################################################################


def get_sites_pos_in_roi(
    trafo_site_to_image, rect=None, center=None, size=None
):
    # Parse parameters
    if rect is None:
        rect = [
            [_c - _s / 2, _c + _s / 2]
            for _c, _s in zip(center, size)
        ]
    rect = np.array(rect)
    # Transform image corners into site space
    image_corners = misc.get_combinations(rect)
    sites_corners = np.round(
        trafo_site_to_image.coord_to_origin(image_corners)
    ).astype(int)
    # Find all sites within big rectangle and transform back
    sites_pos = get_coordinate_meshgrid(*[
        np.arange(np.min(_sc), np.max(_sc) + 1)
        for _sc in sites_corners.T
    ])
    sites_pos = np.reshape(sites_pos, (-1, 2))
    image_pos = trafo_site_to_image.coord_to_target(sites_pos)
    # Select sites within image ROI
    mask = np.logical_and.reduce([
        (image_pos[:, i] >= rect[i, 0]) & (image_pos[:, i] <= rect[i, 1])
        for i in range(image_pos.shape[1])
    ], axis=0)
    image_pos = image_pos[mask]
    return image_pos


def get_embedded_local_psfs(
    local_psfs, offset=None, size=None, normalize=True
):
    # Parse parameters
    lpsfs = local_psfs
    integrated_psfs = lpsfs["psf"]
    X_min, X_max = lpsfs["X_min"], lpsfs["X_max"]
    Y_min, Y_max = lpsfs["Y_min"], lpsfs["Y_max"]
    label_num = len(integrated_psfs)
    if offset is None:
        offset = np.min(X_min), np.min(Y_min)
    offset = np.array(offset)
    if size is None:
        size = np.array(np.max(X_max), np.max(Y_max)) - offset
    size = np.array(size)
    # Generate images
    images = []
    for i in range(label_num):
        # Resize to account for borders
        xmin, xmax = X_min[i] - offset[0], X_max[i] - offset[0]
        ymin, ymax = Y_min[i] - offset[1], Y_max[i] - offset[1]
        rois = []
        if xmin < 0:
            rois.append(slice(-xmin, None))
            xmin = 0
        elif xmax > size[0]:
            rois.append(slice(None, size[0] - xmax))
            xmax = size[0]
        else:
            rois.append(slice(None))
        if ymin < 0:
            rois.append(slice(-ymin, None))
            ymin = 0
        elif ymax > size[1]:
            rois.append(slice(None, size[1] - ymax))
            ymax = size[1]
        else:
            rois.append(slice(None))
        rois = tuple(rois)
        ipsf = integrated_psfs[i][rois]
        # Create image
        if normalize:
            ipsf /= np.sum(ipsf)
        im = np.zeros(size, dtype=float)
        im[xmin:xmax, ymin:ymax] = ipsf
        images.append(im)
    return np.array(images)


###############################################################################
# Generate projectors
###############################################################################


def get_embedded_projectors(embedded_psfs):
    epsfs = np.array(embedded_psfs)
    psf_count = len(epsfs)
    im_shape = epsfs.shape[1:]
    epsfs_vec = epsfs.reshape((psf_count, -1))
    projs_vec = np.linalg.pinv(epsfs_vec)
    projs = projs_vec.reshape(im_shape + (psf_count,))
    projs = np.moveaxis(projs, -1, 0)
    return projs


def get_projector(
    trafo_site_to_fluo, integrated_psf_generator,
    dx=0, dy=0, rel_embedding_size=4
):
    center_shift = (
        np.array([dx, dy]) / integrated_psf_generator.psf_supersample
    )
    # Center trafo
    centered_trafo = copy.deepcopy(trafo_site_to_fluo)
    centered_trafo.offset = np.zeros(2)
    # Build embedding area
    embedding_size = rel_embedding_size * integrated_psf_generator.psf_shape
    image_pos = get_sites_pos_in_roi(
        centered_trafo, center=(0, 0), size=embedding_size
    ) + center_shift
    center_idx = np.argmin(np.linalg.norm((image_pos - center_shift), axis=1))
    # Get local PSFs
    local_psfs = get_local_psfs(
        *image_pos.T, integrated_psf_generator=integrated_psf_generator
    )
    embedded_psfs = get_embedded_local_psfs(
        local_psfs,
        offset=-embedding_size//2,
        size=embedding_size+1,
        normalize=True
    )
    # Get projectors
    embedded_projs = get_embedded_projectors(embedded_psfs)
    roi = misc.cv_index_center_to_slice(
        np.array(embedded_psfs.shape[1:]) // 2,
        size=integrated_psf_generator.psf_shape
    )
    center_proj = embedded_projs[center_idx][roi]
    return center_proj


###############################################################################
# Projector generator class
###############################################################################


class ProjectorGenerator:

    def __init__(
        self, trafo_site_to_image=None,
        integrated_psf_generator=None, rel_embedding_size=4
    ):
        # Protected variables
        self._proj_cache = None
        # Public variables
        self.trafo_site_to_image = trafo_site_to_image
        self.integrated_psf_generator = integrated_psf_generator
        self.rel_embedding_size = rel_embedding_size
        self.proj_cache_built = False

    @property
    def psf_supersample(self):
        return self.integrated_psf_generator.psf_supersample

    @property
    def psf_shape(self):
        return self.integrated_psf_generator.psf_shape

    def setup_cache(self, print_progress=False):
        # Verify integrated psf generator cache
        if not self.integrated_psf_generator.psf_integrated_cache_built:
            self.integrated_psf_generator.setup_cache(
                print_progress=print_progress
            )
        # Build projector cache
        _ss = self.psf_supersample
        _hss = _ss // 2
        self._proj_cache = np.full((
            _ss, _ss, self.psf_shape[0], self.psf_shape[1]
        ), np.nan, dtype=float)
        _iter = misc.get_combinations([
            np.arange(-_hss, _ss - _hss),
            np.arange(-_hss, _ss - _hss)
        ])
        if print_progress:
            _iter = misc.iter_progress(_iter)
        for dx, dy in _iter:
            self._proj_cache[dx, dy] = get_projector(
                self.trafo_site_to_image, self.integrated_psf_generator,
                dx=dx, dy=dy, rel_embedding_size=self.rel_embedding_size
            )
        self.proj_cache_built = True

    def generate_projector(self, dx=0, dy=0):
        if self.proj_cache_built:
            return self._proj_cache[dx, dy].copy()
        else:
            return get_projector(
                self.trafo_site_to_image, self.integrated_psf_generator,
                dx=dx, dy=dy, rel_embedding_size=self.rel_embedding_size
            )
