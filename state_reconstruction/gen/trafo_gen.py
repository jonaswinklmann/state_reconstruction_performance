import numpy as np

from librbl.seq.data import get_trafo_site_to_fluo


def get_trafo_site_to_image(
    angle=np.deg2rad((0, 0)),
    relative_magnification=1
):
    _m = np.mean(get_trafo_site_to_fluo().get_origin_axes()[0])
    gen_trafo_site_to_fluo = get_trafo_site_to_fluo(
        magnification=np.full(2, _m*relative_magnification),
        angle=angle,
        phase_ref_site=(84, 84),
        phase_ref_fluo=(255, 255)
    )
    return gen_trafo_site_to_fluo
