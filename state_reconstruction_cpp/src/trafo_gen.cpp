#include "trafo_gen.hpp"

#include <exception>

#include "linear.hpp"

Eigen::VectorXf get_phase_from_trafo_site_to_image(AffineTrafo2D trafo_site_to_image, std::optional<Eigen::VectorXf> phase_ref_image)
{
    /*Gets the lattice phase and ref. integer site in `librbl` convention.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d`
        Transformation between lattice sites and fluorescence image.
    phase_ref_fluo : `np.ndarray(1, float)`
        Fluorescence image coordinate corresponding to integer lattice site.

    Returns
    -------
    phase : `np.ndarray(1, float)`
        Phase (residual) of lattice w.r.t. to image coordinates (0, 0).
    site : `np.ndarray(1, float)`
        Nearest integer lattice site.*/

    if(!phase_ref_image.has_value())
    {
        //phase_ref_image = TrafoManager.get_phase_ref_image()
        throw std::logic_error("Calling without phase_ref_image not implemented yet");
    }
    Eigen::VectorXf site_float = trafo_site_to_image.coord_to_origin(phase_ref_image.value());
    Eigen::ArrayXf phaseArray = site_float.array() + 0.5;
    //phaseArray = phaseArray.unaryExpr([](const float x) { return fmod(x, 1) - 0.5; });
    phaseArray -= phaseArray.cast<int>().cast<float>();
    phaseArray -= 0.5;
    return phaseArray.matrix();
}