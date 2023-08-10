#include "trafo_gen.hpp"

#include <exception>
#include <fstream>

#include "linear.hpp"

TrafoManager::TrafoManager() : phase_ref_image(std::nullopt), phase_ref_site((std::nullopt)), trafos_site_to_image()
{}

void TrafoManager::setPhaseRefImage(Eigen::Array2d phaseRefImage)
{
    this->phase_ref_image = phaseRefImage;
}

void TrafoManager::setPhaseRefSite(Eigen::Array2d phaseRefSite)
{
    this->phase_ref_site = phaseRefSite;
}

void TrafoManager::setTrafosSiteToImage(std::map<std::string, AffineTrafo2D> trafos)
{
    this->trafos_site_to_image = trafos;
}

AffineTrafo2D TrafoManager::get_trafo_site_to_image(std::optional<std::string> key)
{
    if(key.has_value())
    {
        return this->trafos_site_to_image[key.value()];
    }
    else
    {
        return AffineTrafo2D();
    }
}

std::optional<Eigen::Array2d> TrafoManager::get_phase_ref_image()
{
    return this->phase_ref_image;
}

std::optional<Eigen::Array2d> TrafoManager::get_phase_ref_site()
{
    return this->phase_ref_site;
}


Eigen::VectorXd get_phase_from_trafo_site_to_image_py(py::object &trafoPy, std::optional<Eigen::VectorXd> phase_ref_image)
{
    return get_phase_from_trafo_site_to_image(AffineTrafo2D(trafoPy), phase_ref_image);
}

Eigen::VectorXd get_phase_from_trafo_site_to_image(AffineTrafo2D trafo_site_to_image, std::optional<Eigen::VectorXd> phase_ref_image)
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
    Eigen::VectorXd site_float = trafo_site_to_image.coord_to_origin(phase_ref_image.value());
    Eigen::ArrayXd phaseArray = site_float.array() + 0.5;
    //phaseArray = phaseArray.unaryExpr([](const float x) { return fmod(x, 1) - 0.5; });
    phaseArray -= phaseArray.cast<int>().cast<double>();
    phaseArray -= 0.5;
    return phaseArray.matrix();
}

AffineTrafo2D get_trafo_site_to_image_str(std::optional<std::string> trafo_site_to_image, 
    std::optional<Eigen::Array2d> magnification, std::optional<Eigen::Array2d> angle,
    std::optional<Eigen::Array2d> phase_ref_image, std::optional<Eigen::Array2d> phase_ref_site, 
    std::optional<Eigen::Array2d> phase)
{
    AffineTrafo2D trafo = TrafoManager::getInstance().get_trafo_site_to_image(trafo_site_to_image);
    return get_trafo_site_to_image(trafo, magnification, angle, phase_ref_image, phase_ref_site, phase);
}

AffineTrafo2D get_trafo_site_to_image(AffineTrafo2D trafo_site_to_image, 
    std::optional<Eigen::Array2d> magnification, std::optional<Eigen::Array2d> angle,
    std::optional<Eigen::Array2d> phase_ref_image, std::optional<Eigen::Array2d> phase_ref_site, 
    std::optional<Eigen::Array2d> phase)
{
    /*Gets the default transformation.

    Parameters
    ----------
    trafo_site_to_image : `AffineTrafo2d` or `str` or `None`
        Uses the origin axes from this trafo object.
        Is overwritten by `magnification` and `angle`.
        If `str`, loads a trafo from :py:class:`TrafoManager`.
    magnification, angle : `np.ndarray(1, float)` or `None`
        Origin axes magnification and angle.
    phase_ref_image, phase_ref_site : `np.ndarray(1, float)` or `None`
        Phase reference (phase = (0, 0)) in fluorescence image
        and lattice site coordinates.
    phase : `np.ndarray(float, float)`
        Lattice phase (fluo axes in site space = reference + phase).

    Returns
    -------
    trafo : `AffineTrafo2d`
        Site-to-fluorescence-image transformation.*/

    // Parse parameters
    auto [mT, aT, oT] = trafo_site_to_image.get_origin_axes();
    if(!magnification.has_value())
    {
        magnification.emplace(mT);
    }
    if(!angle.has_value())
    {
        angle.emplace(aT);
    }
    if(!phase_ref_image.has_value())
    {
        phase_ref_image = TrafoManager::getInstance().get_phase_ref_image().value();
    }
    if(!phase_ref_site.has_value())
    {
        phase_ref_site = TrafoManager::getInstance().get_phase_ref_site().value();
    }
    if(!phase.has_value())
    {
        phase = Eigen::Array2d();
        phase.value() << 0, 0;
    }
    // Setup trafo
    AffineTrafo2D trafo;
    trafo.set_origin_axes(magnification.value(), angle.value(), std::nullopt);
    auto [m, a, o] = trafo.get_target_axes();
    Eigen::Array2d offset = phase_ref_site.value() - trafo.coord_to_origin(phase_ref_image.value()).array();
    trafo.set_target_axes(m, a, offset + phase.value());
    return trafo;
}