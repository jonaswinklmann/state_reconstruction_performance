#include <Eigen/Dense>
#include <map>
#include <optional>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class AffineTrafo2D;
class TrafoManager
{
private:
    std::optional<Eigen::Array2d> phase_ref_image, phase_ref_site;
    std::map<std::string, AffineTrafo2D> trafos_site_to_image;
public:
    TrafoManager();
    static TrafoManager& getInstance()
    {
        static TrafoManager instance;
        return instance;
    };
    TrafoManager(TrafoManager const&) = delete;
    void operator=(TrafoManager const&) = delete;
    ~TrafoManager() {};
    void setPhaseRefImage(Eigen::Array2d phaseRefImage);
    void setPhaseRefSite(Eigen::Array2d phaseRefSite);
    void setTrafosSiteToImage(std::map<std::string, AffineTrafo2D> trafos);
    AffineTrafo2D get_trafo_site_to_image(std::optional<std::string> key);
    std::optional<Eigen::Array2d> get_phase_ref_image();
    std::optional<Eigen::Array2d> get_phase_ref_site();
};

Eigen::VectorXd get_phase_from_trafo_site_to_image_py(py::object& trafoPy, std::optional<Eigen::VectorXd> phase_ref_image = std::nullopt);
Eigen::VectorXd get_phase_from_trafo_site_to_image(AffineTrafo2D trafo_site_to_image, std::optional<Eigen::VectorXd> phase_ref_image = std::nullopt);
AffineTrafo2D get_trafo_site_to_image_str(std::optional<std::string> trafo_site_to_image, 
    std::optional<Eigen::Array2d> magnification, std::optional<Eigen::Array2d> angle,
    std::optional<Eigen::Array2d> phase_ref_image, std::optional<Eigen::Array2d> phase_ref_site, 
    std::optional<Eigen::Array2d> phase);
AffineTrafo2D get_trafo_site_to_image(AffineTrafo2D trafo_site_to_image, 
    std::optional<Eigen::Array2d> magnification = std::nullopt, std::optional<Eigen::Array2d> angle = std::nullopt,
    std::optional<Eigen::Array2d> phase_ref_image = std::nullopt, std::optional<Eigen::Array2d> phase_ref_site = std::nullopt, 
    std::optional<Eigen::Array2d> phase = std::nullopt);