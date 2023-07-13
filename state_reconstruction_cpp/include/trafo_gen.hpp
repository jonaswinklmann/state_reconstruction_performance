#include <Eigen/Dense>
#include <optional>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class AffineTrafo2D;
Eigen::VectorXd get_phase_from_trafo_site_to_image_py(py::object& trafoPy, std::optional<Eigen::VectorXd> phase_ref_image = std::nullopt);
Eigen::VectorXd get_phase_from_trafo_site_to_image(AffineTrafo2D trafo_site_to_image, std::optional<Eigen::VectorXd> phase_ref_image = std::nullopt);