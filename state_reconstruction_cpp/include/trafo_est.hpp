#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <optional>
#include <vector>
#include "Eigen/Dense"

namespace py = pybind11;

class TrafoEstimator
{
public:
    Eigen::VectorXf get_trafo_phase_from_projections(const py::EigenDRef<const Eigen::Array<double,-1,-1,Eigen::RowMajor>>& im, py::object& prjgen, 
        std::vector<int> phase_ref_image={0,0}, std::vector<int> phase_ref_site={0,0},
        std::optional<std::vector<int>> subimage_shape = std::nullopt, std::optional<std::vector<int>> subsite_shape = std::nullopt, int search_range = 1);
};