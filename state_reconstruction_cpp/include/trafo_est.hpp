#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <optional>
#include <vector>
#include "Eigen/Dense"

namespace py = pybind11;

class TrafoEstimator
{
public:
    Eigen::VectorXd get_trafo_phase_from_projections(
        const py::EigenDRef<const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>& im, py::object& prjgen, 
        Eigen::Array2d phase_ref_image = Eigen::Array2d::Zero(), Eigen::Array2d phase_ref_site = Eigen::Array2d::Zero(),
        std::optional<std::vector<int>> subimage_shape = std::nullopt, std::optional<std::vector<int>> subsite_shape = std::nullopt, int search_range = 1);
};