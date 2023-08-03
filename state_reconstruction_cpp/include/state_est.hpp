#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <optional>
#include <vector>

namespace py = pybind11;

class AffineTrafo2D;
class StateEstimator
{
public:
    std::vector<double> constructLocalImagesAndApplyProjectors(
        const py::EigenDRef<const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>& image,
        Eigen::Array2i sitesShape, py::object& trafoPy, Eigen::Array2i projShape, int psfSupersample, py::object& projector_generator,
        py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>& emissions);
    void init();
    void loadProj(py::object& prjgen);
};