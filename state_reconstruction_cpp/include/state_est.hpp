#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <optional>
#include <vector>
#include <tuple>

#include "image_est.hpp"

namespace py = pybind11;

class AffineTrafo2D;
class StateEstimator
{
public:
    StateEstimator();
    std::vector<double> constructLocalImagesAndApplyProjectorsPyTrafo(
        const py::EigenDRef<const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>& image,
        Eigen::Array2i sitesShape, py::object& trafoPy, Eigen::Array2i projShape, int psfSupersample, py::object& projector_generator,
        py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> emissions);
    std::vector<double> constructLocalImagesAndApplyProjectors(
        const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &image,
        Eigen::Array2i sitesShape, AffineTrafo2D &trafo, Eigen::Array2i projShape, int psfSupersample, py::object &projector_generator,
        py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> emissions);
    void init();
    void loadProj(py::object& prjgen);
    void setImagePreProcScale(const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& scale,
        std::optional<Eigen::Array2i> outlier_size = std::nullopt, double max_outlier_ratio = 5, double outlier_min_ref_val = 5,
        int outlier_iterations = 2);
    void setImagePreProc(std::optional<Eigen::Array2i> outlier_size = std::nullopt, 
        double max_outlier_ratio = 5, double outlier_min_ref_val = 5, int outlier_iterations = 2);
    std::tuple<std::vector<double>, Eigen::VectorXd, Eigen::Array2d, Eigen::MatrixXd, Eigen::VectorXd> reconstruct(
        py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> image, 
        std::optional<AffineTrafo2D> new_trafo, Eigen::Array2i sitesShape, Eigen::Array2i projShape, int psfSupersample, 
        py::object &projector_generator, Eigen::Array2d phase_ref_image, Eigen::Array2d phase_ref_site,
        py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> emissions, py::object &trafoSiteToImage);
private:
    std::optional<ImagePreprocessor> imgPreproc;
};