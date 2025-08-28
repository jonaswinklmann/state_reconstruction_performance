#include <Eigen/Dense>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <vector>

namespace py = pybind11;

class ImagePreprocessor
{
    /*Class for preprocessing raw images.

    Checks for and removes image outliers.

    Parameters
    ----------
    scale : `Array[2, float]` or `str`
        Image amplitude prescaling.
        Must have same shape as images to be processed.
        If `str`, `scale` is interpreted as file path from which the image
        is loaded.
    outlier_size : `(int, int)`
        Area around outlier over which the outlier is analyzed and removed.
    max_outlier_ratio : `float`
        Maximum accepted ratio between outlier and non-outlier maximum.
    outlier_min_ref_val : `float`
        Minimum reference value to be considered valid outlier.
    outlier_iterations : `int`
        Maximum number of outlier removal iterations.*/
public:
    ImagePreprocessor(const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& scale,
        std::optional<Eigen::Array2i> outlier_size = std::nullopt, double max_outlier_ratio = 5, double outlier_min_ref_val = 5,
        int outlier_iterations = 2) : scale(scale), outlier_size(), max_outlier_ratio(max_outlier_ratio), 
            outlier_min_ref_val(outlier_min_ref_val), outlier_iterations(outlier_iterations)
    {
        if(!outlier_size.has_value())
        {
            this->outlier_size << 5, 5;
        }
        else
        {
            this->outlier_size = outlier_size.value();
        }
    };
    ImagePreprocessor(std::optional<Eigen::Array2i> outlier_size = std::nullopt, double max_outlier_ratio = 5, double outlier_min_ref_val = 5,
        int outlier_iterations = 2) : scale(std::nullopt), outlier_size(), 
        max_outlier_ratio(max_outlier_ratio), outlier_min_ref_val(outlier_min_ref_val), outlier_iterations(outlier_iterations)
    {
        if(!outlier_size.has_value())
        {
            this->outlier_size << 5, 5;
        }
        else
        {
            this->outlier_size = outlier_size.value();
        }
    };
    Eigen::Array2d process_image(py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> image) const;
private:
    std::optional<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> scale;
    Eigen::Array2i outlier_size;
    double max_outlier_ratio, outlier_min_ref_val;
    int outlier_iterations;
};