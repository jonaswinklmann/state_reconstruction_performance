#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

struct Image
{
    Eigen::Map<const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>,0,StrideDyn> image;
    // Rounded PSF center coordinates.
    int X_int, Y_int;
    // Rounded PSF rectangle corners.
    int X_min, X_max, Y_min, Y_max;
    // Subpixel shifts.
    int dx, dy;
};

std::vector<Image> getLocalImages(std::vector<Eigen::Vector2d> coords, 
    const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& fullImage, Eigen::Array2i shape, int psf_supersample=5);
std::vector<double> apply_projectors(std::vector<Image> localImages, py::object& projector_generator);
std::vector<double> apply_projectors_gpu(std::vector<Image> localImages, py::object& projector_generator);