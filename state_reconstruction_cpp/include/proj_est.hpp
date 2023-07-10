#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Image
{
    Eigen::Array<double, -1, -1, Eigen::RowMajor> image;
    // Rounded PSF center coordinates.
    int X_int, Y_int;
    // Rounded PSF rectangle corners.
    int X_min, X_max, Y_min, Y_max;
    // Subpixel shifts.
    int dx, dy;
};

std::vector<Image> getLocalImages(std::vector<Eigen::Vector2f> coords, 
    const Eigen::Array<double,-1,-1,Eigen::RowMajor>& image, Eigen::Array2i shape, int psf_supersample=5);
std::vector<float> apply_projectors(std::vector<Image> localImages, py::object& projector_generator);