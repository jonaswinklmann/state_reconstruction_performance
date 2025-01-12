#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>
#include "pybind11/eigen.h"

#include "data_types.hpp"

namespace py = pybind11;

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

std::vector<Image> getLocalImages(std::vector<Eigen::Vector2d> coords, 
    const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& fullImage, Eigen::Array2i shape, int psf_supersample=5);
std::vector<double> apply_projectors(std::vector<Image>& localImages, py::object& projector_generator);